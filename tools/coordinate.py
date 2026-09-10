#!/usr/bin/env python3
"""Coordinate the sharded sweep from this machine.

Two GPU boxes each run a disjoint half of the declared matrix into their own
runs-dir. Neither can produce an official leaderboard alone -- evaluate.py
requires one directory whose manifest declares every cell it scores. This
pulls both halves here and merges them.

  status   how far each box has got
  pull     copy results here (evidence only unless --checkpoints)
  merge    build one runs-dir from both halves
  verify   check the merged manifest declares the full matrix, once

Evidence is the per-epoch curves, the completion records, the cost rows and
the manifest: about 15 KB a run. Checkpoints are ~2.6 MB each and are only
needed to score, so they are opt-in.
"""
import argparse, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BOXES = {
    # Rebalanced 2026-08-28: box B took test 1 while it sat idle waiting for
    # the mech probe, so test 1's 18 cells moved off box A. Still 477 total.
    "A": dict(port=os.environ.get("HPEC_PORT_A", "22"), pw_env="VMPW_A", tests="2,3,6", cards=5, declared=243),
    "B": dict(port=os.environ.get("HPEC_PORT_B", "22"), pw_env="VMPW_B", tests="1,1.1,4", cards=4, declared=234),
}
HOST = os.environ.get("HPEC_SSH_HOST", "configure-host.invalid")
REMOTE = os.environ.get("HPEC_REMOTE_SWEEP", "~/HPEC2026/outputs/sweep")
LOCAL = REPO / "outputs" / "merged"

# Runs get relaunched a few at a time, so a healthy pull mirrors a handful of
# deletions. Hundreds means the remote lost its outputs, not that we finished.
DELETE_LIMIT = 25

# Everything a run writes except the checkpoint. The per-run .log carries the
# realized step count and the measured FLOPs, and the loss curve is the figure
# for that cell -- both are a few KB, and both are irrecoverable once the
# instance is released, so they travel with the evidence rather than with the
# opt-in checkpoints.
EVIDENCE = ["--include=*/", "--include=manifest.json", "--include=logs/metrics.json",
            "--include=logs/*.log", "--include=plots/*.png",
            "--include=run_complete.json", "--include=run_diverged.json",
            "--include=cost.json", "--include=model_session.json",
            "--include=config_resolved.yaml", "--include=run_fingerprint.json",
            # Exp 0's whole output. It sets the learning rate of every run in
            # the matrix, so losing it means the rates in the paper have no
            # artifact behind them -- regenerable from the Exp 0 runs, but only
            # while those still exist somewhere.
            "--include=lr_selection*.json", "--include=lr_diagnostics*.json",
            "--exclude=*"]
WITH_CKPT = EVIDENCE[:-1] + ["--include=best.pth", "--exclude=*"]


def _short(path):
    """Repo-relative when it is under the repo, absolute otherwise -- the
    output directory is configurable, and a path outside the repo should print,
    not raise."""
    try:
        return path.relative_to(REPO)
    except ValueError:
        return path


def sh(box, cmd, capture=True):
    b = BOXES[box]
    pw = os.environ.get(b["pw_env"])
    if not pw:
        sys.exit(f"set {b['pw_env']} to box {box}'s password")
    full = ["sshpass", "-e", "ssh", "-p", b["port"], "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=25", HOST, cmd]
    env = {**os.environ, "SSHPASS": pw}
    # AutoDL drops concurrent connections; a single refusal is not an outage.
    for _ in range(4):
        r = subprocess.run(full, capture_output=capture, text=True, env=env)
        if r.returncode == 0:
            return r.stdout.strip() if capture else ""
    return ""


def status(args):
    total_done = 0
    for name, b in BOXES.items():
        # Count main-matrix runs only. exp0/ and exp0_fa/ are separate
        # sessions living under the same root, and logs/ is not a run.
        # mkdir -p, not cd-and-fail: a box that has not started its share yet
        # has no sweep directory, and "no runs" is not "no box". Conflating the
        # two reported a live machine as unreachable.
        out = sh(name, f"mkdir -p {REMOTE}; cd {REMOTE}; "
                       f"echo \"$(ls */run_complete.json 2>/dev/null | grep -vc '^exp0') "
                       f"$(ls -d */ 2>/dev/null | grep -v '^exp0' | grep -vc '^logs/') "
                       f"$(ls */run_diverged.json 2>/dev/null | grep -vc '^exp0') "
                       f"$(pgrep -fc 'scripts/train.py' 2>/dev/null||echo 0)\"; true")
        try:
            done, started, div, train = (int(x) for x in out.split())
        except ValueError:
            print(f"  box {name}: no answer (ssh refused four times)"); continue
        total_done += done
        pct = 100 * done / b["declared"]
        print(f"  box {name} ({b['cards']} cards, tests {b['tests']:8}): "
              f"{done:4d}/{b['declared']:<4} {pct:5.1f}%  started={started:<4} "
              f"diverged={div:<3} training={train}")
    print(f"  ---- combined: {total_done}/477 ({100*total_done/477:.1f}%)")


def pull(args):
    LOCAL.mkdir(parents=True, exist_ok=True)
    flags = WITH_CKPT if args.checkpoints else EVIDENCE
    for name, b in BOXES.items():
        dest = LOCAL / f"box{name}"
        dest.mkdir(exist_ok=True)
        pw = os.environ.get(b["pw_env"])
        # --delete matters more than it looks. When a run is relaunched,
        # scripts/sweep.py removes its remote run_complete.json before
        # training; without mirroring that deletion, a rerun that then fails
        # or diverges leaves the PREVIOUS completion record sitting next to the
        # previous checkpoint, still passing provenance validation -- so the
        # coordinator would score the old run as the current result.
        #
        # Excluded files are protected from --delete by default, so an
        # evidence-only pull still cannot remove checkpoints it deliberately
        # did not fetch.
        cmd = ["sshpass", "-e", "rsync", "-az", "--partial", "--delete",
               "-e", f"ssh -p {b['port']} -o StrictHostKeyChecking=no",
               *flags, f"{HOST}:{REMOTE}/", str(dest) + "/"]
        print(f"  pulling box {name} -> {_short(dest)} ...", flush=True)
        env = {**os.environ, "SSHPASS": pw}

        # Mirroring deletions is only safe while the remote still looks like a
        # sweep. AutoDL instances get released and recreated -- a box whose
        # rental lapsed can come back with that path present but empty, and
        # mirroring emptiness would delete the local copy, which by then is the
        # only copy. So price the deletions before committing to them.
        dry = subprocess.run(cmd + ["--dry-run", "-i"], env=env,
                             capture_output=True, text=True)
        if dry.returncode == 0:
            doomed = [ln.split(None, 1)[1] for ln in dry.stdout.splitlines()
                      if ln.startswith("*deleting")]
            if len(doomed) > DELETE_LIMIT and not args.force_delete:
                for f in doomed[:10]:
                    print(f"    would delete {f}")
                if len(doomed) > 10:
                    print(f"    ... and {len(doomed) - 10} more")
                sys.exit(
                    f"box {name}: the remote is missing {len(doomed)} files that "
                    f"are here locally (limit {DELETE_LIMIT}).\n"
                    "That is what a released/recreated instance looks like, and "
                    "this local copy may be the only one left.\n"
                    "Check the box, then re-run with --force-delete if the "
                    "deletions are genuinely intended.")

        for attempt in range(1, 4):
            r = subprocess.run(cmd, env=env)
            if r.returncode == 0:
                break
            print(f"    rsync failed (rc={r.returncode}), attempt {attempt}/3")
        else:
            # Reporting the count already on disk here would make a failed
            # transfer read as a successful refresh, and let merge/verify run
            # on stale artifacts.
            sys.exit(f"box {name}: rsync failed three times -- nothing pulled, "
                     "not continuing to merge on stale data")
        n = len(list(dest.rglob("run_complete.json")))
        print(f"    {n} completed runs on disk")


def merge(args):
    """One runs-dir from both halves. The two declarations are disjoint by
    construction, so the merged manifest is their concatenation -- but that is
    asserted, not assumed: an overlap would mean a cell was trained twice under
    two different declarations."""
    out = LOCAL / "sweep"
    out.mkdir(parents=True, exist_ok=True)
    entries, seen = [], {}
    for name in BOXES:
        src = LOCAL / f"box{name}" / "manifest.json"
        if not src.exists():
            print(f"  box {name}: no manifest pulled yet"); continue
        data = json.loads(src.read_text())
        runs = data.get("runs", data)
        for e in runs:
            # Overlapping DECLARATIONS are legitimate: when box B finishes its
            # shard early it picks up a slice of box A's remaining runs, and
            # both boxes then declare the same test. What must never overlap is
            # COMPLETION -- the same cell trained twice, under two provenances,
            # with no way to say which one the leaderboard reports.
            done = (LOCAL / f"box{name}" / e["name"] / "run_complete.json").exists()
            prev = seen.get(e["name"])
            if prev is not None:
                if prev["done"] and done:
                    sys.exit(f"{e['name']} was completed on BOTH boxes -- "
                             "two provenances for one cell, refusing to pick")
                if done:                      # the finished side wins
                    entries[prev["index"]] = e
                    seen[e["name"]] = {"box": name, "done": True,
                                       "index": prev["index"]}
                continue                      # otherwise keep what we had
            seen[e["name"]] = {"box": name, "done": done, "index": len(entries)}
            entries.append(e)
    # Ownership is settled above, so link each run in from exactly one box.
    # Doing this inside the per-box loop was wrong once declarations could
    # overlap: box B's empty directory for a cell box A had finished came
    # second and, by the refresh rule below, deleted box A's results.
    #
    # Existence is still NOT proof of currency: an evidence-only pull followed
    # later by --checkpoints adds best.pth under boxA/boxB only, and skipping a
    # directory that already exists would leave those checkpoints out of the
    # merged tree.
    for run_name, owner in seen.items():
        d = LOCAL / f"box{owner['box']}" / run_name
        target = out / run_name
        if not d.is_dir():
            if target.is_dir():
                shutil.rmtree(target)
            continue
        target.mkdir(parents=True, exist_ok=True)
        for src_file in d.rglob("*"):
            if not src_file.is_file():
                continue
            dst = target / src_file.relative_to(d)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                if dst.stat().st_ino == src_file.stat().st_ino:
                    continue              # already the same hard link
                dst.unlink()
            os.link(src_file, dst)
        # and drop anything the owning source no longer has, so a cleared
        # completion record does not survive here either
        for dst_file in list(target.rglob("*")):
            if dst_file.is_file() and not (d / dst_file.relative_to(target)).exists():
                dst_file.unlink()

    (out / "manifest.json").write_text(json.dumps(
        {"schema_version": 3, "runs": entries}, indent=2))
    print(f"  merged manifest: {len(entries)} declared "
          f"(A={sum(1 for v in seen.values() if v['box']=='A')}, "
          f"B={sum(1 for v in seen.values() if v['box']=='B')})")
    print(f"  runs on disk:    {len(list(out.rglob('run_complete.json')))} complete")
    print(f"  -> {_short(out)}")


def verify(args):
    sys.path.insert(0, str(REPO))
    from scripts.sweep import build_matrix, DEFAULT_SEEDS
    full = {n for n, _ in build_matrix({"1", "1.1", "2", "3", "4", "6"},
                                       DEFAULT_SEEDS, 2048, 1, accum=15,
                                       effective_batch_size=30087)}
    man = LOCAL / "sweep" / "manifest.json"
    if not man.exists():
        sys.exit("nothing merged yet -- run pull then merge")
    declared = {e["name"] for e in json.loads(man.read_text())["runs"]}
    missing, extra = full - declared, declared - full
    print(f"  declared {len(declared)} | expected {len(full)}")
    print(f"  missing from the merge: {len(missing)}")
    for n in sorted(missing)[:5]:
        print(f"    {n}")
    print(f"  not in the declared matrix: {len(extra)}")
    completes = list((LOCAL / "sweep").rglob("run_complete.json"))
    done = len(completes)
    # rglob also finds the nested exp0/exp0_fa probe runs, so readiness must
    # count only completions whose name is in the declared matrix.
    matrix_done = {rc.parent.name for rc in completes} & full
    print(f"  complete on disk: {len(matrix_done)}/{len(full)} matrix runs"
          f" (+{done - len(matrix_done)} nested probe runs)")

    # A run counted as complete still has to HOLD what the paper needs. Check
    # the artifacts rather than the count, and check the checkpoint against the
    # hash the run recorded for it -- a truncated or half-synced best.pth is
    # otherwise indistinguishable from a good one.
    want = {"logs/metrics.json": "metrics", "cost.json": "flops",
            "best.pth": "checkpoint"}
    absent = {k: [] for k in want.values()}
    logless, unhashed, corrupt = [], [], []
    for rc in completes:
        d = rc.parent
        for rel, label in want.items():
            if not (d / rel).exists():
                absent[label].append(d.name)
        if not list((d / "logs").glob("*.log")):
            logless.append(d.name)
        ck = d / "best.pth"
        try:
            recorded = json.loads(rc.read_text()).get("checkpoint_sha256")
        except Exception:
            recorded = None
        if not recorded:
            unhashed.append(d.name)
        elif ck.exists():
            h = hashlib.sha256(ck.read_bytes()).hexdigest()
            if h != recorded:
                corrupt.append(d.name)

    print()
    print("  artifacts held by the completed runs")
    for label in ("metrics", "flops", "checkpoint"):
        miss = absent[label]
        mark = "ok " if not miss else "MISSING"
        print(f"    {label:<12} {done - len(miss):>4}/{done:<4} {mark}"
              + (f"  e.g. {miss[0]}" if miss else ""))
    print(f"    {'train log':<12} {done - len(logless):>4}/{done:<4} "
          + ("ok " if not logless else f"MISSING  e.g. {logless[0]}"))
    verified = done - len(absent["checkpoint"]) - len(unhashed) - len(corrupt)
    print(f"    {'sha256 match':<12} {verified:>4}/{done:<4} "
          + ("ok" if not corrupt and not unhashed else
             f"CORRUPT={len(corrupt)} UNHASHED={len(unhashed)}"))
    for n in (corrupt + unhashed)[:5]:
        print(f"      ! {n}")

    intact = not any(absent.values()) and not logless and not corrupt and not unhashed
    print()
    if not missing and not extra and len(matrix_done) == len(full) and intact:
        print("  READY: the merged directory declares and holds the full matrix,")
        print("         and every checkpoint matches the hash its run recorded")
    elif not intact:
        print("  NOT READY: some completed runs are missing artifacts above")
    else:
        print("  NOT READY: the merge does not yet declare and hold the full matrix")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    p = sub.add_parser("pull")
    p.add_argument("--checkpoints", action="store_true",
                   help="also pull best.pth (~2.6 MB a run)")
    p.add_argument("--force-delete", action="store_true",
                   help=f"mirror deletions even past {DELETE_LIMIT} files "
                        "(a released instance looks exactly like this)")
    sub.add_parser("merge")
    sub.add_parser("verify")
    a = ap.parse_args()
    {"status": status, "pull": pull, "merge": merge, "verify": verify}[a.cmd](a)
