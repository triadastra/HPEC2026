"""Validate the staged publication and reconcile remote run directories."""
import json
import shutil
from pathlib import Path

from src.utils.run_manifest import validate_completion


SESSION_NAMES = ("exp0", "exp0_fa")


def publication_runs(out):
    """Return valid relative run paths and invalid directories with reasons.

    Completion hashes alone do not establish membership in an experiment.
    Every run must belong to its own session's manifest, including nested probes.
    """
    out = Path(out)
    valid, invalid = set(), []
    for session in [out] + [out / name for name in SESSION_NAMES]:
        if not session.is_dir():
            continue
        manifest = session / "manifest.json"
        # A missing declaration is a failed publication prerequisite, not an
        # empty experiment that should erase previously published checkpoints.
        entries = {e["name"]: e for e in json.loads(manifest.read_text())["runs"]}
        for run in sorted(session.iterdir()):
            if not run.is_dir() or (session == out and run.name in SESSION_NAMES):
                continue
            entry = entries.get(run.name)
            reason = ("not declared in session manifest" if entry is None
                      else validate_completion(session, entry))
            if reason:
                invalid.append((run, reason))
            else:
                valid.add(run.relative_to(out).as_posix())
    return valid, invalid


def drop_invalid_runs(out):
    valid, invalid = publication_runs(out)
    for run, _ in invalid:
        shutil.rmtree(run)
    return valid, invalid


def withdrawn_remote_files(remote_files, valid_runs):
    """Remove remote runs absent from the current validated publication.

    A withdrawn run may still have BOTH its old checkpoint and old completion
    marker remotely. Presence of those two filenames cannot prove currency.
    Nested session paths are preserved so names in different probes cannot collide.
    """
    remote_runs = {p.rsplit("/", 1)[0] for p in remote_files
                   if "/" in p and p.rsplit("/", 1)[1] in
                   {"best.pth", "run_complete.json"}}
    withdrawn = remote_runs - set(valid_runs)
    return [p for p in remote_files
            if any(p.startswith(run + "/") for run in withdrawn)]
