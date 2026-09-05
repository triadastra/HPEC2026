#!/bin/bash
# Pull both boxes, merge them into one scoreable tree, and publish the trained
# outputs to the public model repo.
#
#   VMPW_A=... VMPW_B=... tools/publish_checkpoints.sh [--final]
#
# Scope is outputs/merged/sweep -- checkpoints and their provenance. NOT the
# repo: it is private, and the model repo is public.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
HF_REPO=Celsia/HPEC2026
FINAL=${1:-}
cd "$REPO" || exit 1

echo "[pub] pulling both boxes"
VMPW_A="${VMPW_A:?}" VMPW_B="${VMPW_B:?}" python3 tools/coordinate.py pull --checkpoints 2>&1 | tail -5

echo "[pub] merging"
VMPW_A=x VMPW_B=x python3 tools/coordinate.py merge 2>&1 | tail -4
OUT=outputs/merged/sweep
[ -d "$OUT" ] || { echo "[pub] no merged tree"; exit 1; }

# Publish only runs that are provenance-complete. The merged tree also holds
# directories that were mid-flight when the pull ran: they have a best.pth and
# no completion record, so nothing binds those bytes to a declaration. The
# model card promises every published checkpoint is validated, and a page that
# promises that has to mean it.
# Exp 0 lives in its own session directories and never enters the 477-run
# matrix, so merge does not see it. Those checkpoints are still results -- the
# learning rate every run in the paper trains at was chosen from them, and the
# fa arm's rates in particular reversed a conclusion. Stage them alongside,
# under their own prefix, validated the same way.
echo "[pub] staging the Exp 0 sessions"
python3 - "$OUT" <<'PY_STAGE'
import os
from pathlib import Path
import shutil
import sys
out = Path(sys.argv[1])
for name in ("exp0", "exp0_fa"):
    source = out.parent / "boxA" / name
    target = out / name
    if target.exists():
        shutil.rmtree(target)
    if not source.is_dir():
        continue
    # Portable hard links; failures propagate instead of find -exec hiding a
    # failed cp. Refresh the whole probe so withdrawn runs cannot survive.
    shutil.copytree(source, target, copy_function=os.link)
    print(f"    {name}: {len(list(target.rglob('best.pth')))} checkpoints")
PY_STAGE

echo "[pub] dropping runs without a valid completion record"
python3 - "$OUT" <<'PY'
import sys
from tools.publication import drop_invalid_runs
valid, dropped = drop_invalid_runs(sys.argv[1])
for path, why in dropped[:8]:
    print(f"    dropped {path.name}: {why}")
print(f"    {len(dropped)} run(s) held back, {len(valid)} validated")
PY

echo "[pub] inventory of what will be public"
python3 - "$OUT" <<'PY'
import sys, collections
from pathlib import Path
out = Path(sys.argv[1])
def kind(n):
    if n.startswith("exp0_mech"): return "exp0 mech probe (fa rates)"
    if n.startswith("exp0_"):     return "exp0 base (LR search)"
    if "_aggregate_roll_" in n:   return "test 1.1 rolling origin"
    if "_aggregate_s" in n:       return "test 1 fixed split"
    if "_id_" in n:               return "test 4 identity axial"
    return "tests 2/3/6 N-D grid"
c = collections.Counter(kind(p.parent.name) for p in out.rglob("best.pth"))
for k, v in sorted(c.items()):
    print(f"    {k:30s} {v:4d}")
print(f"    {'TOTAL':30s} {sum(c.values()):4d} checkpoints")
PY

# Refuse to publish anything that is not a training output. The repo carries
# rented-box credentials in its git history; this tree must never pick up a
# stray script or config on its way to a public model page.
STRAY=$(find "$OUT" -type f ! -name best.pth ! -name '*.json' ! -name '*.log' \
        ! -name '*.png' ! -name README.md ! -name .gitattributes | sed -n '1,5p')
if [ -n "$STRAY" ]; then
  echo "[pub] REFUSING -- unexpected file types in the tree:"; echo "$STRAY"; exit 1
fi

MSG="Checkpoints: $(find "$OUT" -name best.pth | wc -l | tr -d ' ') runs"
[ "$FINAL" = "--final" ] && MSG="$MSG (complete matrix)" || MSG="$MSG (sweep in progress)"
# A run held back here must also leave the repo. hf upload only adds, so a
# checkpoint published while it was still valid would outlive the record that
# vouched for it -- and the model card says every published checkpoint is
# validated.
#
# Deleting by glob does not work: `name/**` left six runs behind, each with a
# best.pth and no run_complete.json. Ask the repo what it actually holds and
# delete those paths by name.
echo "[pub] pruning unvalidated runs already on the repo"
python3 - "$HF_REPO" "$OUT" <<'PY'
import sys
from huggingface_hub import HfApi
from tools.publication import publication_runs, withdrawn_remote_files
repo, out = sys.argv[1:]
valid, invalid = publication_runs(out)
if invalid:
    raise RuntimeError("publication changed after validation; refusing to prune")
api = HfApi()
# Failure to inspect the remote must stop publication. Otherwise withdrawals
# would silently remain public while new results were uploaded.
siblings = api.list_repo_files(repo_id=repo, repo_type="model")
victims = withdrawn_remote_files(siblings, valid)
if victims:
    print(f"    pruning {len(victims)} files from withdrawn runs")
    api.delete_files(
        repo_id=repo, repo_type="model", delete_patterns=victims,
        commit_message="Remove runs absent from the validated publication",
    )
else:
    print("    nothing to prune")
PY

echo "[pub] uploading: $MSG"
hf upload "$HF_REPO" "$OUT" . --repo-type=model --commit-message "$MSG" 2>&1 | tail -6
echo "[pub] https://huggingface.co/$HF_REPO"
