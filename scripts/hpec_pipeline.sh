#!/usr/bin/env bash
# =============================================================================
# HPEC CENSUS PIPELINE: train -> eval -> cost, in one command.
#
# Before this, the census benchmark was training only (scripts/sweep.py), with
# evaluation and the cost panel run by hand afterwards. This closes that gap and
# adds the three controls a long retrain needs: resume, selective redo, and a
# graceful stop.
#
# scripts/evaluate.py IS the reporting stage now: it writes the leaderboard,
# the fairness subgroup tables and the disparity summary, and it refuses to
# write an official leaderboard while the sweep manifest is incomplete (pass
# ALLOW_PARTIAL=1 for an explicitly-marked partial run). The old
# build_paper_results.py / collect_results.py / run_all_experiments.sh are
# been removed from the repository.
#
#   # THE RETRAIN, in the order the protocol requires: Exp 0 picks a learning
#   # rate per (arm, model) on validation loss, then the matrix runs on those
#   # rates. Without the exp0 stage every run falls back to a config default
#   # that was never searched for the multidimensional arm at all.
#   GPUS=0,1,2,3 STAGES=exp0 bash scripts/hpec_pipeline.sh
#   GPUS=0,1,2,3 TESTS=1,1.1,2,3,4,6 RESET_MANIFEST=1 bash scripts/hpec_pipeline.sh
#
#   # diagnostic: just the runs one finding invalidated (NOT an official
#   # leaderboard -- strict evaluation will refuse an incomplete manifest)
#   GPUS=0,1 RETRAIN=f3 STAGES=train bash scripts/hpec_pipeline.sh
#
#   # what state is every run in?
#   bash scripts/hpec_pipeline.sh --status
#
#   # eval + cost only, over checkpoints that already exist
#   STAGES=eval,cost bash scripts/hpec_pipeline.sh
#
#   # halt cleanly: in-flight runs finish, nothing new launches
#   touch <session-directory>/STOP  # printed at startup; delete it to continue
#
# STAGES  (default: train,eval,cost)
#   exp0    sweep.py --tests 0 + select_lr.py   learning-rate selection (run
#           FIRST, in its own directory; writes lr_selection.json)
#   train   scripts/sweep.py        run matrix, one run per GPU
#   eval    scripts/evaluate.py     leaderboard + baselines/MASE + fairness tables
#   cost    scripts/model_cost.py   params / FLOPs panel
#   xgb     scripts/xgb_agg.py      Test-1 tabular aggregate arm -- opt-in
#   errors  scripts/dump_errors.py  paired error dump -- opt-in, needs ERRDUMP_RUNS
#   sig     scripts/significance_tests.py   DM / Wilcoxon / paired t, plus TOST
#           equivalence and non-inferiority (needs errors). The hypothesis is
#           H0: ND == flat, so a non-significant difference test is NOT
#           evidence of equality -- TOST is what can support it. Tune with
#           DELTA_MODE / DELTA / ALPHA.
#
# RESUME
#   Every invocation enters the requested stage. The training stage is cheap to
#   resume because sweep.py skips only checkpoints with a matching provenance
#   completion record. Evaluation and cost are regenerated so timestamp-only
#   stage markers can never return stale outputs.
#
# SESSIONS
#   RESET_MANIFEST=1 creates outputs/sweep/sessions/<SESSION_ID>/ and records it
#   in outputs/sweep/current_session. A stopped run remains isolated and can be
#   resumed by rerunning without RESET_MANIFEST. Older sessions are untouched.
#
# SELECTIVE REDO (regex over run names; names carry model/variant/encoder/seed)
#   RETRAIN=f1          every flat run              (step-budget change)
#   RETRAIN=f2          mamba_nd 2d/3d              (leftover encoder added)
#   RETRAIN=f3          every Mamba-3 run           (head count corrected)
#   RETRAIN=mandatory   all of the above (99 diagnostic runs; not official)
#   ONLY='^gru_'        anything you can match by name
#   FORCE=1             re-train even runs that already finished
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

PY="${PY:-python}"
# Unbuffered, or a multi-hour sweep's progress log stays empty until the stage
# exits. Python block-buffers stdout when it is a file rather than a terminal,
# so `nohup ... > sweep.log` -- the documented way to run this -- shows nothing
# while every GPU is busy, and there is no way to tell a stalled sweep from a
# working one without reading the per-run logs.
export PYTHONUNBUFFERED=1
STAGES="${STAGES:-train,eval,cost}"
GPUS="${GPUS:-}"
TESTS="${TESTS:-2,3}"
SEEDS="${SEEDS:-}"
EPOCHS="${EPOCHS:-200}"   # matches sweep.py's default cap
FLAT_BS="${FLAT_BS:-2048}"
COMBO_BS="${COMBO_BS:-1}"
AGGREGATE_BS="${AGGREGATE_BS:-32}"
EXP0_DIR="${EXP0_DIR:-}"          # resolved under SWEEP_ROOT below
EXP0_EPOCHS="${EXP0_EPOCHS:-20}"   # LR ranking settles long before the 200 cap
EXP0_MODELS="${EXP0_MODELS:-}"     # restrict what Exp 0 RUNS (split envs)
EXP0_WINDOW="${EXP0_WINDOW:-}"     # select on epochs [0, window)
# Learning rates for the main matrix. Left empty, the exp0 stage's output is
# used when it exists -- forgetting the flag would otherwise silently revert
# the whole sweep to unsearched config defaults, which is precisely the
# asymmetry Exp 0 exists to remove, and nothing downstream would say so.
LR_SELECTION="${LR_SELECTION:-}"
# A supplementary probe's selection, merged OVER LR_SELECTION. Additive: the
# base file stays the audit record of its own session and is never rewritten.
# Defaults to the fa probe's output when that session exists.
LR_SELECTION_PATCH="${LR_SELECTION_PATCH:-}"
# DataLoader workers. Infrastructure, not protocol: worker COUNT provably does
# not change what is trained (verified in tests/test_combo_loader_equivalence),
# so it rides in the environment and stays out of run_fingerprint -- retuning it
# must not invalidate finished runs. The two paths have opposite appetites: the
# flat arm calls __getitem__ millions of times an epoch, the combo arm ~96.
NUM_WORKERS="${NUM_WORKERS:-}"
COMBO_NUM_WORKERS="${COMBO_NUM_WORKERS:-}"
# TOST equivalence margin for the sig stage.
DELTA_MODE="${DELTA_MODE:-frac}"
DELTA="${DELTA:-0.05}"
ALPHA="${ALPHA:-0.05}"
NPZ="${NPZ:-data/census_port/processed/census_lattice_9ch.npz}"
SERIES="${SERIES:-}"
HYBRID_IDENTITY="${HYBRID_IDENTITY:-0}"  # opt-in identity-aware hybrid arm (+72 runs)
RETRAIN="${RETRAIN:-}"
ONLY="${ONLY:-}"
EXCLUDE="${EXCLUDE:-}"
FORCE="${FORCE:-0}"
NO_STEP_MATCH="${NO_STEP_MATCH:-0}"
RESET_MANIFEST="${RESET_MANIFEST:-0}"   # start a fresh declared run matrix
ALLOW_PARTIAL="${ALLOW_PARTIAL:-0}"     # let evaluate.py write a PARTIAL leaderboard
EVAL_SEEDS="${EVAL_SEEDS:-}"            # required seeds for strict evaluation
ERRDUMP_RUNS="${ERRDUMP_RUNS:-}"        # run names for the errors/sig stages

SWEEP_ROOT="outputs/sweep"
# Exp 0 is deliberately NOT per-session: the learning rates it selects gate
# every session, and re-probing them for each one would both cost a second
# full probe and risk two sessions running on different rates.
EXP0_DIR="${EXP0_DIR:-${SWEEP_ROOT}/exp0}"
# The fa probe runs in its own session so it cannot disturb EXP0_DIR's runs,
# checkpoints or manifest.
EXP0_FA_DIR="${EXP0_FA_DIR:-${SWEEP_ROOT}/exp0_fa}"
SESSION_ID="${SESSION_ID:-}"
SESSION_FILE="${SWEEP_ROOT}/current_session"
valid_session_id() {
  case "$1" in
    ""|"."|".."|*[!A-Za-z0-9._-]*) return 1 ;;
    *) return 0 ;;
  esac
}
if [ -n "$SESSION_ID" ] && ! valid_session_id "$SESSION_ID"; then
  echo "invalid SESSION_ID: use only letters, digits, dot, underscore, or hyphen"
  exit 2
fi
mkdir -p "${SWEEP_ROOT}/sessions"
if [ "$RESET_MANIFEST" = "1" ]; then
  SESSION_ID="${SESSION_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
  SWEEP_DIR="${SWEEP_ROOT}/sessions/${SESSION_ID}"
  if [ -e "$SWEEP_DIR" ] && [ -n "$(ls -A "$SWEEP_DIR" 2>/dev/null)" ]; then
    echo "fresh session already exists and is non-empty: $SWEEP_DIR"
    echo "choose another SESSION_ID, or resume it without RESET_MANIFEST=1"
    exit 2
  fi
  mkdir -p "$SWEEP_DIR"
  printf '%s\n' "$SESSION_ID" > "$SESSION_FILE"
elif [ -n "$SESSION_ID" ]; then
  SWEEP_DIR="${SWEEP_ROOT}/sessions/${SESSION_ID}"
elif [ -f "$SESSION_FILE" ]; then
  SESSION_ID="$(sed -n '1p' "$SESSION_FILE")"
  if ! valid_session_id "$SESSION_ID"; then
    echo "invalid session id in $SESSION_FILE"
    exit 2
  fi
  SWEEP_DIR="${SWEEP_ROOT}/sessions/${SESSION_ID}"
else
  # Backward-compatible location for repositories without a session pointer.
  SWEEP_DIR="$SWEEP_ROOT"
fi
STOP_FILE="${SWEEP_DIR}/STOP"

# ---- assemble the sweep args once; --status reuses them -------------------
sweep_args=(--tests "$TESTS" --epochs "$EPOCHS" --flat-bs "$FLAT_BS"
            --combo-bs "$COMBO_BS" --npz "$NPZ" --runs-dir "$SWEEP_DIR")
# Pick up the selection the exp0 stage wrote, if it is there. Called TWICE:
# once here so --status/--dry-run see it, and again immediately before the
# train stage. The second call is the one that matters for the natural
# STAGES=exp0,train invocation, where the file does not exist yet when these
# args are first assembled -- without it that run would print "the train stage
# picks them up automatically" and then train the whole matrix on config
# defaults.
lr_selection_applied=0
apply_lr_selection() {
  [ "$lr_selection_applied" = "1" ] && return 0
  if [ -z "$LR_SELECTION" ] && [ -f "${EXP0_DIR}/lr_selection.json" ]; then
    LR_SELECTION="${EXP0_DIR}/lr_selection.json"
    echo "=== using learning rates from ${LR_SELECTION} ==="
  fi
  if [ -n "$LR_SELECTION" ]; then
    sweep_args+=(--lr-selection "$LR_SELECTION")
    if [ -z "$LR_SELECTION_PATCH" ] && [ -f "${EXP0_FA_DIR}/lr_selection.json" ]; then
      LR_SELECTION_PATCH="${EXP0_FA_DIR}/lr_selection.json"
      echo "=== folding in the fa probe from ${LR_SELECTION_PATCH} ==="
    fi
    [ -n "$LR_SELECTION_PATCH" ] && sweep_args+=(--lr-selection-patch "$LR_SELECTION_PATCH")
    lr_selection_applied=1
  fi
}
apply_lr_selection
[ -n "$NUM_WORKERS" ]       && sweep_args+=(--num-workers "$NUM_WORKERS")
[ -n "$COMBO_NUM_WORKERS" ] && sweep_args+=(--combo-num-workers "$COMBO_NUM_WORKERS")
[ -n "$GPUS" ]     && sweep_args+=(--gpus "$GPUS")
[ -n "$SEEDS" ]    && sweep_args+=(--seeds "$SEEDS")
[ -n "$SERIES" ]   && sweep_args+=(--series "$SERIES")
[ -n "$RETRAIN" ]  && sweep_args+=(--retrain "$RETRAIN")
[ -n "$ONLY" ]     && sweep_args+=(--only "$ONLY")
[ -n "$EXCLUDE" ]  && sweep_args+=(--exclude "$EXCLUDE")
[ "$HYBRID_IDENTITY" = "1" ] && sweep_args+=(--hybrid-identity)
[ "$FORCE" = "1" ] && sweep_args+=(--force)
[ "$RESET_MANIFEST" = "1" ] && sweep_args+=(--reset-manifest)
[ "$NO_STEP_MATCH" = "1" ] && sweep_args+=(--no-step-match)

# ---- --status / --dry-run short-circuits ----------------------------------
for arg in "$@"; do
  case "$arg" in
    --status)  exec $PY scripts/sweep.py "${sweep_args[@]}" --status ;;
    --dry-run) exec $PY scripts/sweep.py "${sweep_args[@]}" --dry-run ;;
    --help|-h) sed -n '2,68p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg (configure via env vars; see --help)"; exit 2 ;;
  esac
done

if [ -f "$STOP_FILE" ]; then
  echo "STOP flag present ($STOP_FILE) -- delete it before starting a run."
  exit 1
fi

mkdir -p "$SWEEP_DIR"
echo "=== hpec_pipeline | stages: $STAGES | tests: $TESTS | epochs: $EPOCHS |" \
     "gpus: ${GPUS:-auto} | $(date -u +%FT%TZ) ==="
echo "=== session: ${SESSION_ID:-legacy} | directory: $SWEEP_DIR ==="
[ -n "$RETRAIN" ] && echo "=== retrain slice: $RETRAIN (forced re-run) ==="

run_stage() {  # run_stage <name> <command...>
  local name="$1"; shift
  local marker="${SWEEP_DIR}/.stage_${name}.ok"
  rm -f "$marker"
  echo ""
  echo "=============================================================================="
  echo "[pipeline] ${name}  ::  $*"
  echo "=============================================================================="
  local t0 rc
  t0=$(date +%s)
  "$@"
  rc=$?
  local mins=$(( ($(date +%s) - t0) / 60 ))
  echo "[pipeline] ${name} finished rc=${rc} in ${mins} min"
  if [ $rc -eq 0 ]; then
    date -u +%FT%TZ > "$marker"
  else
    # Abort rather than press on: evaluating a half-trained sweep, or building
    # results.xlsx from partial eval output, produces a table that looks
    # complete and is not.
    echo ""
    echo "[pipeline] STOPPING: stage '${name}' failed (rc=${rc})."
    if [ -f "$STOP_FILE" ]; then
      echo "[pipeline] a STOP flag is present; delete ${STOP_FILE}, then re-run"
      echo "[pipeline] then re-run to continue the same session."
    else
      echo "[pipeline] fix the cause, then re-run the same session."
    fi
    exit $rc
  fi
}

case ",$STAGES," in *,exp0,*)
  # Its own directory and its own manifest: Exp 0 probes rates the main matrix
  # never runs, so mixing the two would make the main manifest look like it
  # declared runs nobody intends to report.
  exp0_args=(--tests 0 --epochs "$EXP0_EPOCHS" --flat-bs "$FLAT_BS"
             --combo-bs "$COMBO_BS" --npz "$NPZ" --runs-dir "$EXP0_DIR")
  [ -n "$GPUS" ]        && exp0_args+=(--gpus "$GPUS")
  [ -n "$SEEDS" ]       && exp0_args+=(--seeds "$SEEDS")
  [ -n "$SERIES" ]      && exp0_args+=(--series "$SERIES")
  [ -n "$EXP0_MODELS" ] && exp0_args+=(--exp0-models "$EXP0_MODELS")
  [ "$NO_STEP_MATCH" = "1" ]  && exp0_args+=(--no-step-match)
  [ -n "$NUM_WORKERS" ]       && exp0_args+=(--num-workers "$NUM_WORKERS")
  [ -n "$COMBO_NUM_WORKERS" ] && exp0_args+=(--combo-num-workers "$COMBO_NUM_WORKERS")
  run_stage exp0probe $PY scripts/sweep.py "${exp0_args[@]}"

  select_args=(--runs "$EXP0_DIR")
  [ -n "$EXP0_WINDOW" ] && select_args+=(--window "$EXP0_WINDOW")
  run_stage exp0select $PY scripts/select_lr.py "${select_args[@]}"
  echo "[pipeline] learning rates written to ${EXP0_DIR}/lr_selection.json"
  echo "[pipeline] the train stage picks them up automatically." ;;
esac
case ",$STAGES," in *,train,*)
  apply_lr_selection          # exp0 may have just written it in this same run
  if [ -z "$LR_SELECTION" ]; then
    echo "[pipeline] WARNING: no lr_selection.json -- every run will use its"
    echo "[pipeline] config-default learning rate, which was never searched"
    echo "[pipeline] for the multidimensional arm. Run STAGES=exp0 first for"
    echo "[pipeline] an official cross-arm table."
  fi
  run_stage train  $PY scripts/sweep.py "${sweep_args[@]}" ;;
esac
case ",$STAGES," in *,eval,*)
  eval_args=(--runs "$SWEEP_DIR" --npz "$NPZ")
  [ -n "$EVAL_SEEDS" ]      && eval_args+=(--seeds "$EVAL_SEEDS")
  [ "$ALLOW_PARTIAL" = "1" ] && eval_args+=(--allow-partial)
  run_stage eval   $PY scripts/evaluate.py "${eval_args[@]}" ;;
esac
case ",$STAGES," in *,cost,*)
  run_stage cost   $PY scripts/model_cost.py --npz "$NPZ" \
    --flat-batch "$FLAT_BS" --combo-batch "$COMBO_BS" \
    --aggregate-batch "$AGGREGATE_BS" --out-dir "$SWEEP_DIR" ;;
esac
case ",$STAGES," in *,xgb,*)
  run_stage xgb    $PY scripts/xgb_agg.py --npz "$NPZ" \
                     --out "${SWEEP_DIR}/xgboost_aggregate.json" ;;
esac
case ",$STAGES," in *,errors,*)
  if [ -z "$ERRDUMP_RUNS" ]; then
    echo "[pipeline] stage 'errors' needs ERRDUMP_RUNS='<run> <run> ...'"
    exit 2
  fi
  run_stage errors $PY scripts/dump_errors.py --runs "$SWEEP_DIR" \
    --out "$SWEEP_DIR/errdump.npz" $ERRDUMP_RUNS ;;
esac
case ",$STAGES," in *,sig,*)
  run_stage sig    $PY scripts/significance_tests.py \
    --errors "$SWEEP_DIR/errdump.npz" \
    --delta-mode "$DELTA_MODE" --delta "$DELTA" --alpha "$ALPHA" ;;
esac

echo ""
echo "=== hpec_pipeline DONE $(date -u +%FT%TZ) ==="
echo "  leaderboard : ${SWEEP_DIR}/results.csv  (+ metrics.json)"
echo "  rolling     : ${SWEEP_DIR}/rolling_test_scores.csv, rolling_annual.csv, rolling_pooled.csv"
echo "  fairness    : ${SWEEP_DIR}/subgroups*.csv, subgroup_disparities*.csv"
echo "  manifest    : ${SWEEP_DIR}/manifest.json"
echo "  stage state : ${SWEEP_DIR}/.stage_*.ok"
