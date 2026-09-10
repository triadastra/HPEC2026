"""Within an arm, every model must train under an identical protocol.

The benchmark's claim is that multidimensional structure does not earn its
compute. Any protocol flag that differs between two models in the same arm
makes that claim partly a statement about the budget instead of the
architecture, so this asserts uniformity rather than documenting it.

The experimental variables are part of the arm's identity, not violations:
the encoder separates Tests 2 and 3, --axis-identity separates Test 4, and
--lr is what Exp 0 sweeps.
"""

import collections
import glob
import re
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.sweep import (DEFAULT_SEEDS, FLAT_MODELS, ND_BACKBONES,
                           build_exp0_matrix, build_matrix)

G = 30087
ACCUM = 15
KW = dict(accum=ACCUM, effective_batch_size=G)
PROTOCOL = ("--batch-size", "--grad-accum", "--effective-batch-size",
            "--aggregate", "--refit-normalization", "--fresh-model-session",
            "--axis-identity")


def _flags(argv):
    out, i = {}, 0
    while i < len(argv):
        if argv[i].startswith("--"):
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                out[argv[i]] = argv[i + 1]; i += 2
            else:
                out[argv[i]] = "<set>"; i += 1
        else:
            i += 1
    return out


def _arm(name, f):
    """Arm identity INCLUDING the experimental variables, so that a genuine
    confound is what stands out."""
    m = re.match(r"^exp0_([a-z0-9]+)_", name)
    if m:
        return f"exp0:{m.group(1)}"
    enc = f.get("--combo-encoder")
    if "_aggregate_roll" in name:
        return "rolling-agg"
    if "_aggregate" in name:
        return "fixed-agg"
    if "_1d_" in name:
        return f"flat-1d[{f['--variant']}]"
    if f.get("--axis-identity"):
        return f"identity-nd[{enc}]"
    return f"nd[{enc}]"


def _by_arm(runs):
    out = collections.defaultdict(lambda: collections.defaultdict(set))
    models = collections.defaultdict(set)
    for name, argv in runs:
        f = _flags(argv)
        arm = _arm(name, f)
        models[arm].add(f["--model"])
        for flag in PROTOCOL:
            out[arm][flag].add(f.get(flag, "<absent>"))
    return out, models


MAIN = build_matrix({"1", "1.1", "2", "3", "4", "6"}, DEFAULT_SEEDS, 2048, 1, **KW)
EXP0 = build_exp0_matrix([947], 2048, 1, ACCUM, "x.npz", G)


@pytest.mark.parametrize("label,runs", [("main", MAIN), ("exp0", EXP0)])
def test_every_flag_is_uniform_within_its_arm(label, runs):
    by_arm, models = _by_arm(runs)
    offenders = []
    for arm, flags in sorted(by_arm.items()):
        for flag, values in sorted(flags.items()):
            if len(values) > 1:
                offenders.append(
                    f"{label}/{arm} ({len(models[arm])} models): "
                    f"{flag} takes {sorted(values)}")
    assert not offenders, (
        "a model in one arm trains under a different protocol than its "
        "neighbours, which makes any accuracy comparison partly a budget "
        "comparison:\n  " + "\n  ".join(offenders))


def test_the_flat_arm_is_step_matched_to_the_combo_arm():
    by_arm, _ = _by_arm(MAIN)
    for arm, flags in by_arm.items():
        if not arm.startswith("flat-1d"):
            continue
        assert flags["--grad-accum"] == {str(ACCUM)}
        assert flags["--effective-batch-size"] == {str(G)}
        assert flags["--batch-size"] == {"2048"}


def test_every_run_in_the_matrix_gets_the_same_epoch_cap():
    from scripts.sweep import prepare_runs
    prepared = prepare_runs(MAIN, 200, Path("/tmp/x"))
    caps = {_flags(argv).get("--epochs") for _, argv, _ in prepared}
    assert caps == {"200"}


def test_exp0_probes_run_at_one_shared_budget():
    from scripts.sweep import prepare_runs
    prepared = prepare_runs(EXP0, 20, Path("/tmp/x"))
    caps = {_flags(argv).get("--epochs") for _, argv, _ in prepared}
    assert caps == {"20"}, "a probe with a longer budget would be flattered"


# --------------------------------------------------------------------------
# the configs behind the flags
# --------------------------------------------------------------------------

def test_no_model_or_variant_overrides_the_training_budget():
    """batch size, epochs, accumulation and the stopping rule must come from
    base.yaml for every model. A per-model override would be invisible on the
    command line."""
    budget = {"batch_size", "combo_batch_size", "epochs", "grad_accum",
              "patience", "min_delta", "scheduler", "scheduler_patience",
              "scheduler_factor", "optimizer", "weight_decay", "grad_clip"}
    offenders = []
    for path in sorted(glob.glob(str(REPO / "config/models/*.yaml"))
                       + glob.glob(str(REPO / "config/variants/*.yaml"))):
        training = (yaml.safe_load(open(path)) or {}).get("training") or {}
        clash = budget & set(training)
        if clash:
            offenders.append(f"{Path(path).name}: {sorted(clash)}")
    assert not offenders, f"per-model training-budget overrides: {offenders}"


def test_the_only_per_model_override_is_the_learning_rate():
    """And it is the one Exp 0 exists to remove. If a second key ever appears
    here, the uniform-protocol claim needs re-examining."""
    seen = {}
    for path in sorted(glob.glob(str(REPO / "config/models/*.yaml"))):
        training = (yaml.safe_load(open(path)) or {}).get("training") or {}
        if training:
            seen[Path(path).name] = sorted(training)
    assert seen == {"transformer.yaml": ["lr"]}, (
        f"unexpected per-model training overrides: {seen}")


def test_a_selected_rate_overrides_the_per_model_exception():
    """transformer.yaml pins lr=1e-4 while every other model runs the base
    1e-3 -- the asymmetry Exp 0 removes. The injected rate has to win."""
    selection = {"schema_version": 1, "selected": {
        "flat": {m: "7e-4" for m in FLAT_MODELS},
        "agg": {m: "7e-4" for m in FLAT_MODELS},
        "roll": {m: "7e-4" for m in FLAT_MODELS},
        "nd": {m: "7e-4" for m in ND_BACKBONES},
        "mech": {m: "7e-4" for m in ND_BACKBONES}}}
    runs = build_matrix({"3"}, [947], 2048, 1, lr_selection=selection, **KW)
    transformer = [argv for _, argv in runs
                   if argv[argv.index("--model") + 1] == "transformer"]
    assert transformer
    for argv in transformer:
        assert argv[argv.index("--lr") + 1] == "7e-4"

    source = (REPO / "scripts/train.py").read_text()
    assert "cfg[\"training\"][\"lr\"] = args.lr" in source, (
        "the CLI rate must overwrite the composed config, or the per-model "
        "exception would silently win")
