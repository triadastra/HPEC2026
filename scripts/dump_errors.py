#!/usr/bin/env python
"""Dump provenance-validated per-series x per-month squared errors.

Every requested run must be present in the sweep manifest, have a valid
completion record, and declare the same Census lattice and split contract.
The resulting NPZ embeds provenance metadata consumed by
``scripts/significance_tests.py``.
"""
import argparse
from dataclasses import asdict
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("ev", str(REPO / "scripts/evaluate.py"))
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

from src.data.census_loader import CensusConfig, CensusLattice  # noqa: E402
from src.utils import (  # noqa: E402
    argv_value,
    compose_config,
    run_input_fingerprint,
    validate_completion,
)


def census_config_from_entry(entry):
    """Reconstruct the data configuration used by one manifest command."""
    argv = entry["argv"]
    model = argv_value(argv, "--model")
    variant = argv_value(argv, "--variant")
    if not model or not variant:
        raise ValueError(f"{entry.get('name')}: manifest lacks model/variant")
    base = argv_value(argv, "--config") or "config/base.yaml"
    data_config = argv_value(argv, "--data-config")
    cfg = compose_config(model, variant, base=base, extra=[data_config] if data_config else [])
    data = dict(cfg.get("data", {}))

    for flag, key, cast in (
        ("--input-len", "input_len", int),
        ("--lag-count", "lag_count", int),
        ("--train-end", "train_end", int),
        ("--val-end", "val_end", int),
        ("--test-end", "test_end", int),
    ):
        value = argv_value(argv, flag)
        if value is not None:
            data[key] = cast(value)
    if "--aggregate" in argv:
        data["aggregate"] = True
    if "--refit-normalization" in argv:
        data["refit_normalization"] = True

    npz_value = argv_value(argv, "--npz") or data.get("npz")
    if not npz_value:
        raise ValueError(f"{entry.get('name')}: manifest does not identify a lattice")
    data_dir = Path(argv_value(argv, "--data-dir") or ".")
    npz = Path(npz_value)
    if not npz.is_absolute():
        npz = REPO / data_dir / npz
    data["npz"] = str(npz.resolve())

    fields = CensusConfig.__dataclass_fields__
    return CensusConfig(**{key: value for key, value in data.items() if key in fields})


def validated_entries(runs_dir: Path, names):
    planned = ev._planned_entries(runs_dir)
    if planned is None:
        raise ValueError(f"{runs_dir}/manifest.json is required")
    selected = []
    for name in names:
        entry = planned.get(name)
        if entry is None:
            raise ValueError(f"{name}: not declared in {runs_dir}/manifest.json")
        current = run_input_fingerprint(entry.get("argv", []), REPO)
        if current != entry.get("input_fingerprint"):
            raise ValueError(f"{name}: current inputs differ from the manifest fingerprint")
        error = validate_completion(runs_dir, entry)
        if error:
            raise ValueError(error)
        selected.append(entry)
    return selected


def observation_losses(predictions, targets):
    """Return channel-pooled squared loss and true mean absolute error."""
    errors = predictions - targets
    return (errors ** 2).mean(-1), np.abs(errors).mean(-1)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+", help="manifest run names to include")
    parser.add_argument("--runs", default=str(REPO / "outputs/sweep"))
    parser.add_argument("--out", default=str(REPO / "outputs/sweep/errdump.npz"))
    args = parser.parse_args(argv)

    runs_dir = Path(args.runs).resolve()
    entries = validated_entries(runs_dir, args.names)
    configs = [census_config_from_entry(entry) for entry in entries]
    if any(config.aggregate for config in configs):
        raise ValueError("aggregate runs cannot be mixed into the per-series significance dump")
    reference = asdict(configs[0])
    mismatched = [
        entry["name"] for entry, config in zip(entries, configs)
        if asdict(config) != reference
    ]
    if mismatched:
        raise ValueError(
            "all significance runs must use the identical Census data contract; "
            f"mismatches: {', '.join(mismatched)}"
        )

    cl = CensusLattice(configs[0])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n_series = cl.N
    val_end = cl.config.val_end
    # The test window is bounded by test_end, which a fold may move inside the
    # panel; cl.T is the panel length. evaluate.py was updated for this and this
    # script was not, so a bounded run produced matrices whose month count did
    # not match its predictions.
    test_end = cl.test_end
    n_months = test_end - val_end
    out = {}
    run_metadata = {}

    for entry in entries:
        name = entry["name"]
        cfg = ev.parse_run(name)
        if cfg is None:
            raise ValueError(f"{name}: cannot parse run name")
        checkpoint = runs_dir / name / "best.pth"
        predictions, targets = ev.eval_ckpt(str(checkpoint), cfg, cl, dev)
        squared, absolute = observation_losses(predictions, targets)
        if cfg["combo"]:
            matrix = squared.reshape(n_months, n_series).T
            absolute_matrix = absolute.reshape(n_months, n_series).T
        else:
            expected_value = cl.panel[:, val_end:test_end, cl.target_ch[0]]
            target_value = targets[:, 0]
            if np.allclose(target_value.reshape(n_series, n_months), expected_value, atol=1e-5):
                matrix = squared.reshape(n_series, n_months)
                absolute_matrix = absolute.reshape(n_series, n_months)
            elif np.allclose(
                target_value.reshape(n_months, n_series).T, expected_value, atol=1e-5
            ):
                matrix = squared.reshape(n_months, n_series).T
                absolute_matrix = absolute.reshape(n_months, n_series).T
            else:
                raise RuntimeError(f"{name}: cannot establish sample order")
        out[name] = matrix.astype(np.float32)
        out[f"{name}__absolute"] = absolute_matrix.astype(np.float32)
        completion = json.loads((runs_dir / name / "run_complete.json").read_text())
        run_metadata[name] = {
            "fingerprint": entry["fingerprint"],
            "input_fingerprint": entry["input_fingerprint"],
            "checkpoint_sha256": completion["checkpoint_sha256"],
        }
        print(f"dumped {name}: mean {matrix.mean():.6f}")
        if dev == "cuda":
            torch.cuda.empty_cache()

    target_channels = cl.target_ch
    prediction = cl.panel[:, val_end - 1:test_end - 1, :][:, :, target_channels]
    truth = cl.panel[:, val_end:test_end, :][:, :, target_channels]
    out["persistence"] = (((prediction - truth) ** 2).mean(-1)).astype(np.float32)
    out["persistence__absolute"] = np.abs(prediction - truth).mean(-1).astype(np.float32)
    metadata = {
        "schema_version": 3,
        "runs_dir": str(runs_dir),
        "census_config": reference,
        "shape": [n_series, n_months],
        "loss_arrays": {
            "squared": "<run>",
            "absolute": "<run>__absolute",
        },
        "runs": run_metadata,
    }
    out["__metadata__"] = np.asarray(json.dumps(metadata, sort_keys=True))

    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **out)
    print(f"saved {output} ({len(entries)} runs + persistence)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
