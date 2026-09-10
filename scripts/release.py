#!/usr/bin/env python3
"""Public-release entry points; downloads are pinned and training is explicit."""
import argparse
import json
import os
import hashlib
import shutil
from pathlib import Path, PurePosixPath
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = json.loads((ROOT / "SOURCE.json").read_text())
RESULT_FILES = [
    "README.md", "manifest.json",
    "exp7_exp8_operator_study/deliverables/combined_results.csv",
    "exp7_exp8_operator_study/deliverables/per_run_metrics.csv",
    "exp7_exp8_operator_study/deliverables/model_cost_FINAL.csv",
    "exp7_exp8_operator_study/final/results/significance_FINAL_v3.txt",
    "results/boxA/results.csv", "results/boxB/results.csv",
    "results/boxB/rolling_annual.csv", "results/boxB/rolling_pooled.csv",
]


def safe_run(value):
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or ".." in path.parts
            or "\\" in value or str(path) != value or value == "."):
        raise ValueError("run must be a normalized relative archive path")
    return path


def fetch(filename, destination):
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(
        repo_id=SOURCE["hf_repo"], repo_type="model",
        revision=SOURCE["hf_revision"], filename=filename,
        local_dir=destination,
    ))


def dataset_metadata_problem(metadata):
    if metadata.get("n_series") != 30087:
        return f"dataset has {metadata.get('n_series')} series; archived main runs have 30087"
    if metadata.get("grid_shape") != [1343, 14, 2]:
        return "dataset grid does not match the checkpoint's 1343 commodities, 14 states and 2 flows"
    if metadata.get("cohort_selection", {}).get("state_and_commodity_ranking") != "training_months_only":
        return "dataset lacks the corrected training-only cohort contract"
    return None


def download_data(npz):
    from huggingface_hub import hf_hub_download
    dataset = SOURCE["dataset"]
    cache = ROOT / "outputs/hf-dataset"
    # Inspect the small sidecar before downloading the large, potentially wrong panel.
    metadata_name = "processed/census_lattice_9ch.json"
    sidecar = Path(hf_hub_download(dataset["repo_id"], metadata_name,
                                  repo_type="dataset", revision=dataset["revision"], local_dir=cache))
    if hashlib.sha256(sidecar.read_bytes()).hexdigest() != dataset["files"][metadata_name]:
        raise ValueError("dataset sidecar checksum mismatch")
    problem = dataset_metadata_problem(json.loads(sidecar.read_text()))
    if problem:
        raise ValueError(problem + "; use rebuild-data for a new cohort or obtain the exact corrected training inputs. See README.md.")
    downloaded = {}
    for filename, expected in dataset["files"].items():
        path = Path(hf_hub_download(dataset["repo_id"], filename, repo_type="dataset",
                                    revision=dataset["revision"], local_dir=cache))
        from src.utils.run_manifest import file_sha256
        if file_sha256(path) != expected:
            raise ValueError(f"dataset checksum mismatch: {filename}")
        downloaded[path.suffix] = path
    if npz.exists() or npz.with_suffix(".json").exists():
        raise ValueError("destination already exists; choose a fresh --npz path")
    npz.parent.mkdir(parents=True, exist_ok=True)
    for suffix, path in downloaded.items():
        shutil.copy2(path, npz.with_suffix(suffix))


def rebuild_data(npz):
    from huggingface_hub import snapshot_download
    dataset = SOURCE["dataset"]
    if npz.exists() or npz.with_suffix(".json").exists():
        raise ValueError("rebuild destination already exists; use a fresh --npz path")
    snapshot = Path(snapshot_download(dataset["repo_id"], repo_type="dataset",
                                     revision=dataset["revision"], allow_patterns=["raw/*"],
                                     local_dir=ROOT / "outputs/hf-dataset"))
    call(sys.executable, "scripts/build_census_lattice.py", "--base", str(snapshot / "raw"),
         "--ref", str(snapshot / "raw/reference"), "--out", str(npz.parent), "--name", npz.stem)
    print("Built a NEW cohort from pinned raw archives; this does not establish byte identity with training inputs.")


def checkpoint(run, destination, smoke=False):
    """Verify an archived run before loading tensors; never rewrite provenance."""
    from src.utils.run_manifest import validate_completion
    path = safe_run(run)
    prefix = "" if str(path.parent) == "." else str(path.parent) + "/"
    manifest = json.loads(fetch(prefix + "manifest.json", destination).read_text())
    entries = [entry for entry in manifest["runs"] if entry["name"] == path.name]
    if len(entries) != 1:
        raise ValueError(f"expected exactly one manifest entry for {run}")
    entry = entries[0]
    files = ["best.pth", "run_complete.json", "logs/metrics.json"]
    if "--fresh-model-session" in entry.get("argv", []):
        files.append("model_session.json")
    for filename in files:
        fetch(f"{run}/{filename}", destination)
    error = validate_completion(destination / path.parent, entry)
    if error:
        raise ValueError(error)
    import torch
    state = torch.load(destination / run / "best.pth", map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint is not a nonempty state dictionary")
    if not all(isinstance(value, torch.Tensor) and torch.isfinite(value).all()
               for value in state.values()):
        raise ValueError("checkpoint contains non-tensors or nonfinite tensors")
    report = {"repo": SOURCE["hf_repo"], "revision": SOURCE["hf_revision"],
              "run": run, "verified": True, "loaded_tensors": len(state),
              "parameters_and_buffers": sum(value.numel() for value in state.values()),
              "inference_performed": False}
    if smoke:
        # A cheap real model forward through the archived GRU. No invented
        # observations are used for accuracy reporting.
        if run != "gru_embeddings_1d_s947":
            raise ValueError("smoke currently supports only gru_embeddings_1d_s947")
        from src.models import create_model
        from src.utils import compose_config, model_kwargs_from_config
        kw = model_kwargs_from_config(compose_config("gru", "embeddings", extra=["config/census.yaml"]))
        kw.update(num_numeric_features=35, num_states=14, num_commodities=1343, num_flows=2)
        model = create_model("gru", "embeddings", **kw).eval()
        model.load_state_dict(state, strict=True)
        with torch.no_grad():
            output = model(torch.zeros(2, 36, 35), torch.zeros(2, dtype=torch.long),
                           torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long))
        if tuple(output.shape) != (2, 2) or not torch.isfinite(output).all():
            raise ValueError("GRU smoke forward returned invalid predictions")
        report.update(synthetic_forward_performed=True, output_shape=list(output.shape),
                      accuracy_evaluated=False)
    (destination / run / "release_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def call(*args, env=None):
    subprocess.run(list(args), cwd=ROOT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["test", "results", "checkpoint", "smoke", "plan", "data", "rebuild-data", "data-check", "train"])
    parser.add_argument("--run", default="gru_embeddings_1d_s947")
    parser.add_argument("--destination", type=Path, default=ROOT / "outputs/hf")
    parser.add_argument("--npz", type=Path, default=ROOT / "data/census_port/processed/census_lattice_9ch.npz")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.command == "test":
        env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}
        call(sys.executable, "-m", "pytest", "-q", env=env)
    elif args.command == "results":
        for filename in RESULT_FILES:
            fetch(filename, args.destination)
        print(f"Pinned tables saved to {args.destination}; see docs/audit for qualifications.")
    elif args.command in ("checkpoint", "smoke"):
        checkpoint(args.run, args.destination, smoke=args.command == "smoke")
    elif args.command == "data":
        download_data(args.npz)
    elif args.command == "rebuild-data":
        rebuild_data(args.npz)
    elif args.command == "plan":
        from scripts.sweep import build_matrix, DEFAULT_SEEDS
        runs = build_matrix({"1", "1.1", "2", "3", "4", "6"}, DEFAULT_SEEDS, 2048, 1,
                            accum=15, effective_batch_size=30087)
        out = ROOT / "outputs/planned_matrix.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"planning_only": True, "count": len(runs),
                                  "runs": [{"name": n, "argv": a} for n, a in runs]}, indent=2) + "\n")
        print(f"{len(runs)} planned runs -> {out}. Actual scheduling derives the cohort from the input lattice.")
    else:
        if not args.npz.is_file() or not args.npz.with_suffix(".json").is_file():
            raise ValueError("Missing Census NPZ and matching JSON. See README.md for the archived dataset version mismatch.")
        from src.data.census_loader import CensusLattice, census_config_from_config
        from src.utils import compose_data_config
        lattice = CensusLattice(census_config_from_config(compose_data_config(), npz=str(args.npz)))
        print(f"Validated lattice: {lattice.num_combos} series; {lattice.features_per_group} input features")
        if args.command == "train":
            call(sys.executable, "scripts/check_environment.py", "--require-mamba")
            env = {**os.environ, "PY": sys.executable, "NPZ": str(args.npz),
                   "TESTS": "1,1.1,2,3,4,6", "STAGES": "exp0,train,eval,cost,xgb"}
            call("bash", "scripts/hpec_pipeline.sh", env=env)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Release pipeline stopped: {exc}", file=sys.stderr)
        sys.exit(1)
