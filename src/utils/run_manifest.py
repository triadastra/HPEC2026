"""Run provenance helpers shared by the sweep launcher and evaluator."""

from __future__ import annotations

import hashlib
from importlib import metadata as importlib_metadata
import json
from functools import lru_cache
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Iterable, Optional


COMPLETION_FILE = "run_complete.json"

# A diverged run is a RESULT ("this configuration does not train"), not a crash.
# It gets its own sidecar and its own exit code so the scheduler can count it
# separately and scripts/select_lr.py can treat that rate as
# attempted-and-ineligible rather than as a missing cell that would refuse the
# whole learning-rate selection. Exp 0 probes 1e-2 expecting some of these.
DIVERGED_FILE = "run_diverged.json"
DIVERGED_EXIT_CODE = 17


def write_divergence(out_dir: Path, record: Dict[str, Any]) -> Path:
    """Record a diverged run next to the checkpoint that was never written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / DIVERGED_FILE
    # NaN/Infinity are not valid JSON. json.dump would emit the bare tokens
    # NaN/Infinity, which json.load accepts but every strict parser rejects --
    # so the loss is stored as a string and stays readable everywhere.
    payload = dict(record)
    if "loss" in payload:
        payload["loss"] = str(payload["loss"])
    path.write_text(json.dumps(payload, indent=2, allow_nan=False))
    return path


def argv_value(argv: Iterable[str], flag: str) -> Optional[str]:
    argv = list(argv)
    try:
        return argv[argv.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(path: str, cwd: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else cwd / candidate


def _input_record(path: Path, hash_contents: bool) -> Dict[str, Any]:
    path = path.resolve()
    record: Dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return record
    stat = path.stat()
    record["size"] = stat.st_size
    if hash_contents:
        record["sha256"] = file_sha256(path)
    else:
        record["mtime_ns"] = stat.st_mtime_ns
    return record


_SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".toml"}
_SOURCE_ROOTS = ("src", "scripts", "config", "external")
_RUNTIME_PACKAGES = (
    "torch", "numpy", "pandas", "scipy", "xgboost", "lightgbm",
    "mamba-ssm",
)


def _hash_source_tree(cwd: Path) -> Dict[str, Any]:
    """Hash executable repository inputs, including initialized submodules."""
    digest = hashlib.sha256()
    count = 0
    candidates = []
    for root_name in _SOURCE_ROOTS:
        root = cwd / root_name
        if root.exists():
            candidates.extend(
                path for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in _SOURCE_SUFFIXES
                and "__pycache__" not in path.parts
            )
    for name in ("requirements.txt", "requirements-cuda.txt", "setup.py", ".gitmodules"):
        path = cwd / name
        if path.is_file():
            candidates.append(path)
    for path in sorted(set(candidates), key=lambda value: str(value.relative_to(cwd))):
        relative = str(path.relative_to(cwd)).replace("\\", "/")
        digest.update(relative.encode())
        digest.update(b"\0")
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
        count += 1
    return {"sha256": digest.hexdigest(), "files": count}


def _git_identity(cwd: Path) -> Dict[str, Any]:
    record: Dict[str, Any] = {}
    for key, command in (
        ("head", ["git", "rev-parse", "HEAD"]),
        ("submodules", ["git", "submodule", "status", "--recursive"]),
    ):
        try:
            result = subprocess.run(
                command, cwd=cwd, check=True, capture_output=True, text=True,
            )
            record[key] = result.stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            record[key] = f"unavailable:{type(exc).__name__}"
    return record


@lru_cache(maxsize=16)
def _repository_record(cwd_text: str) -> Dict[str, Any]:
    cwd = Path(cwd_text).resolve()
    return {"source_tree": _hash_source_tree(cwd), "git": _git_identity(cwd)}


@lru_cache(maxsize=1)
def _runtime_record() -> Dict[str, Any]:
    packages = {}
    for package in _RUNTIME_PACKAGES:
        try:
            packages[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def _fingerprint(argv: Iterable[str], cwd: Path, include_runtime: bool) -> str:
    """Fingerprint command, inputs, code, and optionally the active runtime.

    YAML and sidecar files are small enough to hash. The potentially multi-GB
    NPZ is represented by its resolved path, size, and nanosecond mtime; its
    generated JSON sidecar is content-hashed as the semantic data contract.
    """
    argv = [str(value) for value in argv]
    inputs = []
    for flag in ("--config", "--data-config"):
        value = argv_value(argv, flag)
        if value:
            inputs.append(_input_record(_resolve(value, cwd), hash_contents=True))
    model = argv_value(argv, "--model")
    if model:
        model_config = "mamba" if model in {"mamba", "mamba2"} else model
        inputs.append(_input_record(
            cwd / "config" / "models" / f"{model_config}.yaml", hash_contents=True
        ))
    variant = argv_value(argv, "--variant")
    if variant:
        inputs.append(_input_record(
            cwd / "config" / "variants" / f"{variant}.yaml", hash_contents=True
        ))
    npz_value = argv_value(argv, "--npz")
    if npz_value:
        npz = _resolve(npz_value, cwd)
        inputs.append(_input_record(npz, hash_contents=False))
        inputs.append(_input_record(npz.with_suffix(".json"), hash_contents=True))
    payload = {
        "argv": argv,
        "inputs": inputs,
        "repository": _repository_record(str(cwd.resolve())),
    }
    if include_runtime:
        payload["runtime"] = _runtime_record()
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_input_fingerprint(argv: Iterable[str], cwd: Path) -> str:
    """Fingerprint reproducible inputs without binding the caller's runtime.

    Evaluation may run in a different environment from training (the sweep is
    intentionally split across Mamba/S4 environments). This fingerprint lets
    it prove that data/config/source are unchanged, while the full fingerprint
    stored in the completion record still preserves the training environment.
    """
    return _fingerprint(argv, cwd, include_runtime=False)


def run_fingerprint(argv: Iterable[str], cwd: Path) -> str:
    """Full training identity, including source, submodules, and runtime."""
    return _fingerprint(argv, cwd, include_runtime=True)


def completion_path(runs_dir: Path, name: str) -> Path:
    return runs_dir / name / COMPLETION_FILE


def write_completion(runs_dir: Path, entry: Dict[str, Any]) -> None:
    checkpoint = runs_dir / entry["name"] / "best.pth"
    marker = completion_path(runs_dir, entry["name"])
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": entry["name"],
        "fingerprint": entry["fingerprint"],
        "checkpoint_sha256": file_sha256(checkpoint),
    }
    if "--fresh-model-session" in entry.get("argv", []):
        session_file = marker.parent / "model_session.json"
        session = json.loads(session_file.read_text())
        if not session.get("fresh_initialization") or session.get("checkpoint_loaded"):
            raise ValueError(f"{entry['name']}: invalid fresh model session record")
        payload["model_session_id"] = session.get("session_id")
        if not payload["model_session_id"]:
            raise ValueError(f"{entry['name']}: fresh model session has no session_id")
    tmp = marker.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(marker)


def validate_completion(runs_dir: Path, entry: Dict[str, Any]) -> Optional[str]:
    """Return an error string when a planned run is not provenance-complete."""
    name = entry.get("name")
    fingerprint = entry.get("fingerprint")
    if not name or not fingerprint:
        return f"{name or '<unnamed>'}: manifest entry lacks a fingerprint"
    checkpoint = runs_dir / name / "best.pth"
    marker = completion_path(runs_dir, name)
    if not checkpoint.exists():
        return f"{name}: missing best.pth"
    if not marker.exists():
        return f"{name}: missing {COMPLETION_FILE}"
    try:
        completion = json.loads(marker.read_text())
    except (OSError, ValueError, TypeError) as exc:
        return f"{name}: unreadable {COMPLETION_FILE}: {exc}"
    if completion.get("fingerprint") != fingerprint:
        return f"{name}: completion fingerprint does not match manifest"
    if "--fresh-model-session" in entry.get("argv", []):
        session_file = marker.parent / "model_session.json"
        if not session_file.exists():
            return f"{name}: missing model_session.json"
        try:
            session = json.loads(session_file.read_text())
        except (OSError, ValueError, TypeError) as exc:
            return f"{name}: unreadable model_session.json: {exc}"
        if not session.get("fresh_initialization") or session.get("checkpoint_loaded"):
            return f"{name}: model session does not prove fresh initialization"
        if completion.get("model_session_id") != session.get("session_id"):
            return f"{name}: completion record is bound to a different model session"
    try:
        checkpoint_hash = file_sha256(checkpoint)
    except OSError as exc:
        return f"{name}: cannot hash checkpoint: {exc}"
    if completion.get("checkpoint_sha256") != checkpoint_hash:
        return f"{name}: checkpoint hash does not match completion record"
    return None
