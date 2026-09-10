"""Release boundaries: reject incompatible inputs and corrupt hosted checkpoints."""
import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts import release


@pytest.mark.parametrize("path", ["../run", "/tmp/run", "x/../../run", "x\\run", "x//run", "."])
def test_archive_paths_cannot_escape_destination(path):
    with pytest.raises(ValueError):
        release.safe_run(path)


def archived_run(tmp_path, monkeypatch):
    name = "gru_embeddings_1d_s947"
    folder = tmp_path / name
    (folder / "logs").mkdir(parents=True)
    torch.save({"weight": torch.ones(2)}, folder / "best.pth")
    checksum = hashlib.sha256((folder / "best.pth").read_bytes()).hexdigest()
    (folder / "run_complete.json").write_text(json.dumps({"fingerprint": "training-fp", "checkpoint_sha256": checksum}))
    (folder / "logs/metrics.json").write_text("[]")
    (tmp_path / "manifest.json").write_text(json.dumps({"runs": [{"name": name, "fingerprint": "training-fp", "argv": []}]}))
    monkeypatch.setattr(release, "fetch", lambda filename, destination: destination / filename)
    return name, folder


def test_verified_tensor_load_keeps_archived_provenance(tmp_path, monkeypatch):
    name, folder = archived_run(tmp_path, monkeypatch)
    before = (folder / "run_complete.json").read_bytes()
    release.checkpoint(name, tmp_path)
    assert (folder / "run_complete.json").read_bytes() == before
    report = json.loads((folder / "release_verification.json").read_text())
    assert report["verified"] and report["loaded_tensors"] == 1
    assert report["inference_performed"] is False


@pytest.mark.parametrize("corruption", ["checkpoint", "fingerprint", "undeclared"])
def test_invalid_archive_is_rejected_before_torch_load(tmp_path, monkeypatch, corruption):
    name, folder = archived_run(tmp_path, monkeypatch)
    if corruption == "checkpoint":
        (folder / "best.pth").write_bytes(b"corrupt")
    elif corruption == "fingerprint":
        (folder / "run_complete.json").write_text('{"fingerprint": "wrong"}')
    else:
        (tmp_path / "manifest.json").write_text('{"runs": []}')
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("loaded an unverified checkpoint"))
    with pytest.raises(ValueError):
        release.checkpoint(name, tmp_path)
    assert not (folder / "release_verification.json").exists()


def test_legacy_dataset_is_rejected_before_large_download(tmp_path, monkeypatch):
    import huggingface_hub
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"n_series": 28292, "grid_shape": [1263, 14, 2]}))
    source = {"dataset": {"repo_id": "test/data", "revision": "pinned", "files": {
        "processed/census_lattice_9ch.json": hashlib.sha256(metadata.read_bytes()).hexdigest()}}}
    monkeypatch.setattr(release, "SOURCE", source)
    calls = []
    def download(repo, filename, **kw):
        calls.append(filename)
        assert filename.endswith(".json"), "must not download incompatible NPZ"
        assert kw["repo_type"] == "dataset" and kw["revision"] == "pinned"
        return str(metadata)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    destination = tmp_path / "input.npz"
    with pytest.raises(ValueError, match="28292"):
        release.download_data(destination)
    assert not destination.exists()
    assert len(calls) == 1


def test_existing_data_is_never_overwritten_by_rebuild(tmp_path):
    path = tmp_path / "data.npz"
    path.write_bytes(b"existing training data")
    with pytest.raises(ValueError, match="already exists"):
        release.rebuild_data(path)
    assert path.read_bytes() == b"existing training data"
