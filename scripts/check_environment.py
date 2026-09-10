#!/usr/bin/env python3
"""Fail-fast dependency check for the committed HPEC training environment."""
import argparse
import importlib.metadata
import platform
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-mamba", action="store_true",
                        help="also require Linux, CUDA, Triton, and every Mamba model")
    args = parser.parse_args()
    errors = []

    try:
        import torch
        print(f"torch {torch.__version__} | CUDA runtime {torch.version.cuda} | "
              f"CUDA available {torch.cuda.is_available()}")
    except Exception as exc:
        print(f"torch import failed: {exc}")
        return 1

    if args.require_mamba:
        if platform.system() != "Linux":
            errors.append("the full Mamba roster requires Linux")
        if not torch.cuda.is_available():
            errors.append("CUDA is not available to PyTorch")
        try:
            import triton
            print(f"triton {triton.__version__}")
        except Exception as exc:
            errors.append(f"Triton import failed: {exc}")
        for distribution in ("tilelang", "apache-tvm-ffi", "quack-kernels"):
            try:
                print(f"{distribution} {importlib.metadata.version(distribution)}")
            except importlib.metadata.PackageNotFoundError:
                errors.append(f"missing package: {distribution}")

    try:
        from src.models import ModelFactory
        registered = set(ModelFactory.list_models())
        print("registered models: " + ", ".join(sorted(registered)))
        if args.require_mamba:
            missing = {"mamba2", "mamba3", "mamba_nd"} - registered
            if missing:
                errors.append("unavailable model registrations: " + ", ".join(sorted(missing)))
    except Exception as exc:
        errors.append(f"model registry import failed: {exc}")

    if errors:
        print("ENVIRONMENT CHECK FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print("environment check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
