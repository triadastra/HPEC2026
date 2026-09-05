#!/usr/bin/env python
"""
Training script for time series forecasting models.

Usage:
    python scripts/train.py --model lstm --variant embeddings
    python scripts/train.py --model gru --variant film_attention --config config/base.yaml
"""

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

# Add repo root to path so ``from src.x import y`` resolves regardless of cwd.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (DIVERGED_EXIT_CODE, write_divergence,
                       load_config, get_device, setup_logging, set_seed,
                       seed_worker, resolve_model_config,
                       flatten_nested_model_cfg, normalize_model_kwargs)
from src.data import TradeDataPipeline, DataConfig
from src.models import COMBO_GRID_VARIANTS, create_model
from src.training import NonFiniteLossError, Trainer


# Per-config combo batch sizes used by the paper. The combo path
# (cross_attention / film_attention for lstm/gru) builds (B, L, G, F)
# windows over ALL groups at once, so its memory footprint is far larger
# than the per-series path and it OOMs at the per-series default of 64.
# The paper did NOT use a single combo batch size -- each combo config
# used its own value (paper included/{LSTM,GRU}/{LSTM,GRU}_combo*.py). We
# reproduce those exactly so default runs match the paper bit-for-bit.
# Override with --batch-size for smoke tests (e.g. 8 fits the 5090).
COMBO_BATCH_SIZES = {
    # Axial (S4ND-style) combo variants: same per-time-step combo dataset.
    **{("lstm", f"{stem}_{d}d"): 4 for d in (2, 3, 4)
       for stem in ("asa",)},
    **{("gru", f"{stem}_{d}d"): 2 for d in (2, 3, 4)
       for stem in ("asa",)},
    **{(host, f"{stem}_{d}d"): 4 for d in (2, 3, 4)
       for host in ("transformer", "gpt")
       for stem in ("asa",)},
    # Mamba-ND scans the dense (S,C,Flow) lattice x time -> small batch.
    **{("mamba_nd", f"{stem}_{d}d"): 2 for d in (2, 3, 4)
       for stem in ("grid",)},
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train time series forecasting models"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        # s4 = REAL S4 (S4Block/DPLR from external/s4) via src/models/s4.py, so it
        # can run the aggregate (Exp 1) path here too; the per-combo Exp 2 S4 runs
        # still go through the upstream repo (external/s4/train.py). s4d remains
        # out (no registered github model).
        choices=[
            "lstm", "gru", "transformer", "gpt",
            "xgboost", "lightgbm", "mamba", "mamba2", "mamba3", "mamba_nd",
            "s4", "s4nd"
        ],
        help="Model architecture to train",
    )

    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        # Current names first, then the legacy spellings they replaced (kept so
        # an old manifest command still parses): "cross_attention_*d" described
        # SELF-attention, "cafa_*d" borrowed the authors' MODEL name for a local
        # build of their FA operator, and the SSM grid tags never meant attention.
        choices=["onehot", "embeddings", *COMBO_GRID_VARIANTS],
        help="Encoding variant",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/base.yaml",
        help="Base configuration file",
    )

    parser.add_argument(
        "--data-config",
        type=str,
        default="config/census.yaml",
        help="Extra config merged LAST (default: config/census.yaml) so its data/"
             "embedding_dims override the base+model+variant composition.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/runs",
        help="Output directory for checkpoints",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Base directory containing data/ folder",
    )

    parser.add_argument(
        "--npz",
        type=str,
        default=None,
        help="Override the Census lattice path from --data-config.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )

    parser.add_argument(
        "--combo-encoder", type=str, default=None, choices=["embeddings", "onehot"],
        help="Axial/CaFA leftover-categorical encoder for the not-yet-an-axis "
             "categoricals (Test 3=embeddings, Test 2=onehot). onehot@2d is heavy.",
    )

    parser.add_argument(
        "--axis-identity", action="store_true",
        help="Test 4: learned identity embeddings on the PROMOTED axes of the "
             "axial/CaFA modules (fair identity-aware arm; default off keeps "
             "Tests 2/3 permutation-equivariant semantics).",
    )

    parser.add_argument(
        "--aggregate", action="store_true",
        help="Test 1: collapse all combos into one summed series.",
    )
    parser.add_argument(
        "--train-end", type=int, default=None,
        help="Override the exclusive Census training target-month boundary.",
    )
    parser.add_argument(
        "--val-end", type=int, default=None,
        help="Override the exclusive Census validation target-month boundary.",
    )
    parser.add_argument(
        "--test-end", type=int, default=None,
        help="Override the exclusive Census test boundary (Test 1.1 uses 12 months).",
    )
    parser.add_argument(
        "--refit-normalization", action="store_true",
        help="Test 1.1: refit aggregate log1p/MinMax normalization on this fold's "
             "training months. Restricted to --aggregate.",
    )
    parser.add_argument(
        "--fresh-model-session", action="store_true",
        help="Start from a newly initialized model in this process and write an "
             "auditable per-run session marker. Test 1.1 requires this flag.",
    )
    parser.add_argument(
        "--num-layers", type=int, default=None,
        help="Override model layer count (the benchmark protocol pins 4; "
             "see PLAN.md §4c).",
    )
    parser.add_argument(
        "--input-len", type=int, default=None,
        help="Override input window length (months).",
    )
    parser.add_argument(
        "--lag-count", type=int, default=None,
        help="Override number of lag features.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )

    parser.add_argument(
        "--grad-accum",
        type=int,
        default=None,
        help="Accumulate gradients over N micro-batches before stepping. "
             "Effective batch = --batch-size * N, and one optimizer step is "
             "taken per N micro-batches. Used to give the per-series (flat) "
             "path the SAME effective batch and optimizer-step count as the "
             "combo/grid path; --effective-batch-size makes the final "
             "micro-batch smaller so the effective count is exact. See RETRAIN.md.",
    )

    parser.add_argument(
        "--effective-batch-size",
        type=int,
        default=None,
        help="Exact number of flat observations per optimizer step. The Census "
             "loader groups micro-batches so no observation is dropped and every "
             "optimizer step contains exactly this many samples.",
    )

    parser.add_argument(
        "--fa-dim-head",
        type=int,
        default=None,
        help="override fa_dim_head (sensitivity ablation; main table stays pinned)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Seed everything BEFORE any model construction or dataloader instantiation.
    # The returned generator is passed to the training DataLoader for
    # reproducible shuffling. See src/utils/seeding.py.
    rng = set_seed(args.seed)

    print("=" * 80)
    print(f"Training {args.model} with {args.variant} variant (seed={args.seed})")
    print("=" * 80)

    # Load configurations
    print("Loading configurations...")
    cfg = load_config(args.config)

    # Load model-specific config. MODEL_CONFIG_ALIAS covers registry names that
    # do not have a YAML of their own (``mamba2`` is the honest name for the
    # wrapper configured by mamba.yaml). Anything with neither a file nor an
    # alias is an error, not a silent fall-through to base.yaml defaults --
    # that is how ``--model mamba2`` used to train on code defaults while
    # config/models/mamba.yaml sat unread. (F9)
    model_config_path = Path(resolve_model_config(args.model))
    if model_config_path.exists():
        cfg = load_config(str(model_config_path), cfg)
        print(f"  Loaded model config from {model_config_path}")
    else:
        raise FileNotFoundError(
            f"No model config for --model {args.model} (looked for "
            f"{model_config_path}). Add the YAML or an entry in "
            "MODEL_CONFIG_ALIAS; models must not silently run on code defaults."
        )

    # Load variant-specific config
    variant_config_path = f"config/variants/{args.variant}.yaml"
    if Path(variant_config_path).exists():
        cfg = load_config(variant_config_path, cfg)
        print(f"  Loaded variant config from {variant_config_path}")

    # Merge the data-config LAST so it wins over base/model/variant (census.yaml
    # carries the census loader + its authoritative embedding_dims 88/7/2).
    if args.data_config and Path(args.data_config).exists():
        cfg = load_config(args.data_config, cfg)
        print(f"  Loaded data config from {args.data_config}")

    # Override with command line arguments
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.npz:
        cfg["data"]["npz"] = args.npz
    if args.aggregate:
        cfg["data"]["aggregate"] = True
    for arg_name, config_name in (
        ("train_end", "train_end"),
        ("val_end", "val_end"),
        ("test_end", "test_end"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            cfg["data"][config_name] = value
    if args.refit_normalization:
        cfg["data"]["refit_normalization"] = True
    if args.num_layers:
        # Write BOTH spellings: a model YAML that already carries n_layers
        # (s4, mamba*, mamba_nd) would otherwise win over the alias filled in
        # by normalize_model_kwargs, and the flag would keep silently doing
        # nothing for exactly those models. (F8)
        cfg["model"]["num_layers"] = args.num_layers
        cfg["model"]["n_layers"] = args.num_layers
    if args.input_len:
        cfg["data"]["input_len"] = args.input_len
    if args.lag_count:
        cfg["data"]["lag_count"] = args.lag_count
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
        if args.model in ("xgboost", "lightgbm"):
            cfg["model"]["n_estimators"] = args.epochs
    if args.lr:
        cfg["training"]["lr"] = args.lr
    if args.grad_accum:
        cfg["training"]["grad_accum"] = args.grad_accum
    if args.effective_batch_size:
        cfg["training"]["effective_batch_size"] = args.effective_batch_size
    if args.fa_dim_head:
        # Sensitivity ablation only. The main table is pinned by
        # config/variants/fa_*.yaml and is never selected from this sweep.
        cfg["model"]["fa_dim_head"] = args.fa_dim_head
    if args.combo_encoder:
        cfg["model"]["combo_encoder"] = args.combo_encoder
    if args.axis_identity:
        cfg["model"]["axis_identity"] = True

    # Update config with paths
    base_dir = Path(args.data_dir)
    if cfg["data"].get("loader") != "census":
        cfg["data"]["imports_dir"] = str(base_dir / cfg["data"]["imports_dir"])
        cfg["data"]["exports_dir"] = str(base_dir / cfg["data"]["exports_dir"])
    cfg["checkpointing"]["save_dir"] = args.out_dir
    cfg["logging"]["log_dir"] = str(Path(args.out_dir) / "logs")

    if args.refit_normalization and args.aggregate and not args.fresh_model_session:
        raise ValueError("rolling aggregate runs require --fresh-model-session")
    if args.fresh_model_session:
        # train.py has no checkpoint-loading path: every invocation constructs a
        # new model below. Record a unique process/session identity so this
        # property is auditable for every rolling fold and seed.
        session_id = uuid.uuid4().hex
        session_dir = Path(args.out_dir)
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "model_session.json").write_text(json.dumps({
            "session_id": session_id,
            "pid": os.getpid(),
            "model": args.model,
            "variant": args.variant,
            "seed": args.seed,
            "fresh_initialization": True,
            "checkpoint_loaded": False,
        }, indent=2))
        print(f"  Fresh model session: {session_id} (pid={os.getpid()})")

    # Set experiment name
    if cfg["logging"]["experiment_name"] is None:
        cfg["logging"]["experiment_name"] = f"{args.model}_{args.variant}"

    print("Configuration loaded")
    print(f"  Model: {args.model}")
    print(f"  Variant: {args.variant}")
    _ga = cfg["training"].get("grad_accum", 1)
    _effective = cfg["training"].get("effective_batch_size")
    print(f"  Batch size: {cfg['training']['batch_size']}"
          + (f" x grad_accum {_ga} = effective "
             f"{_effective or cfg['training']['batch_size'] * _ga}"
             if _ga > 1 else ""))
    print(f"  Epochs: {cfg['training']['epochs']}")
    print(f"  Learning rate: {cfg['training']['lr']}")
    print()

    # Create data pipeline: census HS6 lattice (CensusLattice) vs the WCTR
    # TradeDataPipeline. Both expose get_dataloaders(...) + num_combos /
    # combo_coords / lattice_dims, so the model-construction path below is shared.
    is_census = cfg["data"].get("loader") == "census"
    print("Loading and preprocessing data...")
    if is_census:
        from src.data.census_loader import CensusLattice, census_config_from_config
        # One shared reader of the composed ``data:`` block, so training,
        # evaluation, the cost panel and the aggregate runner cannot drift. It
        # filters by dataclass field, so a new CensusConfig option reaches every
        # entry point without a hand-maintained key list per script.
        pipeline = CensusLattice(census_config_from_config(
            cfg, npz=str(base_dir / cfg["data"]["npz"])))
        num_states, num_commodities, num_flows = (
            pipeline.num_states, pipeline.num_commodities, pipeline.num_flows)
        num_numeric_features = pipeline.features_per_group
    else:
        data_config = DataConfig(**cfg["data"])
        pipeline = TradeDataPipeline(data_config)
        pipeline.load_data()
        train_df, val_df, test_df = pipeline.create_splits()
        print(f"  splits: train {len(train_df)}, val {len(val_df)}, test {len(test_df)}")
        pipeline.build_id_maps(train_df)
        num_states, num_commodities, num_flows = (
            len(pipeline.state2id), len(pipeline.comm2id), len(pipeline.flow2id))
        num_numeric_features = len(pipeline.feat_cols)
    print()

    # Detect combo path. Combo variants (cross_attention, film_attention)
    # for LSTM and GRU use a (B, L, G, F) per-group window with group
    # masks; their model classes set requires_combo_loader=True. The
    # dataloader needs to be built in combo mode AND the model factory
    # needs num_combos / features_per_group BEFORE construction.
    # gru/lstm support flat + axial combo variants; transformer/gpt support
    # only the axial (S4ND-style) variants (they already have native attention).
    # Variants that consume the (B, L, G, F) combo window. Current names plus
    # the legacy spellings they replaced: "cafa_*" borrowed the authors' MODEL
    # name for a local build of their FA operator, "cross_attention_*"
    # described self-attention, and the SSM "grid_*" tags never meant attention.
    _axial = COMBO_GRID_VARIANTS
    combo = (
        (args.model in ("lstm", "gru")
         and args.variant in ("cross_attention", "film_attention") + _axial)
        # transformer/gpt: native attention; mamba_nd/s4nd: grid-native SSMs;
        # mamba/mamba2/mamba3: Test-6 axial/CaFA hybrids (attention grid mixing
        # + SSM temporal backbone) — all consume the (B, L, G, F) combo window.
        or (args.model in ("transformer", "gpt", "mamba_nd", "s4nd",
                           "mamba", "mamba2", "mamba3") and args.variant in _axial)
    )

    # Apply the paper's per-config combo batch size unless the user gave an
    # explicit --batch-size. Done here (not in a YAML) because the value
    # depends on BOTH model and variant, which no single config file knows.
    if combo and not args.batch_size:
        if is_census:
            cfg["training"]["batch_size"] = cfg["training"].get("combo_batch_size", 4)
            print(f"  Combo path (census): batch size {cfg['training']['batch_size']} "
                  f"(training.combo_batch_size). Override with --batch-size.")
        else:
            paper_bs = COMBO_BATCH_SIZES.get((args.model, args.variant))
            if paper_bs is not None:
                cfg["training"]["batch_size"] = paper_bs
                print(f"  Combo path: using paper batch size {paper_bs} for "
                      f"({args.model}, {args.variant}). Override with --batch-size.")

    # Loader worker count. INFRASTRUCTURE, not protocol: the number of workers
    # provably does not change what is trained. The DataLoader draws its base
    # seed from `generator` once per iterator construction regardless of worker
    # count, and neither census dataset uses RNG in __getitem__, so 0, 1, 2 and
    # 4 workers all yield the identical batch sequence -- verified, and pinned
    # by tests/test_combo_loader_equivalence.py. It therefore travels by
    # environment variable rather than argv or config, so that retuning it does
    # not change any run's fingerprint and invalidate finished work.
    #
    # (Contrast persistent_workers, which DOES perturb the shuffle -- see the
    # note in census_loader.get_dataloaders. Worker COUNT is safe; worker
    # LIFETIME is not. Do not generalise one to the other.)
    #
    # Split flat/combo because the two paths have opposite appetites: the flat
    # arm calls __getitem__ G x 96 times an epoch (millions), while the combo
    # arm calls it 96 times. Oversubscription is the real risk -- the sweep runs
    # one job per GPU, so total processes is gpus x workers.
    workers_env = "CENSUS_COMBO_NUM_WORKERS" if combo else "CENSUS_NUM_WORKERS"
    try:
        num_workers = int(os.environ.get(workers_env, 4))
    except ValueError:
        sys.exit(f"{workers_env} must be an integer, got "
                 f"{os.environ[workers_env]!r}")
    if num_workers < 0:
        sys.exit(f"{workers_env} must be >= 0, got {num_workers}")

    # Create dataloaders
    print(f"Creating dataloaders (combo={combo}, num_workers={num_workers} "
          f"from ${workers_env})...")
    train_loader, val_loader, test_loader = pipeline.get_dataloaders(
        batch_size=cfg["training"]["batch_size"],
        num_workers=num_workers,
        generator=rng,
        worker_init_fn=seed_worker,
        combo=combo,
        **({"effective_batch_size": cfg["training"].get("effective_batch_size")}
           if is_census and not combo else {}),
    )

    # Save artifacts (TradeDataPipeline only; census reads a prebuilt .npz)
    if not is_census:
        pipeline.save_artifacts(cfg["checkpointing"]["save_dir"])

    # Create model
    print(f"Creating model: {args.model}...")
    # Copy the model config and remove keys that we pass explicitly below.
    # Otherwise create_model(name=..., variant=..., **model_cfg) raises
    # TypeError: got multiple values for keyword argument 'name'/'variant'.
    model_cfg = dict(cfg["model"])
    model_cfg.pop("name", None)
    model_cfg.pop("variant", None)
    # These are passed explicitly below (derived from the pipeline). census.yaml
    # also lists them under model: for reference, so drop the config copies to
    # avoid a "got multiple values for keyword argument" collision.
    for _k in ("num_states", "num_commodities", "num_flows", "num_numeric_features"):
        model_cfg.pop(_k, None)

    # Translate config embedding_dims to the kwarg names BaseModel expects,
    # so the repo-wide standard (state=8, commodity=32, flow=2) is config-driven.
    # Tree models are stochastic (subsample / colsample) but have no torch RNG
    # to seed, so the run seed must reach them as a constructor kwarg. (F13)
    if args.model in ("xgboost", "lightgbm"):
        model_cfg.setdefault("random_state", args.seed)

    # F4/F8: lift the nested cross_attention/film blocks to the kwarg names the
    # modules read, and expose width/depth under both naming conventions.
    model_cfg = flatten_nested_model_cfg(model_cfg)
    model_cfg = normalize_model_kwargs(model_cfg)

    # F10: S4ND sizes its Time-axis DPLR kernels from input_len, which lives
    # under data:, not model:. Without this it silently built length-36 kernels
    # for whatever window the run actually used.
    model_cfg.setdefault("input_len", cfg["data"]["input_len"])

    emb_dims = model_cfg.pop("embedding_dims", None)
    if emb_dims:
        model_cfg.setdefault("state_embed_dim", emb_dims.get("state", 8))
        model_cfg.setdefault("comm_embed_dim", emb_dims.get("commodity", 32))
        model_cfg.setdefault("flow_embed_dim", emb_dims.get("flow", 2))

    # num_states / num_commodities / num_flows / num_numeric_features were set
    # in the data-pipeline branch above (census or TradeDataPipeline).
    # Combo models need the group dimension + per-group feature count at construction.
    if combo:
        model_cfg["num_combos"] = pipeline.num_combos
        model_cfg["features_per_group"] = num_numeric_features
        # Axial combo variants need the (S,C,Flow) lattice + per-combo coords.
        model_cfg["combo_coords"] = pipeline.combo_coords
        model_cfg["lattice_dims"] = pipeline.lattice_dims

    model = create_model(
        name=args.model,
        variant=args.variant,
        num_numeric_features=num_numeric_features,
        num_states=num_states,
        num_commodities=num_commodities,
        num_flows=num_flows,
        **model_cfg,
    )

    print(f"Model created with {model.get_num_params():,} parameters")
    print()

    # Create trainer
    print("Initializing trainer...")
    trainer = Trainer(cfg)

    # Train
    print("Starting training...")
    print("-" * 80)

    try:
        history = trainer.fit(model, train_loader, val_loader)
    except NonFiniteLossError as exc:
        # Divergence is an ANSWER, not a crash: this configuration does not
        # train. Record it and exit with the reserved code so the scheduler
        # buckets it separately and select_lr.py can count the rate as
        # attempted-and-ineligible instead of refusing the whole selection over
        # a cell whose measurement actually succeeded.
        #
        # Nothing is written that would let this run masquerade as complete:
        # no best.pth, so no completion record, so every provenance gate
        # downstream still rejects it as evidence.
        record = write_divergence(Path(args.out_dir), {
            "status": "diverged",
            "loss": exc.loss,
            "epoch": exc.epoch,
            "step": exc.step,
            "model": args.model,
            "variant": args.variant,
            "seed": args.seed,
            "lr": cfg["training"].get("lr"),
        })
        print("-" * 80)
        print(f"DIVERGED: {exc}")
        print(f"Divergence recorded at: {record}")
        sys.exit(DIVERGED_EXIT_CODE)

    print("-" * 80)
    print("Training completed!")
    print()

    # Print final metrics
    train_loss = history["train_loss"][-1]
    val_loss = history["val_loss"][-1]
    print(f"Final train loss: {train_loss:.6f}")
    print(f"Final val loss: {val_loss:.6f}")
    print()

    print(f"Best model saved to: {args.out_dir}/best.pth")
    print(f"Logs saved to: {cfg['logging']['log_dir']}")

    # Per-run compute cost sidecar (PLAN.md 4.1). best.pth is a bare state_dict,
    # so the cost row cannot ride inside it; evaluate.py joins on the run dir.
    cost = dict(history.get("cost") or {"status": "not_recorded"})
    cost.update({"model": args.model, "variant": args.variant, "seed": args.seed})
    cost_path = Path(args.out_dir) / "cost.json"
    cost_path.parent.mkdir(parents=True, exist_ok=True)
    with cost_path.open("w") as f:
        json.dump(cost, f, indent=2)
    print(f"Cost row saved to: {cost_path}")

    # Save training loss curve plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out_plots = Path(args.out_dir) / "plots"
        out_plots.mkdir(parents=True, exist_ok=True)
        epochs = range(1, len(history["train_loss"]) + 1)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, history["train_loss"], label="train loss")
        ax.plot(epochs, history["val_loss"],   label="val loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{args.model}/{args.variant} — training loss")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_plots / "loss_curve.png", dpi=150)
        plt.close(fig)
        print(f"Loss curve saved to {out_plots}/loss_curve.png")
    except Exception as e:
        print(f"[warn] loss plot failed: {e}")


if __name__ == "__main__":
    main()
