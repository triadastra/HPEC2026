"""
Unified trainer class for all models.

Handles training loop, validation, checkpointing, and early stopping.
"""

import math
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR

try:                                        # measured fwd+bwd cost (PLAN.md 4.1)
    from torch.utils.flop_counter import FlopCounterMode
except Exception:                            # older torch: cost rows degrade, run does not
    FlopCounterMode = None

from .losses import MSELoss, MaskedMSELoss
from .callbacks import EarlyStopping, CheckpointManager
from ..utils.logging import TrainingLogger


class NonFiniteLossError(RuntimeError):
    """The loss became NaN or Inf, so this run has diverged.

    Raised instead of continuing because continuing is provably useless: the
    backward pass writes NaN into every parameter, so every later forward is
    NaN too. The run is over the moment this fires; the only question is
    whether that fact is reported or hidden.

    It used to be hidden. The axial modules scrubbed NaN to zero mid-forward
    (``torch.nan_to_num``), which made the loss finite again, so a diverged run
    trained to its early-stopping patience, wrote a checkpoint, earned a
    completion record, passed every provenance gate, and landed in results.csv
    as an ordinary bad number -- indistinguishable from a model that simply
    fits poorly. ``scripts/evaluate.py`` already raises on non-finite
    predictions: the loud check existed, just at the wrong end of the pipeline.

    Divergence is a RESULT, not an error to paper over -- Exp 0 deliberately
    probes 1e-2 to find where training breaks down, and cannot report that if
    the evidence is scrubbed.

    Caveat worth knowing: grad-clip converts some divergence into finite-but-bad
    rather than NaN (the "clipped divergence" RETRAIN.md records for mamba3).
    This catches the non-finite kind; the clipped kind still reaches the
    leaderboard, honestly rather than scrubbed.
    """

    def __init__(self, loss, epoch, step):
        self.loss, self.epoch, self.step = loss, epoch, step
        super().__init__(
            f"loss became {loss} at epoch {epoch}, optimizer step {step}; "
            "the run has diverged and is being stopped"
        )


class Trainer:
    """
    Unified trainer for all deep learning models.

    Supports:
    - Multiple optimizers (Adam, AdamW, SGD)
    - Learning rate scheduling
    - Early stopping
    - Checkpointing
    - Mixed precision training (AMP)
    - Gradient clipping
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer.

        Args:
            config: Configuration dictionary with training parameters
        """
        self.config = config
        self.device = self._get_device()

        # Training parameters
        training_cfg = config.get("training", {})
        self.batch_size = training_cfg.get("batch_size", 32)
        self.epochs = training_cfg.get("epochs", 200)
        self.lr = training_cfg.get("lr", 1e-3)
        self.weight_decay = training_cfg.get("weight_decay", 0.0)
        self.patience = training_cfg.get("patience", 10)
        self.min_delta = training_cfg.get("min_delta", 1e-4)

        # Gradient accumulation. The effective batch is
        # ``batch_size * grad_accum``; the optimizer steps once every
        # ``grad_accum`` mini-batches. This is what lets the per-series (flat)
        # path run at the SAME effective batch and the SAME optimizer-step
        # count as the combo/grid path without materialising a 28k-sample
        # batch in memory. See RETRAIN.md ("step-matched protocol").
        self.grad_accum = max(int(training_cfg.get("grad_accum", 1)), 1)
        self.effective_batch_size = training_cfg.get("effective_batch_size")
        if self.effective_batch_size is not None:
            self.effective_batch_size = int(self.effective_batch_size)

        # Optimizer and scheduler
        self.optimizer_name = training_cfg.get("optimizer", "adam")
        self.scheduler_name = training_cfg.get("scheduler", "reduce_on_plateau")
        self.scheduler_patience = training_cfg.get("scheduler_patience", 5)
        self.scheduler_factor = training_cfg.get("scheduler_factor", 0.5)
        self.grad_clip = training_cfg.get("grad_clip")
        self.grad_clip_norm = training_cfg.get("grad_clip_norm", 1.0)

        # Mixed precision
        self.use_amp = training_cfg.get("use_amp", False)
        self.scaler = None
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        # Checkpointing
        ckpt_cfg = config.get("checkpointing", {})
        self.save_dir = Path(ckpt_cfg.get("save_dir", "outputs/runs"))
        self.save_best_only = ckpt_cfg.get("save_best_only", True)
        self.monitor = ckpt_cfg.get("monitor", "val_loss")
        self.mode = ckpt_cfg.get("mode", "min")
        self.save_last = ckpt_cfg.get("save_last", True)

        # Logging
        log_cfg = config.get("logging", {})
        self.log_dir = log_cfg.get("log_dir", "outputs/logs")
        self.experiment_name = log_cfg.get("experiment_name")

        # Realized optimizer-step accounting (audit trail for the
        # step-matched protocol: the log must show what was actually run,
        # not what the config intended).
        self.steps_per_epoch = 0
        self.total_steps = 0
        self.current_epoch = 0
        self.effective_observations = 0
        self.total_observations = 0

        # State
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.early_stopping = None
        self.checkpoint_manager = None
        self.logger = None

        # History
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
            "steps": [],
            "observations": [],
        }

        # Per-run compute cost. Populated by _record_cost() at the end of fit().
        self.cost: Dict[str, Any] = {"status": "not_recorded"}

    def _get_device(self) -> torch.device:
        """Get device from config."""
        training_cfg = self.config.get("training", {})
        device_str = training_cfg.get("device", "cuda")

        if device_str == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif device_str == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def setup(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        """
        Setup trainer with model and data.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        exact_sampler = getattr(train_loader, "batch_sampler", None)
        sampler_effective = getattr(exact_sampler, "effective_batch_size", None)
        if self.effective_batch_size is not None:
            if sampler_effective != self.effective_batch_size:
                raise ValueError(
                    "exact effective batching was requested but the train loader "
                    "does not expose a matching ExactGroupBatchSampler"
                )
            self.grad_accum = int(exact_sampler.micro_batches_per_group)

        # Detect combo-window models (LSTMComboCA*, GRUComboCA*) — these
        # take (B, L, G, F) input, output (B, G, 2), and need a masked
        # group loss so padded series don't contaminate the gradient.
        self.combo = bool(getattr(model, "requires_combo_loader", False))

        # Create optimizer
        self._create_optimizer()

        # Create scheduler
        self._create_scheduler()

        # Create loss function — masked variant for combo models.
        self.criterion = MaskedMSELoss() if self.combo else MSELoss()

        # Create early stopping
        self.early_stopping = EarlyStopping(
            patience=self.patience,
            min_delta=self.min_delta,
            mode=self.mode,
        )

        # Create checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            save_dir=self.save_dir,
            monitor=self.monitor,
            mode=self.mode,
            save_best_only=self.save_best_only,
            save_last=self.save_last,
        )

        # Create logger
        self.logger = TrainingLogger(
            log_dir=self.log_dir,
            experiment_name=self.experiment_name,
        )

        self.logger.log_info(f"Trainer initialized on device: {self.device}")
        self.logger.log_info(f"Model parameters: {model.get_num_params():,}")
        # Step-budget audit trail (RETRAIN.md). Log the realized numbers so a
        # run's optimizer-step count can be read straight out of its log
        # instead of re-derived from dataset sizes and batch shapes.
        micro = len(train_loader)
        steps = -(-micro // self.grad_accum)          # ceil
        # Effective batch is quoted in OBSERVATIONS so the two arms are
        # comparable. A combo sample is one month carrying every group, so its
        # effective batch is batch_size x G, not batch_size: quoting the raw
        # sample count made a correctly step-matched pair look like an 18x
        # mismatch in exactly the log line RETRAIN.md tells the operator to
        # compare.
        self.effective_observations = self._effective_observations(train_loader)
        self.logger.log_info(
            f"Batch budget: micro-batch {self.batch_size} x grad_accum "
            f"{self.grad_accum} = effective batch {self.effective_observations} "
            f"observations | {micro} micro-batches/epoch "
            f"-> {steps} optimizer steps/epoch | max {steps * self.epochs} steps "
            f"over {self.epochs} epochs"
        )

    def _effective_observations(self, train_loader) -> int:
        """Observations per optimizer step, comparable across flat and combo.

        The flat arm pins this explicitly via ``effective_batch_size``. The
        combo arm cannot: one sample is a whole month, so the count is
        ``batch_size x groups``.
        """
        if self.effective_batch_size is not None:
            return int(self.effective_batch_size)
        groups = getattr(getattr(train_loader, "dataset", None), "G", None)
        if self.combo and groups:
            return int(self.batch_size * self.grad_accum * groups)
        return int(self.batch_size * self.grad_accum)

    def _create_optimizer(self) -> None:
        """Create optimizer."""
        if self.optimizer_name == "adam":
            self.optimizer = Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "adamw":
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "sgd":
            self.optimizer = SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

    def _create_scheduler(self) -> None:
        """Create learning rate scheduler."""
        if self.scheduler_name == "reduce_on_plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
            )
        elif self.scheduler_name == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
            )
        elif self.scheduler_name == "step":
            self.scheduler = StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1,
            )
        else:
            self.scheduler = None

    def _forward_and_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Run model + loss for a single batch. Branches on self.combo.

        Per-series path: input (B, L, F); model gets state/comm/flow IDs;
        loss is plain MSE on (B, 2) preds vs (B, 2) targets.

        Combo path: input (B, L, G, F); model gets group_mask; loss is
        masked MSE that ignores padded slots so the optimizer doesn't
        chase zeros for missing series. State/comm/flow IDs are not
        present in combo batches and are not used by combo models.
        """
        x_numeric = batch["x_numeric"].to(self.device)
        target_value = batch["target_value"].to(self.device)
        target_weight = batch["target_weight"].to(self.device)

        if self.combo:
            group_mask = batch["group_mask"].to(self.device)
            predictions = self.model(x_numeric, group_mask=group_mask)  # (B, G, 2)
            loss = self.criterion(predictions, target_value, target_weight, group_mask)
        else:
            state_ids = batch["state_ids"].to(self.device)
            comm_ids = batch["comm_ids"].to(self.device)
            flow_ids = batch["flow_ids"].to(self.device)
            target = torch.stack([target_value, target_weight], dim=-1)
            predictions = self.model(x_numeric, state_ids, comm_ids, flow_ids)
            loss = self.criterion(predictions, target)
        return loss

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        total_observations = 0
        steps = 0
        group_observations = 0
        grad_norms = []          # device-side; reduced once at epoch end

        accum = self.grad_accum
        num_micro = len(self.train_loader)
        self.optimizer.zero_grad()

        for i, batch in enumerate(self.train_loader):
            # Per-batch losses are means. Exact step-matched groups can end in a
            # smaller micro-batch, so weight each mean by its observation share;
            # ordinary accumulation keeps the equal-micro-batch convention.
            is_last = (i + 1) == num_micro
            group_start = (i // accum) * accum
            group_size = min(accum, num_micro - group_start)
            step_now = ((i + 1) % accum == 0) or is_last
            batch_observations = int(batch["target_value"].numel())
            group_observations += batch_observations
            loss_scale = (batch_observations / self.effective_batch_size
                          if self.effective_batch_size is not None
                          else 1.0 / group_size)
            if (step_now and self.effective_batch_size is not None
                    and group_observations != self.effective_batch_size):
                raise RuntimeError(
                    f"optimizer group contained {group_observations} observations; "
                    f"expected {self.effective_batch_size}"
                )

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    loss = self._forward_and_loss(batch)
                self.scaler.scale(loss * loss_scale).backward()
                if step_now:
                    if self.grad_clip:
                        self.scaler.unscale_(self.optimizer)
                        grad_norms.append(torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip_norm,
                        ).detach())
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    steps += 1
                    group_observations = 0
            else:
                loss = self._forward_and_loss(batch)
                (loss * loss_scale).backward()
                if step_now:
                    if self.grad_clip:
                        grad_norms.append(torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip_norm,
                        ).detach())
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    steps += 1
                    group_observations = 0

            # Piggyback on the .item() this line already needs: the sync is
            # paid regardless, so the divergence check is free. Checking after
            # the step rather than before is deliberate -- the weights are
            # already poisoned either way, and the run is being abandoned.
            batch_loss = loss.item()
            if not math.isfinite(batch_loss):
                raise NonFiniteLossError(batch_loss, self.current_epoch, steps)
            total_loss += batch_loss * batch_observations
            total_observations += batch_observations

        self.steps_per_epoch = steps
        self.total_steps += steps
        self.total_observations += total_observations
        # clip_grad_norm_ already computes the pre-clip norm and the old code
        # threw it away. Kept on-device and reduced once per epoch rather than
        # .item()'d per step: a per-step sync would cost real time, while one
        # reduction over a few dozen scalars costs nothing measurable.
        #
        # Worth logging because it is the evidence behind the stability
        # finding: grad-clip converts some divergence into finite-but-bad loss
        # rather than NaN, so a run can look merely poor while actually being
        # clipped on every step. clip_fraction says which.
        if grad_norms:
            stacked = torch.stack(grad_norms)
            self.last_grad_stats = {
                "grad_norm_mean": float(stacked.mean()),
                "grad_norm_max": float(stacked.max()),
                "clip_fraction": float(
                    (stacked > self.grad_clip_norm).float().mean()),
            }
        else:
            self.last_grad_stats = {}
        return total_loss / total_observations

    @torch.no_grad()
    def validate(self) -> float:
        """Validate using an observation-weighted mean loss.

        Weighting batch means equally overweights the final partial batch and
        makes checkpoint selection depend on how an architecture batches the
        same validation observations.
        """
        self.model.eval()
        total_loss = 0.0
        total_elements = 0
        for batch in self.val_loader:
            loss = self._forward_and_loss(batch)
            if self.combo:
                elements = int(batch["group_mask"].sum().item()) * 2
            else:
                elements = int(batch["target_value"].numel()) * 2
            total_loss += loss.item() * elements
            total_elements += elements
        if not total_elements:
            raise ValueError("validation loader contained no target observations")
        return total_loss / total_elements

    def _fit_sklearn(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, Any]:
        """
        Train a sklearn-style model (e.g., XGBoost) that provides its own
        fit()/predict() API and does not use back-propagation.

        Args:
            model: Model with ``requires_training = False``
            train_loader: Training data loader
            val_loader: Validation data loader

        Returns:
            Training history
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Logger only — no optimizer / scheduler / early-stopping needed
        log_cfg = self.config.get("logging", {})
        self.logger = TrainingLogger(
            log_dir=log_cfg.get("log_dir", "outputs/logs"),
            experiment_name=log_cfg.get("experiment_name"),
        )
        self.logger.log_info(
            f"Trainer initialized for sklearn-style model: {model.__class__.__name__}"
        )

        start_time = time.time()

        # Train via sklearn-style API
        model.fit(train_loader, val_loader, verbose=True)

        # Validate — prefer model-specific metrics when available (XGBoost
        # reports both MAE and MSE in z-space, matching the paper tables).
        if hasattr(model, "evaluate_loader"):
            val_metrics = model.evaluate_loader(val_loader)
            val_loss = val_metrics["mse"]
        else:
            preds, targets = model.predict_loader(val_loader)
            val_loss = float(np.mean((preds - targets) ** 2))
            val_metrics = {"mse": val_loss}

        total_time = time.time() - start_time
        self.logger.log_info(f"Training completed in {total_time:.2f} seconds")
        self.logger.log_info(f"Validation MSE: {val_loss:.6f}")
        if "mae" in val_metrics:
            self.logger.log_info(
                "Validation MAE: "
                f"{val_metrics['mae']:.6f} "
                f"(value={val_metrics['value_mae']:.6f}, "
                f"weight={val_metrics['weight_mae']:.6f})"
            )

        # Save checkpoint using the model's native save method
        ckpt_cfg = self.config.get("checkpointing", {})
        save_dir = Path(ckpt_cfg.get("save_dir", "outputs/runs"))
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(save_dir / "best_xgb"))

        self.cost = {"status": "not_applicable",
                     "reason": "non-gradient model (no torch FLOPs to count)"}
        self.history = {
            "train_loss": [0.0],
            "val_loss": [val_loss],
            "lr": [0.0],
            "cost": self.cost,
        }
        self.logger.save_metrics()
        return self.history

    def _dispatcher_blind_modules(self) -> list:
        """Submodules whose arithmetic FlopCounterMode cannot see.

        FlopCounterMode is a TorchDispatchMode: it observes registered ATen
        ops. The Mamba-2/Mamba-3 SSD scan and Mamba-ND run as raw Triton kernel
        launches (src/models/mamba.py, mamba3.py, mamba_nd.py), which bypass
        the dispatcher entirely, so the dominant term of those models is
        absent from the total. S4 FFT convolutions and fused CUDA RNNs also
        lack counter formulas. These totals remain lower bounds; the static
        forward-cost panel's analytic corrections do not cover backward work.
        """
        blind = set()
        for mod in self.model.modules():
            origin = getattr(type(mod), "__module__", "") or ""
            # FFT arithmetic has no FlopCounterMode formulas. CUDA RNNs
            # likewise use a fused cuDNN op with no training-cost formula.
            fft = type(mod).__name__ in {"FFTConv", "S4NDLayer"}
            fused_rnn = (self.device.type == "cuda" and
                         isinstance(mod, (nn.RNN, nn.GRU, nn.LSTM)))
            if fft or fused_rnn or origin.startswith("mamba_ssm") or ".ops.triton" in origin:
                blind.add(f"{origin}.{type(mod).__name__}")
        return sorted(blind)

    def _measure_step_flops(self) -> Dict[str, Any]:
        """Measure one optimizer step's **forward+backward** FLOPs.

        Called after the epoch loop, never before it: the instrumented step
        consumes dropout RNG and writes grads, so running it up front would
        shift the training trajectory relative to an uninstrumented run. FLOPs
        depend only on architecture and batch shapes, so the end of training
        measures the same quantity the first step would have.

        A whole accumulation **group** is measured, not one micro-batch. The
        step-matched flat arm closes its group on a short micro-batch
        (RETRAIN.md 1: ``13x2,048 + 1x1,668``), so ``micro_flops * grad_accum``
        would overcount. The result is normalised per observation because
        ``train_epoch`` may also close a short final *group*; callers scale by
        observations, not steps.

        Measuring fwd+bwd directly removes the ``~3x forward`` approximation,
        which is wrong for scan-based SSM backward and for any
        gradient-checkpointed path.
        """
        if FlopCounterMode is None:
            return {"status": "unavailable",
                    "reason": "torch.utils.flop_counter not importable"}
        accum = max(1, int(getattr(self, "grad_accum", 1) or 1))
        was_training = self.model.training
        try:
            # validate() leaves the model in eval mode and it runs last, so the
            # measurement would otherwise backward through an eval-mode graph:
            # cuDNN RNN backward rejects that outright (every GRU/LSTM run would
            # report status "failed"), and dropout would be absent from the
            # counted graph even where it succeeds.
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            counter = FlopCounterMode(display=False)
            micro, observations = 0, 0
            with counter:
                for batch in self.train_loader:
                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            loss = self._forward_and_loss(batch)
                    else:
                        loss = self._forward_and_loss(batch)
                    loss.backward()
                    observations += int(batch["target_value"].numel())
                    micro += 1
                    if micro >= accum:
                        break
            flops = int(counter.get_total_flops())
        except Exception as exc:                       # never fail a finished run
            return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
        finally:
            self.model.train(was_training)
            if self.optimizer is not None:
                self.optimizer.zero_grad(set_to_none=True)
        if not observations:
            return {"status": "failed", "reason": "no observations in measured group"}
        blind = self._dispatcher_blind_modules()
        return {
            # A partial total must not be labelled measured: the Pareto would
            # silently omit exactly the work this benchmark compares.
            "status": "partial" if blind else "measured",
            "flops_uncounted_modules": blind,
            "micro_batches_measured": micro,
            "observations_measured": observations,
            # Only a group that filled grad_accum is a whole optimizer step;
            # a short loader yields a group that is not one.
            "flops_per_optimizer_step": flops if micro == accum else None,
            "flops_per_observation": flops / observations,
        }

    def _best_epoch_index(self) -> Optional[int]:
        """Index of the epoch whose val loss produced best.pth, or None."""
        losses = self.history.get("val_loss") or []
        if not losses:
            return None
        return (int(np.argmin(losses)) if self.mode == "min"
                else int(np.argmax(losses)))

    def _record_cost(self) -> None:
        """Attach measured per-run training FLOPs to self.cost / self.history.

        ``*_to_best`` is the number that belongs on a cost-accuracy Pareto: the
        compute actually spent to reach the checkpoint that gets evaluated.
        ``*_total`` includes the early-stopping patience epochs that were run
        and thrown away.
        """
        self.cost = self._measure_step_flops()
        best_idx = self._best_epoch_index()

        def _at_best(series):
            return (int(series[best_idx])
                    if best_idx is not None and best_idx < len(series) else None)

        steps_to_best = _at_best(self.history.get("steps") or [])
        obs_to_best = _at_best(self.history.get("observations") or [])
        # Scale by OBSERVATIONS, not optimizer steps. train_epoch() closes a
        # short final group when len(loader) % grad_accum != 0, so that step
        # costs less than a full one and flops_per_step * total_steps would
        # overcharge it. FLOPs are linear in the batch dimension for every
        # model here, and the combo arm runs accum=1 with a constant group, so
        # per-observation scaling is exact for both arms.
        per_obs = self.cost.get("flops_per_observation")
        self.cost.update({
            "epochs_run": len(self.history.get("train_loss") or []),
            "epochs_to_best": None if best_idx is None else best_idx + 1,
            "optimizer_steps_total": self.total_steps,
            "optimizer_steps_to_best": steps_to_best,
            "observations_total": self.total_observations,
            "observations_to_best": obs_to_best,
            # In OBSERVATIONS, so the two arms are comparable. A combo sample
            # is one month carrying every group, so batch_size * grad_accum is
            # 1 for that arm and would record a step-matched pair as a 28,292x
            # mismatch -- the same units bug that was fixed in the log line.
            "effective_batch_size": (getattr(self, "effective_observations", 0)
                                     or self.effective_batch_size
                                     or self.batch_size * self.grad_accum),
            "train_flops_total": (per_obs * self.total_observations
                                  if per_obs is not None else None),
            "train_flops_to_best": (per_obs * obs_to_best
                                    if per_obs is not None and obs_to_best is not None
                                    else None),
            "validation_flops": None,   # excluded: forward-only over the val split
        })
        self.history["cost"] = self.cost
        if self.cost["status"] in ("measured", "partial"):
            to_best = self.cost["train_flops_to_best"]
            blind = self.cost.get("flops_uncounted_modules") or []
            self.logger.log_info(
                f"Cost [{self.cost['status']}]: "
                f"{self.cost['train_flops_total']/1e12:.2f} TFLOP total (fwd+bwd); "
                + (f"{to_best/1e12:.2f} TFLOP to best "
                   f"(epoch {self.cost['epochs_to_best']})"
                   if to_best is not None else "to-best unavailable")
                + (f" -- UNDERCOUNTED, dispatcher-blind: {', '.join(blind)}"
                   if blind else "")
            )
        else:
            self.logger.log_info(
                f"Cost: not measured ({self.cost['status']}: "
                f"{self.cost.get('reason', 'n/a')})"
            )

    def fit(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader

        Returns:
            Training history
        """
        # Sklearn-style branch (e.g. XGBoost)
        if getattr(model, "requires_training", True) is False:
            return self._fit_sklearn(model, train_loader, val_loader)

        # Setup
        self.setup(model, train_loader, val_loader)

        # Training loop
        best_metric = float("inf") if self.mode == "min" else float("-inf")
        start_time = time.time()

        for epoch in range(self.epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch()
            train_seconds = time.time() - epoch_start

            # Validate
            _val_start = time.time()
            val_loss = self.validate()
            val_seconds = time.time() - _val_start

            # Update scheduler
            if self.scheduler_name == "reduce_on_plateau":
                self.scheduler.step(val_loss)
            elif self.scheduler is not None:
                self.scheduler.step()

            # Log
            # Everything here is already computed or is a host-side counter:
            # no extra kernel launches and no device syncs beyond the ones the
            # loop already pays. Detail is close to free; a sync is not.
            record = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": self.optimizer.param_groups[0]["lr"],
                "epoch_seconds": time.time() - epoch_start,
                "train_seconds": round(train_seconds, 3),
                "val_seconds": round(val_seconds, 3),
                "steps": self.steps_per_epoch,
                "total_steps": self.total_steps,
                "observations": self.total_observations,
                "best_val_loss": min(best_metric, val_loss)
                if self.mode == "min" else max(best_metric, val_loss),
                "epochs_without_improvement": self.early_stopping.counter
                if self.early_stopping is not None else None,
            }
            record.update(getattr(self, "last_grad_stats", {}) or {})
            if torch.cuda.is_available():
                # A counter torch already maintains; reading it is host-side.
                record["gpu_peak_gb"] = round(
                    torch.cuda.max_memory_allocated() / 1e9, 3)
            self.logger.log_epoch(epoch, record)

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])
            self.history["steps"].append(self.total_steps)
            self.history["observations"].append(self.total_observations)

            # Save checkpoint. Honour checkpointing.mode instead of assuming
            # "min": a max-mode monitor would otherwise checkpoint on the wrong
            # comparison while EarlyStopping used the right one. (F11)
            improved = (val_loss < best_metric if self.mode == "min"
                        else val_loss > best_metric)
            if improved:
                best_metric = val_loss
            # The manager decides whether best.pth improves; last.pth must
            # also see non-improving epochs, including the early-stop epoch.
            if math.isfinite(val_loss):
                self.checkpoint_manager.save(
                    self.model, self.optimizer, epoch, val_loss,
                )

            # Early stopping
            if self.early_stopping(val_loss):
                self.logger.log_info(
                    f"Early stopping triggered at epoch {epoch}"
                )
                break

        # Training complete
        total_time = time.time() - start_time
        self.logger.log_info(f"Training completed in {total_time:.2f} seconds")
        self.logger.log_info(
            f"Realized optimizer steps: {self.total_steps} "
            f"({self.steps_per_epoch}/epoch x {len(self.history['train_loss'])} epochs) "
            f"at effective batch "
            f"{getattr(self, 'effective_observations', self.batch_size * self.grad_accum)} "
            f"observations"
        )

        # Per-run compute cost (PLAN.md 4.1). After the loop so the
        # instrumented step cannot perturb the training trajectory.
        self._record_cost()

        # Save history
        self.logger.save_metrics()

        return self.history

    def save_checkpoint(self, path: str) -> None:
        """
        Save training checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "config": self.config,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        self.logger.log_info(f"Checkpoint saved to {path}")
