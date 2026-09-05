#!/usr/bin/env python
"""Static cost panel: parameters + FLOPs/forward (+ peak GPU mem if on cuda) per
model x variant, built from the SAME composed config the sweep trains with
(config/base.yaml < config/models/<model>.yaml < config/variants/<variant>.yaml
< config/census.yaml) + census dims.

Params and FLOPs are hardware-independent, so run this on CPU (GPU-free) while the
sweep is training:  CUDA_VISIBLE_DEVICES="" python scripts/model_cost.py
Feeds the cost-accuracy Pareto (PLAN.md 4.1). Training FLOPs ~= 3x the forward.

COUNTING CONVENTION: FLOPs come from torch.utils.flop_counter.FlopCounterMode,
which counts only ops with registered formulas (matmul/conv/attention-class).
Elementwise work — SSM scan recurrences, gates, norms, activations — counts as
zero, and a custom-kernel op with no registered formula silently contributes 0.
Two cores the counter cannot see are COMPUTED from the shapes they receive and
added back, each in its own column so the split stays visible:
  analytic_rnn_topup  the GRU/LSTM recurrence (fused cuDNN on CUDA, and the
                      LSTM even on CPU under torch 2.12, count as zero)
  analytic_fft_topup  S4 / S4ND FFT convolutions (aten::_fft_* has no formula)
  analytic_ssd_topup  Mamba-2 / Mamba-3 chunked SSD scan (raw Triton launches)
  analytic_attn_topup nn.MultiheadAttention's QK^T and PV, only where the
                      backend routed them through an unregistered op (the
                      CPU flash path on torch 2.9); zero on CUDA
Parameter-only work -- S4's DPLR kernel generation, cacheable at deployment --
is excluded by convention and named in `param_only_uncounted`. Anything else
that owns weights and contributed zero lands in `blind_modules` and marks the
row `partial`. Elementwise work (gates, decay, norms, activations, rotary)
counts zero everywhere, so the residual bias is the same for every family.

RUN THIS ON CPU. The count is not hardware-independent even though the quantity
is: what gets counted depends on which ATen op the dispatcher picks, and that
depends on the backend. On CUDA, nn.GRU/nn.LSTM dispatch to the fused
aten::_cudnn_rnn, which has no registered formula, so the entire recurrent core
counts as zero — GRU and LSTM then report byte-identical totals despite ~132k
different parameters. On CPU the same modules dispatch to ops that ARE counted
(GRU 1.65 GFLOPs, LSTM 2.20 GFLOPs on this benchmark's shapes), and their ratio
is 1.3333 = exactly the 4/3 gate ratio, matching the previous submission's
hand-computed top-up (1.3375) from an independent direction.

So: CPU for the FLOPs column, CUDA only for peak_mem_GB. They are different
measurements and belong in separate runs.

Failed measurements are written as diagnostic rows and make the command fail,
so the pipeline cannot certify an incomplete cost/accuracy panel.
"""
import argparse, csv, math, re, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.census_loader import CensusLattice, census_config_from_config
from src.models import create_model
from src.utils import compose_config, compose_data_config, model_kwargs_from_config

try:
    from torch.utils.flop_counter import FlopCounterMode
    HAVE_FLOP = True
except Exception:
    HAVE_FLOP = False

FLAT_MODELS = ("gru", "lstm", "transformer", "s4", "mamba2", "mamba3")
FLAT = [(m, enc) for m in FLAT_MODELS for enc in ("onehot", "embeddings")]
AGGREGATE = [(m, "embeddings", "aggregate") for m in FLAT_MODELS]
COMBO = [(m, f"{mech}_{d}d", enc)
         for m in ("gru", "lstm", "transformer")
         for mech in ("asa", "aca", "fa") for d in (2, 3, 4)
         for enc in ("onehot", "embeddings")]
COMBO += [("mamba_nd", f"grid_{d}d", "embeddings") for d in (2, 3, 4)]  # grid-native SSM
# Test 6: mamba2/mamba3 axial+FA hybrids + genuine S4ND (grid tags).
COMBO += [(m, f"{mech}_{d}d", enc)
          for m in ("mamba2", "mamba3")
          for mech in ("asa", "aca", "fa") for d in (2, 3, 4)
          for enc in ("onehot", "embeddings")]
# The two diagnostic mixers, embeddings only: fa_local (Exp 7, our own build of
# the FA operator -- the one the draft's CaFA numbers came from) and fa_sm
# (Exp 8, the authors' operator on their softmax switch). Costing them is what
# lets the paper say what each arm buys and what it costs.
#
# Two different expectations, and both are worth a table row. fa_sm must match
# fa_* EXACTLY -- softmax adds no weights -- so any divergence there means the
# geometry has drifted from its comparator. fa_local is genuinely cheaper:
# 4H^2 per attended axis plus one shared V, against the authors' 6H-wide
# channel mixer. That gap is the capacity story the mixer comparison needs.
COMBO += [(m, f"{mech}_{d}d", "embeddings")
          for m in ("gru", "lstm", "transformer")
          for mech in ("fa_local", "fa_sm") for d in (2, 3, 4)]
COMBO += [("s4nd", f"grid_{d}d", enc)
          for d in (2, 3, 4) for enc in ("onehot", "embeddings")]
# Test 4 has extra promoted-axis identity embeddings, so its parameter and
# memory costs are not interchangeable with the identity-blind cells above.
# The mamba2/mamba3 rows cost the OPT-IN --hybrid-identity arm (+72 runs, see
# future_work.md): the cost table is static, so measuring it is what makes
# "is the identity-aware hybrid arm worth 72 runs?" an informed decision —
# the same bargain the aca rows already make.
COMBO_ID = [(m, f"{mech}_{d}d", enc)
            for m in ("gru", "lstm", "transformer", "mamba2", "mamba3")
            for mech in ("asa", "aca", "fa") for d in (2, 3, 4)
            for enc in ("onehot", "embeddings")]


def _build(model, variant, enc, combo, cl, axis_identity=False):
    # Same composed config the sweep trains with (base < model < variant <
    # census), so the reported parameter/FLOP counts describe the model that
    # actually runs. The old hardcoded hidden_size/num_layers pair silently
    # costed GPT at its code default d_model=256. (F7)
    composed = compose_config(model, variant, extra=["config/census.yaml"])
    kw = model_kwargs_from_config(composed)
    kw.update(num_numeric_features=cl.features_per_group, num_states=cl.num_states,
              num_commodities=cl.num_commodities, num_flows=cl.num_flows)
    if combo:
        kw.update(num_combos=cl.num_combos, features_per_group=cl.features_per_group,
                  combo_coords=cl.combo_coords, lattice_dims=cl.lattice_dims,
                  combo_encoder=enc, axis_identity=axis_identity)
    return create_model(model, variant, **kw)


_RNN_GATES = {torch.nn.GRU: 3, torch.nn.LSTM: 4, torch.nn.RNN: 1}

# Modules whose zero contribution is the documented convention, not a blind
# spot: normalisation is elementwise, embeddings are a gather.
_EXPECTED_ZERO = (torch.nn.LayerNorm, torch.nn.GroupNorm, torch.nn.Embedding,
                  torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.Dropout)


def rnn_input_shapes(model):
    """Capture what each recurrent module ACTUALLY receives, via forward hooks.

    Not inferable from the model's own input. The combo path enters as
    (B, L, G, F) and folds the lattice into the batch before the backbone, so
    the RNN sees B*G sequences -- 30,087 of them, not 1. Deriving the top-up
    from the outer batch under-counted the combo rows by exactly that factor,
    while the flat rows came out right, which is the kind of error that looks
    like a modelling result rather than a measurement fault.

    Every call is kept, not just the last: the Mamba hosts push the folded
    batch through their mixer in 8,192-row chunks, so one forward is several
    calls, and keeping only the last one costed a 30,087-row scan as 5,511.
    """
    shapes, handles = {}, []
    for name, sub in model.named_modules():
        if type(sub) in _RNN_GATES:
            def grab(mod, args, out, key=name):
                if args and hasattr(args[0], "shape"):
                    shapes.setdefault(key, []).append(tuple(args[0].shape))
            handles.append(sub.register_forward_hook(grab, with_kwargs=False))
    return shapes, handles


def analytic_rnn_flops(module, batch, seq_len):
    """Forward FLOPs of a torch RNN stack, from its shape rather than a counter.

    Derived, not fitted: per layer per timestep each gate does one input
    projection (I x H) and one recurrent projection (H x H), and a MAC is two
    FLOPs. Validated against FlopCounterMode where the counter still sees the
    module -- GRU 205,922,304 and LSTM 274,563,072 at (35 -> 128, 4 layers,
    B=8, L=36), both EXACT.

    This exists because the counter's coverage is neither backend- nor
    version-stable: on CUDA both GRU and LSTM dispatch to the fused
    aten::_cudnn_rnn and count as zero; on CPU under torch 2.12 the LSTM alone
    counts as zero while GRU still counts; under torch 2.9 both count. A figure
    that changes with the machine it was measured on is not a property of the
    model, so the roster's recurrent cores are computed instead of measured.
    """
    gates = _RNN_GATES[type(module)]
    hidden, layers = module.hidden_size, module.num_layers
    dirs = 2 if module.bidirectional else 1
    per_step = 0
    for layer in range(layers):
        in_dim = module.input_size if layer == 0 else hidden * dirs
        per_step += gates * (in_dim * hidden + hidden * hidden) * dirs
    return 2 * per_step * seq_len * batch


# Modules whose per-sequence work is an FFT convolution. FlopCounterMode has no
# formula for aten::_fft_r2c / _fft_c2r, so the S4 core -- the only thing an
# S4 block does that is not a Linear -- counted as zero even after warm-up.
_FFT_CONV_CLASSES = ("FFTConv",)       # vendored S4 (external/s4)
_S4ND_LAYER_CLASSES = ("S4NDLayer",)   # src/models/s4nd.py
# Modules whose forward depends on the parameters only: the DPLR kernel
# generation (Cauchy resolvent) and its FFT. A deployed model computes these
# once; the vendored code recomputes them every forward. Excluded from the
# per-forecast figure by convention and named in `param_only_uncounted`, so
# the exclusion is visible rather than silent.
_PARAM_ONLY_PREFIXES = ("SSMKernel",)


def _rfft_flops(n):
    """Real-input FFT of length n: 2.5 n log2 n, half the 5 n log2 n usually
    quoted for a complex FFT (the standard split-radix estimate)."""
    return 2.5 * n * math.log2(n)


def fft_input_shapes(model):
    """Forward hooks on every FFT-convolution module, capturing what it receives.

    Same reason as rnn_input_shapes: the count depends on the shape the module
    actually sees, and S4ND's lattice folds into the leading dimensions before
    the layer runs.
    """
    shapes, handles = {}, []
    for name, sub in model.named_modules():
        cls = type(sub).__name__
        if cls in _FFT_CONV_CLASSES or cls in _S4ND_LAYER_CLASSES:
            def grab(mod, args, out, key=name):
                if args and hasattr(args[0], "shape"):
                    shapes.setdefault(key, []).append(tuple(args[0].shape))
            handles.append(sub.register_forward_hook(grab, with_kwargs=False))
    return shapes, handles


def analytic_fftconv_flops(module, shape):
    """Per-forward FLOPs of the vendored FFTConv, from its input shape.

    Follows the forward line by line (external/s4/models/s4/s4.py, FFTConv):
    rfft of the input, a complex multiply against the kernel spectrum, an
    irfft, and the D skip term. The kernel side -- generating k and its rfft --
    depends on the parameters only and is excluded (see _PARAM_ONLY_PREFIXES).
    """
    if module.transposed:
        batch, hidden, seq = shape
    else:
        batch, seq, hidden = shape
    l_kernel = seq if getattr(module, "L", None) is None else min(seq, module.L)
    n = l_kernel + seq
    channels = module.channels
    per = batch * hidden * _rfft_flops(n)                    # rfft(x)
    per += batch * channels * hidden * (n // 2 + 1) * 6      # complex multiply
    per += batch * channels * hidden * _rfft_flops(n)        # irfft
    per += batch * channels * hidden * seq * 2               # D skip
    return int(round(per))


def analytic_s4nd_flops(module, shape):
    """Per-forward FLOPs of S4NDLayer: one two-sided FFT convolution per
    lattice axis (src/models/s4nd.py, _axis_conv), then the per-feature skip.
    The input is (B, T, S, C, Fl, H); every axis convolution treats all the
    other dimensions as batch.
    """
    numel = math.prod(shape)
    hidden = shape[-1]
    total = 0
    for seq in module.axis_lens:
        n = 2 * seq
        rows = numel // (hidden * seq)
        total += rows * hidden * _rfft_flops(n)              # rfft(x)
        total += rows * hidden * (n // 2 + 1) * 6            # complex multiply
        total += rows * hidden * _rfft_flops(n)              # irfft
    total += numel * 2                                       # D skip
    return int(round(total))


# Ops through which an attention core can reach the counter. The CPU flash
# path (aten._scaled_dot_product_flash_attention_for_cpu) is not among the
# registered formulas on torch 2.9, so on CPU nn.MultiheadAttention counted its
# projections and nothing else: the Transformer's QK^T and PV were missing,
# 2,654,208 per forecast at four layers, which is exactly how far the CPU and
# CUDA panels disagreed. CUDA's flash / efficient kernels ARE registered.
_ATTN_COUNTED_OPS = ("scaled_dot_product", "flash_attention", "efficient_attention",
                     "cudnn_attention", "bmm", "baddbmm")


def mha_input_shapes(model):
    """Forward hooks on nn.MultiheadAttention capturing (N, Lq, Lk)."""
    shapes, handles = {}, []
    for name, sub in model.named_modules():
        if isinstance(sub, torch.nn.MultiheadAttention):
            def grab(mod, args, kwargs, out, key=name, mod_=sub):
                q = args[0] if args else kwargs.get("query")
                k = (args[1] if len(args) > 1 else kwargs.get("key", q))
                if q is None or not hasattr(q, "shape"):
                    return
                if mod_.batch_first:
                    n, lq, lk = q.shape[0], q.shape[1], k.shape[1]
                else:
                    n, lq, lk = q.shape[1], q.shape[0], k.shape[0]
                shapes.setdefault(key, []).append((n, lq, lk))
            handles.append(sub.register_forward_hook(grab, with_kwargs=True))
    return shapes, handles


def analytic_attention_core_flops(module, shape):
    """QK^T and PV of one nn.MultiheadAttention call: 2 N h Lq Lk d each,
    with h d = embed_dim, so 4 N Lq Lk E in total. The projections are
    ordinary Linears and are always counted; this is only the core."""
    n, lq, lk = shape
    return 4 * n * lq * lk * module.embed_dim


def attention_topup(model, shapes, flop_counts):
    """Analytic attention cores for every MHA whose core the counter missed."""
    total = 0
    mods = dict(model.named_modules())
    for name, calls in shapes.items():
        seen = set()
        for key, ops in flop_counts.items():
            # Keys are "<RootClass>.<path>" for children and the bare root
            # class name (or "Global") when the MHA is itself the model.
            mine = (("." not in key) if name == "" else
                    (key == name or key.split(".", 1)[-1] == name))
            if mine:
                seen.update(str(op) for op, n in ops.items() if n)
        if not any(tag in op for op in seen for tag in _ATTN_COUNTED_OPS):
            for shape in calls:
                total += analytic_attention_core_flops(mods[name], shape)
    return total


_SSD_CLASSES = ("Mamba2", "Mamba3")   # vendored mamba_ssm modules


def ssd_input_shapes(model):
    """Forward hooks on every Mamba-2 / Mamba-3 mixer, capturing (N, L, D)."""
    shapes, handles = {}, []
    for name, sub in model.named_modules():
        origin = getattr(type(sub), "__module__", "") or ""
        if type(sub).__name__ in _SSD_CLASSES and origin.startswith("mamba_ssm"):
            def grab(mod, args, out, key=name):
                if args and hasattr(args[0], "shape"):
                    shapes.setdefault(key, []).append(tuple(args[0].shape))
            handles.append(sub.register_forward_hook(grab, with_kwargs=False))
    return shapes, handles


def analytic_ssd_flops(seq_len, batch, d_state, headdim, nheads, chunk_size,
                       ngroups=1, cb_per_head=False):
    """Forward FLOPs of the chunked SSD scan, as the Triton kernels compute it.

    The scan is four matmuls per chunk, read off the kernels rather than the
    paper: in ssd_chunk_scan / ssd_chunk_state (Mamba-2) and mamba3_siso_fwd
    (Mamba-3) each chunk of length Lc does
        C B^T          2 Lc^2 N   (per group in Mamba-2's ssd_bmm, per head in
                                   Mamba-3, whose grid is one program per head)
        (mask o CB) X  2 Lc^2 P   per head
        B^T X          2 Lc N P   per head   (chunk state)
        C states       2 Lc N P   per head   (previous-state contribution)
    Chunk decay, state passing between chunks, the D skip, softplus, rotary,
    gating and Mamba-3's trapezoidal row product are elementwise and count
    zero -- the same convention every other row is under, so a Linear-heavy
    and a scan-heavy model are costed alike. Chunks are the kernels' actual
    ones: ceil(L / chunk_size) of them, the last one short.

    Validated against a pure-torch chunked SSD that reproduces the vendored
    ssd_minimal_discrete numerically and is counted by FlopCounterMode
    (tests/test_model_cost_topups.py).
    """
    full, rem = divmod(seq_len, chunk_size)
    chunks = [chunk_size] * full + ([rem] if rem else [])
    per_seq = 0
    for lc in chunks:
        per_seq += 2 * lc * lc * d_state * (nheads if cb_per_head else ngroups)
        per_seq += nheads * (2 * lc * lc * headdim + 4 * lc * d_state * headdim)
    return per_seq * batch


def ssd_flops_for_module(module, shape, conv_counted=True):
    """Map a Mamba-2 / Mamba-3 module and its (N, L, D) input onto the count.

    The depthwise conv1d in Mamba-2 is an ordinary nn.Conv1d and IS counted
    unless the causal_conv1d package routes it through Triton; when the
    caller reports it uncounted, its 2 L conv_dim d_conv is added here.
    """
    batch, seq_len = shape[0], shape[1]
    cls = type(module).__name__
    if cls == "Mamba2":
        total = analytic_ssd_flops(seq_len, batch, module.d_state, module.headdim,
                                   module.nheads, module.chunk_size,
                                   ngroups=module.ngroups)
        if not conv_counted:
            conv = module.conv1d
            total += 2 * seq_len * conv.in_channels * conv.kernel_size[0] * batch
        return total
    if cls == "Mamba3":
        if getattr(module, "is_mimo", False):
            raise NotImplementedError("Mamba-3 MIMO scan is not costed")
        return analytic_ssd_flops(seq_len, batch, module.d_state, module.headdim,
                                  module.nheads, module.chunk_size, cb_per_head=True)
    raise TypeError(f"not an SSD mixer: {cls}")


def ssd_topup(model, shapes, flop_counts):
    """Analytic FLOPs for every hooked SSD mixer that ran."""
    total = 0
    mods = dict(model.named_modules())
    for name, calls in shapes.items():
        conv_name = f"{name}.conv1d"
        conv_counted = any(
            (key == conv_name or key.split(".", 1)[-1] == conv_name) and sum(ops.values())
            for key, ops in flop_counts.items()
        ) if hasattr(mods[name], "conv1d") else True
        for shape in calls:
            total += ssd_flops_for_module(mods[name], shape, conv_counted=conv_counted)
    return total


def fft_topup(model, shapes):
    """Analytic FLOPs for every hooked FFT-convolution module that ran."""
    total = 0
    mods = dict(model.named_modules())
    for name, calls in shapes.items():
        sub = mods[name]
        cls = type(sub).__name__
        for shape in calls:
            if cls in _FFT_CONV_CLASSES:
                total += analytic_fftconv_flops(sub, shape)
            elif cls in _S4ND_LAYER_CLASSES:
                total += analytic_s4nd_flops(sub, shape)
    return total


def blind_parameterised_modules(model, flop_counts):
    """Submodules that own weights yet contributed zero counted FLOPs.

    The previous detector listed module origins it already knew about
    (mamba_ssm, triton). That could only ever catch blind spots someone had
    already found: the fused cuDNN RNN and the eval-mode TransformerEncoder
    fast path both sailed through it, and every affected run reported
    `status: measured` with an empty uncounted list. Asking "did this module
    with weights contribute anything?" catches those two and whatever the next
    fused kernel turns out to be.
    """
    blind = []
    for name, sub in model.named_modules():
        # TWO checks, because they catch different failures and neither
        # subsumes the other. Origin first: a module can contribute SOME
        # counted FLOPs -- Mamba's in/out projections are ordinary Linears --
        # while its dominant term, the SSD scan, runs as raw Triton launches
        # that bypass the dispatcher entirely. Zero-contribution testing sees
        # that module as counted and waves it through, which is how replacing
        # the old origin check with a "more general" one silently stopped
        # flagging every Mamba row.
        origin = getattr(type(sub), "__module__", "") or ""
        if origin.startswith("mamba_ssm") or ".ops.triton" in origin:
            blind.append(f"{name}:{type(sub).__name__}")
            continue
        if not name or not any(True for _ in sub.parameters(recurse=False)):
            continue
        # Normalisation and lookup layers are SUPPOSED to count zero -- they do
        # elementwise or gather work, which FlopCounterMode never counts by
        # design (see COUNTING CONVENTION above). Flagging them made 223 of 225
        # rows "partial" and buried the two blind spots that matter. Only
        # matmul-class modules are expected to contribute.
        if isinstance(sub, _EXPECTED_ZERO) or "Norm" in type(sub).__name__:
            continue
        # A child of a fused op is not blind, it is ATTRIBUTED to the parent.
        # nn.MultiheadAttention counts its whole forward -- out_proj included --
        # against itself, so out_proj's own line reads zero while its 4,718,592
        # FLOPs are already in the total. Flagging it marked 45 rows partial for
        # arithmetic that was never missing.
        if any(isinstance(anc, torch.nn.MultiheadAttention)
               for anc_name, anc in model.named_modules()
               if anc_name and name.startswith(anc_name + ".")):
            continue
        counted = 0
        for key, ops in flop_counts.items():
            tail = key.split(".", 1)[-1]
            if tail == name or key == name:
                counted += sum(ops.values())
        if counted == 0:
            blind.append(f"{name}:{type(sub).__name__}")
    return blind


def forecast_normalized_flops(flops, batch):
    """Return forecast count and FLOPs per forecast for one measured batch.

Measured on a WARMED forward, so one-time setup is excluded -- see cost_one.

    That correction matters most for S4. Reading a cold forward suggested the
    metric was batch-dependent for it (40,213,504 per forecast at batch 64
    against 11,577,936 at 2048, a 3.5x spread while GRU and Transformer held
    constant). It is not: the first forward builds the DPLR kernel and every
    one after reuses it, so three consecutive calls at batch 64 give
    2,573,664,256 / 681,869,312 / 681,869,312. Warmed, S4 sits at 10,654,208
    per forecast at both batch sizes -- a model property like the others.

    The one-time cost is real and is kept in `one_time_setup_flops`; it is
    simply not a per-forecast figure.
    """
    n_forecasts = int(batch["target_value"].numel())
    if n_forecasts <= 0:
        raise ValueError("cost batch contains no forecasts")
    return n_forecasts, (flops / n_forecasts if flops is not None else None)


def cost_exit_code(failures, allow_failures=False):
    """Fail official cost generation when any required cell was unmeasured."""
    return 1 if failures and not allow_failures else 0


def cost_one(model, variant, enc, combo, cl, dev, fbs, cbs, axis_identity=False):
    m = _build(model, variant, enc, combo, cl, axis_identity=axis_identity).to(dev).eval()
    params = sum(p.numel() for p in m.parameters())
    _, _, test = cl.get_dataloaders(batch_size=(cbs if combo else fbs), num_workers=0,
                                    combo=combo, shuffle_train=False)
    b = next(iter(test))
    x = b["x_numeric"].to(dev)
    args = (x,) if combo else (x, b["state_ids"].to(dev), b["comm_ids"].to(dev), b["flow_ids"].to(dev))
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    flops = None
    # NOT `torch.no_grad()`. FlopCounterMode observes dispatched ATen ops, and
    # eval + no_grad sends nn.TransformerEncoder down its fused fast path, which
    # has no registered formula: the whole encoder stack counted as EXACTLY
    # zero, so a Transformer's forward figure was its input projection and head
    # alone -- 1.2M against the ~61M its own training-time counter implies.
    # Enabling grad restores the ordinary op sequence. Measured on torch 2.9.1:
    # 0 FLOPs under eval+no_grad, 3.62 GFLOPs under eval+grad, same module and
    # input. Backward is never run, so this is still a forward-only count.
    blind = []
    topup = 0
    fft_top = 0
    ssd_top = 0
    attn_top = 0
    param_only = []
    warmup_only = 0
    rnn_shapes, rnn_handles = rnn_input_shapes(m)
    fft_shapes, fft_handles = fft_input_shapes(m)
    ssd_shapes, ssd_handles = ssd_input_shapes(m)
    mha_shapes, mha_handles = mha_input_shapes(m)
    with torch.enable_grad():
        if HAVE_FLOP:
            # Measure the SECOND forward. The first one pays every one-time
            # setup cost the module has -- S4 builds its DPLR kernel once and
            # caches it, and a cold forward therefore counted 2,573,664,256
            # against 681,869,312 for the identical call right after, a 3.8x
            # overstatement that landed on every S4 and S4ND row because each
            # cell builds a fresh model and measured it once.
            #
            # Warming up rather than special-casing S4 means lazy inits, cached
            # buffers and anything else amortised gets excluded too, without
            # this function needing to know which models do it.
            cold = FlopCounterMode(display=False)
            with cold:
                m(*args)
            # The hooks saw the warm-up too. Now that they accumulate every
            # call, the shapes must describe the measured forward alone.
            for seen in (rnn_shapes, fft_shapes, ssd_shapes, mha_shapes):
                seen.clear()
            fc = FlopCounterMode(display=False)
            with fc:
                m(*args)
            flops = fc.get_total_flops()
            # Not discarded silently: a large gap is a real property of the
            # model (a kernel someone has to build before the first forecast),
            # it just is not a per-forecast cost.
            warmup_only = max(0, cold.get_total_flops() - flops)
            blind = blind_parameterised_modules(m, fc.get_flop_counts())
            # Recurrent cores are computed, not measured, whenever the counter
            # missed them -- the analytic form is exact and does not move with
            # the backend or the torch version. Anything else that comes back
            # blind stays in the list and downgrades the row to "partial": a
            # number nobody can reproduce is worse than a flagged gap.
            seq_len = x.shape[1]
            batch_n = x.shape[0]
            # FFT convolutions likewise: S4's core is aten::_fft_r2c/_fft_c2r,
            # which has no formula, so a warmed S4 block counted only its
            # Linears. Computed from the shape each module received.
            fft_top = fft_topup(m, fft_shapes)
            # And the SSD scan: Mamba's in/out projections are Linears and
            # count, the chunked scan is raw Triton and does not.
            ssd_top = ssd_topup(m, ssd_shapes, fc.get_flop_counts())
            # And attention cores the backend hid (CPU flash path, see
            # _ATTN_COUNTED_OPS). Zero wherever the counter saw them.
            attn_top = attention_topup(m, mha_shapes, fc.get_flop_counts())
            still_blind = []
            for entry in blind:
                name, cls = entry.split(":", 1)
                sub = dict(m.named_modules()).get(name)
                if type(sub) in _RNN_GATES:
                    # batch_first=True, so the captured shape is (N, L, H).
                    for got in rnn_shapes.get(name, [(batch_n, seq_len)]):
                        topup += analytic_rnn_flops(sub, got[0], got[1])
                elif cls in _FFT_CONV_CLASSES or cls in _S4ND_LAYER_CLASSES:
                    pass                      # covered by fft_top above
                elif cls.startswith(_PARAM_ONLY_PREFIXES):
                    param_only.append(entry)  # cacheable, excluded by convention
                elif cls in _SSD_CLASSES and name in ssd_shapes:
                    pass                      # covered by ssd_top above
                elif "Norm" in cls:
                    pass                      # mamba_ssm's gated RMSNorm: elementwise
                else:
                    still_blind.append(entry)
            blind = still_blind
            flops += topup + fft_top + ssd_top + attn_top
            for h in rnn_handles + fft_handles + ssd_handles + mha_handles:
                h.remove()
        else:
            m(*args)
    peak = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else float("nan")
    n_forecasts, flops_per_forecast = forecast_normalized_flops(flops, b)
    del m
    if dev == "cuda":
        torch.cuda.empty_cache()
    return (params, flops, n_forecasts, flops_per_forecast, peak, blind,
            topup, warmup_only, fft_top, param_only, ssd_top, attn_top)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/census_port/processed/census_lattice_9ch.npz")
    ap.add_argument("--flat-batch", type=int, default=2048)
    ap.add_argument("--combo-batch", type=int, default=1)
    ap.add_argument("--aggregate-batch", type=int, default=32)
    ap.add_argument("--out-dir", default="outputs/sweep")
    ap.add_argument("--allow-failures", action="store_true",
                    help="diagnostic only: write failed rows but return success")
    ap.add_argument("--only", default=None,
                    help="regex on 'model:variant'; other cells are skipped. The "
                         "output is then a PARTIAL panel to merge, not a certified one")
    a = ap.parse_args()
    only = re.compile(a.only) if a.only else None
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # Same composed data contract as training/evaluation (see F7/F10): the
    # cost panel must describe the model on the window it actually runs.
    data_cfg = compose_data_config()
    cl = CensusLattice(census_config_from_config(data_cfg, npz=a.npz))
    cl_agg = CensusLattice(census_config_from_config(
        data_cfg, npz=a.npz, aggregate=True))
    print(f"cost panel | device={dev} | flops={'yes' if HAVE_FLOP else 'NO'} | "
          f"batch flat={a.flat_batch} combo={a.combo_batch}")
    rows = []
    failures = []

    def gflops(fl):
        return f"{fl/1e9:.2f}G" if fl is not None else "n/a"

    def row(model, variant, enc, res):
        p, fl, nfc, fl_fc, mem, blind, topup, warm, fft, param_only, ssd, attn = res
        return (model, variant, enc, p, fl or "", nfc, fl_fc or "", f"{mem:.3f}",
                topup, fft, ssd, attn, warm, ";".join(param_only), ";".join(blind),
                "partial" if blind else "ok", "")

    def failed_row(model, variant, enc, e):
        return (model, variant, enc, "", "", "", "", "", "", "", "", "", "", "", "",
                "failed", f"{type(e).__name__}: {e}")

    def wanted(model, variant):
        return only is None or only.search(f"{model}:{variant}") is not None

    def show(model, variant, enc, res):
        p, fl, _, fl_fc, mem = res[:5]
        print(f"  {model:12}{variant:20}{enc:8} params {p:>10,}  "
              f"fwd {gflops(fl):>9}  per-fc {gflops(fl_fc):>9}  mem {mem:.2f}GB")

    for model, enc in FLAT:
        if not wanted(model, enc):
            continue
        try:
            res = cost_one(model, enc, enc, False, cl, dev, a.flat_batch, a.combo_batch)
            rows.append(row(model, enc, enc, res))
            show(model, enc, "flat", res)
        except Exception as e:
            print(f"  {model:12}{enc:20}flat   FAIL {type(e).__name__}: {e}")
            rows.append(failed_row(model, enc, enc, e))
            failures.append((model, enc, enc, e))
    for model, variant, enc in AGGREGATE:
        if not wanted(model, variant):
            continue
        try:
            res = cost_one(model, variant, enc, False, cl_agg, dev,
                           a.aggregate_batch, a.combo_batch)
            rows.append(row(model, variant, enc, res))
            show(model, variant, enc, res)
        except Exception as e:
            print(f"  {model:12}{variant:20}{enc:8} FAIL {type(e).__name__}: {e}")
            rows.append(failed_row(model, variant, enc, e))
            failures.append((model, variant, enc, e))
    for model, variant, enc in COMBO:
        if not wanted(model, variant):
            continue
        try:
            res = cost_one(model, variant, enc, True, cl, dev, a.flat_batch, a.combo_batch)
            rows.append(row(model, variant, enc, res))
            show(model, variant, enc, res)
        except Exception as e:
            print(f"  {model:12}{variant:20}{enc:8} FAIL {type(e).__name__}: {str(e)[:60]}")
            rows.append(failed_row(model, variant, enc, e))
            failures.append((model, variant, enc, e))

    for model, variant, enc in COMBO_ID:
        reported_variant = f"{variant}_id"
        if not wanted(model, reported_variant):
            continue
        try:
            res = cost_one(model, variant, enc, True, cl, dev, a.flat_batch,
                           a.combo_batch, axis_identity=True)
            rows.append(row(model, reported_variant, enc, res))
            show(model, reported_variant, enc, res)
        except Exception as e:
            print(f"  {model:12}{reported_variant:20}{enc:8} "
                  f"FAIL {type(e).__name__}: {str(e)[:60]}")
            rows.append(failed_row(model, reported_variant, enc, e))
            failures.append((model, reported_variant, enc, e))

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "model_cost.csv"
    with output.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "variant", "enc", "params", "flops_fwd_total",
                    "n_forecasts", "flops_per_forecast", "peak_mem_GB",
                    "analytic_rnn_topup", "analytic_fft_topup",
                    "analytic_ssd_topup", "analytic_attn_topup",
                    "one_time_setup_flops", "param_only_uncounted",
                    "blind_modules", "status", "error"])
        w.writerows(rows)
    print(f"-> {output} ({len(rows)} rows, {len(failures)} failed)")
    if failures and not a.allow_failures:
        print("cost panel incomplete; rerun with --allow-failures only for diagnostics")
    return cost_exit_code(failures, a.allow_failures)


if __name__ == "__main__":
    sys.exit(main())
