"""The cost panel's analytic top-ups: exact where the counter can be checked,
and applied only to the modules they describe."""
import math

import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode

from scripts.model_cost import (analytic_fftconv_flops, analytic_rnn_flops,
                                analytic_s4nd_flops, blind_parameterised_modules,
                                fft_input_shapes, fft_topup, rnn_input_shapes,
                                _PARAM_ONLY_PREFIXES, _rfft_flops)


def _counted(model, *args):
    fc = FlopCounterMode(display=False)
    with torch.enable_grad(), fc:
        model(*args)
    return fc


def test_rnn_topup_is_exact_where_the_counter_still_sees_the_module():
    gru = torch.nn.GRU(35, 128, num_layers=4, batch_first=True)
    x = torch.randn(8, 36, 35)
    fc = _counted(gru, x)
    counted = fc.get_total_flops()
    if counted == 0:
        pytest.skip("this torch build does not count the GRU on CPU; nothing to compare")
    assert analytic_rnn_flops(gru, 8, 36) == counted


def test_rnn_hook_reports_the_folded_batch_the_module_actually_sees():
    class Fold(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = torch.nn.GRU(4, 8, batch_first=True)

        def forward(self, x):                 # (B, L, G, F) -> RNN over B*G
            b, l, g, f = x.shape
            y, _ = self.rnn(x.permute(0, 2, 1, 3).reshape(b * g, l, f))
            return y

    m = Fold()
    shapes, handles = rnn_input_shapes(m)
    m(torch.randn(1, 5, 7, 4))
    for h in handles:
        h.remove()
    assert shapes["rnn"] == [(7, 5, 4)]


def test_fftconv_topup_follows_the_forward_it_describes():
    from src.models.s4 import _s4mod
    conv = _s4mod.FFTConv(16, transposed=False)          # standalone vendored class
    x = torch.randn(3, 10, 16)
    shapes, handles = fft_input_shapes(conv)
    with torch.enable_grad():
        conv(x)
    for h in handles:
        h.remove()
    assert shapes[""] == [(3, 10, 16)]
    n = 20                                                # l_kernel + L, both 10
    expect = (3 * 16 * _rfft_flops(n)                     # rfft(x)
              + 3 * 1 * 16 * (n // 2 + 1) * 6             # complex multiply
              + 3 * 1 * 16 * _rfft_flops(n)               # irfft
              + 3 * 1 * 16 * 10 * 2)                      # D skip
    assert analytic_fftconv_flops(conv, (3, 10, 16)) == int(round(expect))
    assert _rfft_flops(n) == pytest.approx(2.5 * n * math.log2(n))


def test_s4nd_topup_treats_every_other_axis_as_batch():
    from src.models.s4nd import S4NDLayer
    layer = S4NDLayer(8, axis_lens=[4, 6], d_state=8)
    shape = (2, 3, 4, 6, 1, 8)                           # (B, T, S, C, Fl, H)
    numel = math.prod(shape)
    expect = 0
    for seq in (4, 6):
        n, rows = 2 * seq, numel // (8 * seq)
        expect += rows * 8 * (2 * _rfft_flops(n) + (n // 2 + 1) * 6)
    expect += numel * 2
    assert analytic_s4nd_flops(layer, shape) == int(round(expect))


def test_s4_blind_list_resolves_to_fft_topup_and_param_only_kernel():
    """A warmed S4 forward counts only its Linears. The FFTConv and its DPLR
    kernel come back blind; the first is computed, the second is the
    parameter-only exclusion, and neither should read as an unexplained gap."""
    from src.models.s4 import _s4mod
    m = torch.nn.Sequential(_s4mod.S4Block(16, d_state=8, transposed=False))
    x = torch.randn(2, 12, 16)
    shapes, handles = fft_input_shapes(m)
    _counted(m, x)                                        # warm-up
    shapes.clear()                                        # as cost_one does
    fc = _counted(m, x)
    for h in handles:
        h.remove()
    blind = blind_parameterised_modules(m, fc.get_flop_counts())
    classes = {e.split(":", 1)[1] for e in blind}
    assert "FFTConv" in classes
    assert any(c.startswith(_PARAM_ONLY_PREFIXES) for c in classes)
    assert fft_topup(m, shapes) == analytic_fftconv_flops(m[0].layer, (2, 12, 16))
    assert fft_topup(m, shapes) > 0


# ---- the SSD scan -----------------------------------------------------------
import importlib.util
from pathlib import Path

from scripts.model_cost import (analytic_ssd_flops, ssd_flops_for_module,
                                ssd_input_shapes, ssd_topup)

_SSD_MIN = Path(__file__).resolve().parents[1] / "external/mamba/mamba_ssm/modules/ssd_minimal.py"


def _ssd_minimal():
    """The vendored reference, exec'd from source with its one triton import
    (used only by its own self-test) dropped, so it runs on a CPU box."""
    src = "\n".join(line for line in _SSD_MIN.read_text().splitlines()
                     if not line.startswith("from mamba_ssm"))
    ns = {}
    exec(compile(src, str(_SSD_MIN), "exec"), ns)
    return ns["ssd_minimal_discrete"]


def chunked_ssd(X, A, B, C, chunk, cb_per_head=True):
    """The chunked algorithm the Triton kernels run, written as the four
    matmuls the count is made of. X (b,L,h,p); A (b,L,h) log-decay; B, C
    (b,L,h,n). With cb_per_head=False, C B^T is formed once (one group) and
    shared across heads, as ssd_bmm does for Mamba-2."""
    b, L, h, p = X.shape
    n = B.shape[-1]
    y = torch.zeros_like(X)
    state = X.new_zeros(b, h, p, n)
    for start in range(0, L, chunk):
        sl = slice(start, min(L, start + chunk))
        Xc, Bc, Cc = X[:, sl].permute(0, 2, 1, 3), B[:, sl].permute(0, 2, 1, 3), C[:, sl].permute(0, 2, 1, 3)
        cum = torch.cumsum(A[:, sl], dim=1).permute(0, 2, 1)          # (b,h,lc)
        lc = cum.shape[-1]
        mask = torch.tril(torch.ones(lc, lc, dtype=torch.bool))
        Lmat = torch.exp(cum[..., :, None] - cum[..., None, :]) * mask
        if cb_per_head:
            CB = torch.matmul(Cc, Bc.transpose(-1, -2))               # (b,h,lc,lc)
        else:
            CB = torch.matmul(Cc[:, :1], Bc[:, :1].transpose(-1, -2)).expand(b, h, lc, lc)
        Yd = torch.matmul(CB * Lmat, Xc)                              # (b,h,lc,p)
        Yo = torch.matmul(Cc, state.transpose(-1, -2)) * torch.exp(cum)[..., None]
        y[:, sl] = (Yd + Yo).permute(0, 2, 1, 3)
        decay = torch.exp(cum[..., -1:] - cum)                        # (b,h,lc)
        BX = torch.matmul((Bc * decay[..., None]).transpose(-1, -2), Xc)   # (b,h,n,p)
        state = state * torch.exp(cum[..., -1])[..., None, None] + BX.transpose(-1, -2)
    return y


def _ssd_inputs(b, L, h, p, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(b, L, h, p, generator=g)
    A = -torch.rand(b, L, h, generator=g)
    B = torch.randn(b, L, h, n, generator=g)
    C = torch.randn(b, L, h, n, generator=g)
    return X, A, B, C


def test_chunked_ssd_reference_reproduces_the_vendored_minimal_ssd():
    X, A, B, C = _ssd_inputs(2, 64, 3, 4, 8)
    ref = _ssd_minimal()(X, A, B, C, block_len=16)[0]
    ours = chunked_ssd(X, A, B, C, chunk=16)
    assert torch.allclose(ours, ref, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("L,chunk", [(64, 16), (36, 16), (36, 64), (1343, 256)])
def test_ssd_count_matches_the_counter_on_the_reference_algorithm(L, chunk):
    b, h, p, n = 2, 3, 4, 8
    X, A, B, C = _ssd_inputs(b, L, h, p, n)
    for per_head in (True, False):
        fc = FlopCounterMode(display=False)
        with fc:
            chunked_ssd(X, A, B, C, chunk, cb_per_head=per_head)
        assert fc.get_total_flops() == analytic_ssd_flops(
            L, b, n, p, h, chunk, ngroups=1, cb_per_head=per_head)


class _FakeMamba2(torch.nn.Module):
    """Carries what ssd_flops_for_module reads; does no scan itself."""
    d_state, headdim, nheads, ngroups, chunk_size = 64, 32, 8, 1, 256

    def __init__(self):
        super().__init__()
        self.in_proj = torch.nn.Linear(16, 648)
        self.conv1d = torch.nn.Conv1d(384, 384, 4, groups=384, padding=3)
        self.out_proj = torch.nn.Linear(256, 16)
        self.A_log = torch.nn.Parameter(torch.zeros(8))

    def forward(self, u):
        zxbcdt = self.in_proj(u)
        xbc = self.conv1d(zxbcdt[..., 256:640].transpose(1, 2))[..., :u.shape[1]].transpose(1, 2)
        return self.out_proj(torch.cat([xbc[..., :256]], -1))


_FakeMamba2.__module__ = "mamba_ssm.modules.mamba2"
_FakeMamba2.__name__ = _FakeMamba2.__qualname__ = "Mamba2"


def test_ssd_topup_uses_the_folded_shape_and_only_adds_the_conv_when_uncounted():
    m = torch.nn.Sequential(_FakeMamba2())
    shapes, handles = ssd_input_shapes(m)
    fc = _counted(m, torch.randn(5, 36, 16))
    for h in handles:
        h.remove()
    assert shapes["0"] == [(5, 36, 16)]
    expect = analytic_ssd_flops(36, 5, 64, 32, 8, 256, ngroups=1)
    assert ssd_topup(m, shapes, fc.get_flop_counts()) == expect      # conv1d was counted
    assert ssd_flops_for_module(m[0], (5, 36, 16), conv_counted=False) \
        == expect + 2 * 36 * 384 * 4 * 5
    assert "0:Mamba2" in blind_parameterised_modules(m, fc.get_flop_counts())


def test_ssd_count_on_the_paper_shapes_is_a_third_of_the_projections():
    """Sanity on the roster's own geometry: d_model 128, 8 heads of 32, N=64,
    L=36. The scan is not a rounding error next to the Linears."""
    scan = analytic_ssd_flops(36, 1, 64, 32, 8, 256, ngroups=1)
    projections = 2 * 36 * (128 * 648 + 256 * 128)
    assert 0.3 < scan / projections < 0.5


# ---- attention cores the backend hides ---------------------------------------
from scripts.model_cost import (analytic_attention_core_flops, attention_topup,
                                mha_input_shapes)


def test_attention_topup_fills_the_core_only_when_the_counter_missed_it():
    mha = torch.nn.MultiheadAttention(128, 4, batch_first=True).eval()
    x = torch.randn(8, 36, 128)
    shapes, handles = mha_input_shapes(mha)
    fc = _counted(mha, x, x, x)
    for h in handles:
        h.remove()
    assert shapes[""] == [(8, 36, 36)]
    projections = 4 * 2 * 8 * 36 * 128 * 128
    core = 4 * 8 * 36 * 36 * 128
    assert analytic_attention_core_flops(mha, (8, 36, 36)) == core
    counted = fc.get_total_flops()
    top = attention_topup(mha, shapes, fc.get_flop_counts())
    # Whichever path this backend took, the sum is the whole module.
    assert counted in (projections, projections + core)
    assert counted + top == projections + core


def test_attention_topup_covers_every_layer_of_a_transformer_encoder():
    layer = torch.nn.TransformerEncoderLayer(64, 4, 128, batch_first=True, norm_first=True)
    enc = torch.nn.TransformerEncoder(layer, 3).eval()
    x = torch.randn(2, 12, 64)
    shapes, handles = mha_input_shapes(enc)
    fc = _counted(enc, x)
    for h in handles:
        h.remove()
    assert sorted(shapes) == ["layers.0.self_attn", "layers.1.self_attn", "layers.2.self_attn"]
    top = attention_topup(enc, shapes, fc.get_flop_counts())
    assert top in (0, 3 * 4 * 2 * 12 * 12 * 64)


def test_a_mixer_called_in_chunks_is_costed_for_every_chunk():
    """The Mamba hosts push the folded batch through the mixer in 8,192-row
    slices; one forward is several calls, and every one counts."""
    class Chunked(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mixer = _FakeMamba2()

        def forward(self, x):
            return torch.cat([self.mixer(x[i:i + 4]) for i in range(0, x.shape[0], 4)])

    m = Chunked()
    shapes, handles = ssd_input_shapes(m)
    fc = _counted(m, torch.randn(10, 36, 16))                # 4 + 4 + 2 rows
    for h in handles:
        h.remove()
    assert [s[0] for s in shapes["mixer"]] == [4, 4, 2]
    assert ssd_topup(m, shapes, fc.get_flop_counts()) == analytic_ssd_flops(36, 10, 64, 32, 8, 256)
