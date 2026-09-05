"""Encoders that interface between input data and model."""

import datetime
import math
from typing import ForwardRef

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

import src.models.nn.utils as U
import src.utils as utils
import src.utils.config
from src.models.sequence.backbones.block import SequenceResidualBlock
from src.models.nn import Normalization

class Encoder(nn.Module):
    """Encoder abstraction.

    Accepts a tensor and optional kwargs. Outside of the main tensor, all other arguments should be kwargs.
    Returns a tensor and optional kwargs.
    Encoders are combined via U.PassthroughSequential which passes these kwargs through in a pipeline. The resulting kwargs are accumulated and passed into the model backbone.

    """

    def forward(self, x, **kwargs):
        """
        x: input tensor
        *args: additional info from the dataset (e.g. sequence lengths)

        Returns:
        y: output tensor
        *args: other arguments to pass into the model backbone
        """
        return x, {}



# Adapted from https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class PositionalEncoder(Encoder):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoder(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=16384, pe_init=None):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        if pe_init is not None:
            self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))
            nn.init.normal_(self.pe, 0, pe_init)
            # self.pe = pe.unsqueeze(1)
        else:
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0.0, max_len).unsqueeze(1)
            div_term = torch.exp(
                -math.log(10000.0) * torch.arange(0.0, d_model, 2.0) / d_model
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe)

        self.attn_mask = None

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
            lens: actual lengths of sequences
        Shape:
            x: [l_sequence, n_batch, d_model]
            Returns: [l_sequence, n_batch, d_model]
            attn_mask: [l_sequence, l_sequence]
            padding_mask:
        """

        x = x + self.pe[: x.size(-2)]
        return self.dropout(x)


class ClassEmbedding(Encoder):
    # Should also be able to define this by subclassing Embedding
    def __init__(self, n_classes, d_model):
        super().__init__()
        self.embedding = nn.Embedding(n_classes, d_model)

    def forward(self, x, y):
        x = x + self.embedding(y).unsqueeze(-2)  # (B, L, D)
        return x


class Conv1DEncoder(Encoder):
    def __init__(self, d_input, d_model, kernel_size=25, stride=1, padding='same'):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=d_input,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        # BLD -> BLD
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return x

class LayerEncoder(Encoder):
    """Use an arbitary SequenceModule layer"""

    def __init__(self, d_model, prenorm=False, norm='layer', layer=None):
        super().__init__()

        # Simple stack of blocks
        layer["transposed"] = False
        self.layer = SequenceResidualBlock(
            d_input=d_model,
            prenorm=prenorm,
            layer=layer,
            residual='R',
            norm=norm,
            pool=None,
        )

    def forward(self, x):
        x, _ = self.layer(x) # Discard state
        return x


class TimestampEmbeddingEncoder(Encoder):
    """
    General time encoder for Pandas Timestamp objects (encoded as torch tensors).
    See MonashDataset for an example of how to return time features as 'z's.
    """

    cardinalities = {
        'day': (1, 31),
        'hour': (0, 23),
        'minute': (0, 59),
        'second': (0, 59),
        'month': (1, 12),
        'year': (1950, 2010), # (1800, 3000) used to be (1970, datetime.datetime.now().year + 1) but was not enough for all datasets in monash
        'dayofweek': (0, 6),
        'dayofyear': (1, 366),
        'quarter': (1, 4),
        'week': (1, 53),
        'is_month_start': (0, 1),
        'is_month_end': (0, 1),
        'is_quarter_start': (0, 1),
        'is_quarter_end': (0, 1),
        'is_year_start': (0, 1),
        'is_year_end': (0, 1),
        'is_leap_year': (0, 1),
    }

    def __init__(self, d_model, table=False, features=None):
        super().__init__()
        self.table = table
        self.ranges = {k: max_val - min_val + 2 for k, (min_val, max_val) in self.cardinalities.items()} # padding for null included

        if features is None:
            pass
        else:
            self.cardinalities = {k: v for k, v in self.cardinalities.items() if k in features}

        if table:
            self.embedding = nn.ModuleDict({
                attr: nn.Embedding(maxval - minval + 2, d_model, padding_idx=0)
                for attr, (minval, maxval) in self.cardinalities.items()
            })
        else:
            self.embedding = nn.ModuleDict({
                attr: nn.Linear(1, d_model)
                for attr in self.cardinalities
            })



    def forward(self, x, timestamps=None):
        for attr in timestamps:
            mask = timestamps[attr] == -1
            timestamps[attr] = timestamps[attr] - self.cardinalities[attr][0]
            timestamps[attr][mask] = 0
            if self.table:
                x = x + self.embedding[attr](timestamps[attr].to(torch.long))
            else:
                x = x + self.embedding[attr]((2 * timestamps[attr] / self.ranges[attr] - 1).unsqueeze(-1))

            #x = x + self.embedding(timestamps[attr].to(torch.float)).unsqueeze(1)
        return x

# TODO is this used anymore?
class TSIndexEmbeddingEncoder(Encoder):
    """
    Embeds location of sample in the time series
    """

    def __init__(self, n_ts, d_model, table=True):
        super().__init__()

        self.table = table
        self.n_ts = n_ts
        if table:
            self.embedding = nn.Embedding(n_ts, d_model)
        else:
            # self.embedding = nn.Linear(1, d_model)
            # self.linear = nn.Linear(2 * d_model, d_model)

            self.linear = nn.Linear(d_model + 1, d_model)

    def forward(self, x, z=None, idxs=None):
        if self.table:
            x = x + self.embedding(idxs.to(torch.long)).unsqueeze(1)
        else:
            # x = self.linear(torch.cat([x, self.embedding((2 * idxs / self.n_ts - 1)[:, None, None]).repeat((1, x.shape[1], 1))], axis=-1))
            x = self.linear(torch.cat([x, ((2 * idxs / self.n_ts - 1)[:, None, None]).repeat((1, x.shape[1], 1))], axis=-1))
        #x = x + self.embedding(idxs.unsqueeze(1).to(torch.float)).unsqueeze(1)
        return x


class TimeEncoder(Encoder):
    def __init__(self, n_tokens_time, d_model, timeenc=0):
        super().__init__()

        self.timeenc = timeenc
        if self.timeenc == 0:
            self.encoders = nn.ModuleList(
                [nn.Embedding(v, d_model) for v in n_tokens_time]
            )
        else:
            self.encoders = nn.Linear(len(n_tokens_time), d_model)
        self.mask_embed = nn.Embedding(2, d_model)

    def forward(self, x, mark=None, mask=None):
        assert mark is not None and mask is not None, "Extra arguments should be returned by collate function"
        if self.timeenc == 0:
            assert mark.size(-1) == len(self.encoders)
            embeddings = [
                embed(z) for embed, z in zip(self.encoders, torch.unbind(mark, dim=-1))
            ]
            time_encode = torch.sum(torch.stack(embeddings), dim=0)
        else:
            time_encode = self.encoders(mark)
        mask_encode = self.mask_embed(mask.squeeze(-1))
        return x + time_encode + mask_encode  # (B, L, d_model)

class EEGAgeEncoder(Encoder):
    def __init__(self, d_model):
        super().__init__()
        self.encoder = nn.Linear(1, d_model)

    def forward(self, x, age=None):
        z = self.encoder(((age - 50.0) / 100.0).unsqueeze(1))
        return x + z.unsqueeze(1)

class PackedEncoder(Encoder):
    def forward(self, x, len_batch=None):
        assert len_batch is not None
        x = nn.utils.rnn.pack_padded_sequence(
            x, len_batch.cpu(), enforce_sorted=False, batch_first=True,
        )
        return x


class OneHotEncoder(Encoder):
    def __init__(self, n_tokens, d_model):
        super().__init__()
        assert n_tokens <= d_model
        self.d_model = d_model

    def forward(self, x):
        return F.one_hot(x.squeeze(-1), self.d_model).float()


class CFBagEncoder(Encoder):
    """
    Concatenate Commodity×Flow bag embeddings to the last feature dim.
    Expects kwargs:
      - comm_counts: (B, S, L, n_comms) multi-hot or counts
      - flow_counts: (B, S, L, 2) counts/one-hot for Export/Import
    """
    def __init__(self, n_comms: int, d_comm: int = 32, project_flow: bool = False, d_flow: int = 2):
        super().__init__()
        self.proj_comm = nn.Linear(n_comms, d_comm, bias=False)
        # Repo-wide standard init for category projections: normal(0.02)
        nn.init.normal_(self.proj_comm.weight, std=0.02)
        self.project_flow = project_flow
        self.flow = nn.Identity() if not project_flow else nn.Linear(2, d_flow, bias=False)
        if project_flow:
            nn.init.normal_(self.flow.weight, std=0.02)
        self.d_comm = d_comm
        self.d_flow = d_flow if project_flow else 2

    def forward(self, x, comm_counts=None, flow_counts=None):
        assert comm_counts is not None and flow_counts is not None, "CFBagEncoder requires comm_counts and flow_counts"
        # Shapes: x (B, ..., F), comm_counts (B, S, L, C), flow_counts (B, S, L, 2)
        e_comm = self.proj_comm(comm_counts.float())  # (B,S,L, d_comm)
        e_flow = self.flow(flow_counts.float())       # (B,S,L, d_flow or 2)
        x = torch.cat([x, e_comm, e_flow], dim=-1)
        return x


class CFIDEncoder(Encoder):
    """
    Append per-sequence embeddings using (commodity_ids, flow_ids).
    Expects kwargs:
      - commodity_ids: (B, S, G) long
      - flow_ids: (B, S, G) long with values {0:Export,1:Import}
    x is (B, S, G, L, F). Embeddings are broadcast across L and concatenated on the last dim.
    """
    def __init__(
        self,
        n_comms: int,
        d_comm: int = 32,
        learn_flow: bool = True,
        d_flow: int = 2,
        flow_init: str = 'normal',  # 'identity' | 'normal'
        flow_std: float = 0.02,
    ):
        super().__init__()
        self.comm = nn.Embedding(n_comms, d_comm)
        # Repo-wide embedding standard: comm=32 dims, normal(0.02)
        nn.init.normal_(self.comm.weight, std=0.02)
        if learn_flow:
            self.flow = nn.Embedding(2, d_flow)
            if flow_init == 'identity' and d_flow == 2:
                with torch.no_grad():
                    self.flow.weight.copy_(torch.eye(2))
            elif flow_init == 'normal':
                nn.init.normal_(self.flow.weight, std=flow_std)
            else:
                # default PyTorch initialization
                pass
        else:
            self.flow = None
        self.d_comm = d_comm
        self.d_flow = d_flow

    def forward(self, x, commodity_ids=None, flow_ids=None):
        assert commodity_ids is not None and flow_ids is not None, "CFIDEncoder requires commodity_ids and flow_ids"
        # x: (B,S,G,L,F)
        b, s, g, l, f = x.shape
        e_comm = self.comm(commodity_ids.long())              # (B,S,G,32)
        if self.flow is None:
            e_flow = F.one_hot(flow_ids.long(), num_classes=2).float()  # (B,S,G,2)
        else:
            e_flow = self.flow(flow_ids.long())               # (B,S,G,d_flow)
        e = torch.cat([e_comm, e_flow], dim=-1)               # (B,S,G, d_comm + d_flow_or_2)
        e = e.unsqueeze(-2).expand(b, s, g, l, e.size(-1))    # (B,S,G,L, ...)
        x = torch.cat([x, e], dim=-1)
        return x


class CFStateEncoder(Encoder):
    """
    Append per-sample commodity+flow embeddings to a (B, S, L, F) State grid.

    This is the "2-D by State" *flat* variant: State is the only convolved
    categorical axis and each (Commodity, Import/Export) series is folded into the
    batch, so commodity/flow live in the features rather than on a grid axis.
    Expects kwargs:
      - commodity_ids: (B, S) long  (constant across S within a sample)
      - flow_ids: (B, S) long with values {0:Export, 1:Import}
    Embeddings are broadcast across L and concatenated on the last (feature) dim,
    giving (B, S, L, F + d_comm + d_flow). Mirrors CFIDEncoder but for the
    (B,S,L,F) rank instead of (B,S,G,L,F).
    """
    def __init__(
        self,
        n_comms: int,
        d_comm: int = 32,
        learn_flow: bool = True,
        d_flow: int = 2,
        flow_init: str = 'normal',  # 'identity' | 'normal'
        flow_std: float = 0.02,
    ):
        super().__init__()
        self.comm = nn.Embedding(n_comms, d_comm)
        # Repo-wide embedding standard: comm=32 dims, normal(0.02)
        nn.init.normal_(self.comm.weight, std=0.02)
        if learn_flow:
            self.flow = nn.Embedding(2, d_flow)
            if flow_init == 'identity' and d_flow == 2:
                with torch.no_grad():
                    self.flow.weight.copy_(torch.eye(2))
            elif flow_init == 'normal':
                nn.init.normal_(self.flow.weight, std=flow_std)
            else:
                pass
        else:
            self.flow = None
        self.d_comm = d_comm
        self.d_flow = d_flow

    def forward(self, x, commodity_ids=None, flow_ids=None):
        assert commodity_ids is not None and flow_ids is not None, "CFStateEncoder requires commodity_ids and flow_ids"
        # x: (B, S, L, F)
        b, s, l, f = x.shape
        e_comm = self.comm(commodity_ids.long())              # (B, S, d_comm)
        if self.flow is None:
            e_flow = F.one_hot(flow_ids.long(), num_classes=2).float()  # (B, S, 2)
        else:
            e_flow = self.flow(flow_ids.long())               # (B, S, d_flow)
        e = torch.cat([e_comm, e_flow], dim=-1)               # (B, S, d_comm + d_flow)
        e = e.unsqueeze(2).expand(b, s, l, e.size(-1))        # (B, S, L, ...)
        x = torch.cat([x, e], dim=-1)
        return x


class SCIDEncoder(Encoder):
    """
    Append per-sequence embeddings using (state_ids, commodity_ids).
    Expects kwargs:
      - state_ids: (B, S, C) long
      - commodity_ids: (B, S, C) long
    x is (B, S, C, L, F). Embeddings are broadcast across L and concatenated on the last dim.
    """
    def __init__(
        self,
        n_states: int,
        n_comms: int,
        d_state: int = 8,
        d_comm: int = 32,
    ):
        super().__init__()
        self.state = nn.Embedding(n_states, d_state)
        self.comm = nn.Embedding(n_comms, d_comm)
        # Repo-wide embedding standard: dims (state=8, comm=32, flow=2), normal(0.02)
        nn.init.normal_(self.state.weight, std=0.02)
        nn.init.normal_(self.comm.weight, std=0.02)
        self.d_state = d_state
        self.d_comm = d_comm

    def forward(self, x, state_ids=None, commodity_ids=None):
        # For higher-dimensional inputs where State and Commodity are explicit axes (e.g., (B,S,C,L,F)),
        # do not concatenate embeddings. Flow stays in features; SCID acts as a no-op.
        if x.dim() >= 5:
            return x

        assert state_ids is not None and commodity_ids is not None, "SCIDEncoder requires state_ids and commodity_ids"
        # Normalize id tensor shapes to (B, S, C)
        if state_ids.ndim == 2:
            state_ids = state_ids.unsqueeze(0)
        if commodity_ids.ndim == 2:
            commodity_ids = commodity_ids.unsqueeze(0)

        if x.dim() == 4:
            # Treat as (B, L, F) with per-example IDs (S=1,C=1)
            b, l, f = x.shape
            assert state_ids.shape[0] == b and commodity_ids.shape[0] == b, "state/commodity ids batch mismatch"
            e_state = self.state(state_ids.long()).view(b, 1, -1).expand(b, l, -1)
            e_comm = self.comm(commodity_ids.long()).view(b, 1, -1).expand(b, l, -1)
            x = torch.cat([x, e_state, e_comm], dim=-1)
        else:
            raise ValueError(f"SCIDEncoder expects x with >=4 dims, got shape {tuple(x.shape)}")
        return x

class FlowIDEncoder(Encoder):
    """
    Append per-sequence flow embeddings (Export/Import) as 2-d features.
    Expects kwargs:
      - flow_ids: Long tensor of shape matching leading axes of x without time/feature dims.
    For 5D inputs x: (B,S,C,L,F), flow_ids should be (B,S,C) with values {0:Export,1:Import}.
    """
    def __init__(self, d_flow: int = 2):
        super().__init__()
        self.emb = nn.Embedding(2, d_flow)
        # Repo-wide embedding standard: flow=2 dims, normal(0.02)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.d_flow = d_flow

    def forward(self, x, flow_ids=None):
        assert flow_ids is not None, "FlowIDEncoder requires flow_ids"
        if x.dim() == 5:
            b, s, c, l, f = x.shape
            e = self.emb(flow_ids.long())
            e = e.unsqueeze(-2).expand(b, s, c, l, self.d_flow)
            x = torch.cat([x, e], dim=-1)
            return x
        elif x.dim() == 6:
            b, s, c, g, l, f = x.shape
            e = self.emb(flow_ids.long())
            e = e.unsqueeze(-2).unsqueeze(-2).expand(b, s, c, g, l, self.d_flow)
            x = torch.cat([x, e], dim=-1)
            return x
        else:
            raise ValueError(f"FlowIDEncoder expects x with 5 or 6 dims, got {tuple(x.shape)}")

class TradeIDEncoder(Encoder):
    """Append per-sequence embeddings using (state_ids, comm_ids, flow_ids) for 3D inputs (B, L, F)."""

    def __init__(self, n_states: int, n_comms: int, n_flows: int = 2, d_state: int = 8, d_comm: int = 32, d_flow: int = 2):
        super().__init__()
        self.state = nn.Embedding(n_states, d_state)
        self.comm = nn.Embedding(n_comms, d_comm)
        self.flow = nn.Embedding(n_flows, d_flow)
        # Match the paper LSTM/GRU/Transformer embedding spec exactly:
        # dims (state=8, comm=32, flow=2), init normal(std=0.02).
        nn.init.normal_(self.state.weight, std=0.02)
        nn.init.normal_(self.comm.weight, std=0.02)
        nn.init.normal_(self.flow.weight, std=0.02)
        self.d_state = d_state
        self.d_comm = d_comm
        self.d_flow = d_flow

    def forward(self, x, state_ids=None, comm_ids=None, flow_ids=None):
        assert x.dim() == 3, f"TradeIDEncoder expects x of shape (B,L,F); got {tuple(x.shape)}"
        assert state_ids is not None and comm_ids is not None and flow_ids is not None, "TradeIDEncoder requires state_ids, comm_ids, flow_ids"
        b, l, _ = x.shape
        e_state = self.state(state_ids.long()).unsqueeze(1).expand(b, l, self.d_state)
        e_comm  = self.comm(comm_ids.long()).unsqueeze(1).expand(b, l, self.d_comm)
        e_flow  = self.flow(flow_ids.long()).unsqueeze(1).expand(b, l, self.d_flow)
        x = torch.cat([x, e_state, e_comm, e_flow], dim=-1)
        return x


class PlanBOneHotEncoder(Encoder):
    """Append identity one-hot vectors for state, commodity, and flow IDs to (B,L,F) inputs."""

    def __init__(self, n_states: int, n_comms: int, n_flows: int = 2):
        super().__init__()
        self.n_states = n_states
        self.n_comms = n_comms
        self.n_flows = n_flows

    def forward(self, x, state_ids=None, comm_ids=None, flow_ids=None):
        assert x.dim() == 3, f"PlanBOneHotEncoder expects x of shape (B,L,F); got {tuple(x.shape)}"
        assert state_ids is not None and comm_ids is not None and flow_ids is not None, "PlanBOneHotEncoder requires ids"
        device = x.device
        b, l, _ = x.shape

        state_idx = state_ids.long().to(device)
        comm_idx = comm_ids.long().to(device)
        flow_idx = flow_ids.long().to(device)

        e_state = F.one_hot(state_idx, num_classes=self.n_states).to(dtype=x.dtype)
        e_comm = F.one_hot(comm_idx, num_classes=self.n_comms).to(dtype=x.dtype)
        e_flow = F.one_hot(flow_idx, num_classes=self.n_flows).to(dtype=x.dtype)

        e = torch.cat([e_state, e_comm, e_flow], dim=-1).unsqueeze(1).expand(b, l, -1)
        x = torch.cat([x, e], dim=-1)
        return x

class Conv3DPatchEncoder(Encoder):
    """For encoding 3D data (e.g. videos) into a sequence of patches.

    Arguments:
      - d_emb: dim of embedding output
      - filter_sizes: tuple, with ft, fh, fw
      - max_len: int, max seq len
    """
    def __init__(self, d_emb, filter_sizes, pos_enc=False, max_len=2352):
        self.pos_enc = pos_enc
        ft, fh, fw = filter_sizes

        super().__init__()
        assert len(filter_sizes) == 3

        self.encoder = nn.Conv3d(3, d_emb, kernel_size=(ft, fh, fw), stride=(ft, fh, fw))

    def forward(self, x):
        """
        x: shape = [b, c, t, h, w]

        Returns tuple with x, with new shape = [b, seq_len, c_out]
        """
        x = self.encoder(x)
        b, c, t, h, w = x.shape

        x = x.reshape([b, c, t*h*w])  # flatten spatial / temporal dim
        x = x.permute(0, 2, 1)  # permute the c and seq len for s4

        return x

class Conv2DPatchEncoder(Encoder):
    """For encoding images into a sequence of patches.

    Arguments:
      - d_input: dim of encoder input (data dimension)
      - d_model: dim of encoder output (model dimension)
      - filter_sizes: tuple with fh, fw
      - flat: if image is flattened from dataloader (like in cifar),
        then we need to reshape back to 2D before conv
    """

    def __init__(self, d_input, d_model, filter_sizes, flat=False):
        fh, fw = filter_sizes
        self.flat = flat

        super().__init__()
        assert len(filter_sizes) == 2

        self.encoder = nn.Conv2d(d_input, d_model, kernel_size=(fh, fw), stride=(fh, fw))

    def forward(self, x):
        """
        x shape = [b, h, w, c]
        Returns tuple with x, with new shape = [b, seq_len, c_out]
        """

        x = rearrange(x, 'b h w c -> b c h w')
        x = self.encoder(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x

class TextConditionalEncoder(Encoder):

    def __init__(self, vocab_size, d_model, n_layers, layer, reversal=False):
        super().__init__()
        # d_model = 2 * d_model
        self.reversal = reversal
        self.padding_idx = vocab_size - 1
        self.text_embedding = nn.Embedding(vocab_size, d_model)

        # Simple stack of blocks
        self.text_encoder = nn.ModuleList([
            SequenceResidualBlock(
                d_input=d_model,
                i_layer=i,
                prenorm=True,
                layer=layer,
                residual='R',
                norm='layer',
                pool=None,
                transposed=True,
            ) for i in range(n_layers)
        ])

        # self.output_linear = nn.Linear(d_model, d_model // 2)

        # self.norm = Normalization(d_model, transposed=True, _name_='layer')

    def forward(self, x, tokens=None, text_lengths=None):
        # Arguments must be in this order
        # lengths, tokens, text_lengths = args
        assert tokens is not None and text_lengths is not None


        # Calculate the text embedding
        text_embedding = self.text_embedding(tokens) # (B, L, D)
        text_embedding = text_embedding.transpose(1, 2) # (B, D, L)
        for layer in self.text_encoder:
            text_embedding, _ = layer(text_embedding)

            if self.reversal:
                # Reverse the sequence
                text_embedding = text_embedding.fliplr()
        # text_embedding = self.norm(text_embedding)
        text_embedding = text_embedding.transpose(1, 2)

        # Zero out the embedding for padding tokens
        mask = (tokens != self.padding_idx).unsqueeze(2)
        text_embedding = text_embedding * mask.float()

        # Calculate the mean embedding for each sequence (normalizing by appropriate token lengths)
        text_embedding = text_embedding.sum(dim=1) / text_lengths.float().unsqueeze(1)
        # text_embedding = self.output_linear(text_embedding)

        # Add the text embedding to the sequence embedding (for global conditioning)
        x = x + text_embedding.unsqueeze(1)

        return x



# For every type of encoder/decoder, specify:
# - constructor class
# - list of attributes to grab from dataset
# - list of attributes to grab from model

registry = {
    "stop": Encoder,
    "id": nn.Identity,
    "embedding": nn.Embedding,
    "linear": nn.Linear,
    "lazy_linear": nn.LazyLinear,
    "position": PositionalEncoder,
    "class": ClassEmbedding,
    "pack": PackedEncoder,
    "time": TimeEncoder,
    "onehot": OneHotEncoder,
    "conv1d": Conv1DEncoder,
    "eegage": EEGAgeEncoder,
    "patch3d": Conv3DPatchEncoder,
    "patch2d": Conv2DPatchEncoder,
    "textcond": TextConditionalEncoder,
    "timestamp_embedding": TimestampEmbeddingEncoder,
    "tsindex_embedding": TSIndexEmbeddingEncoder,
    "layer": LayerEncoder,
    "cfbag": CFBagEncoder,
    "cfid": CFIDEncoder,
    "cfstate": CFStateEncoder,
    "scid": SCIDEncoder,
    "flowid": FlowIDEncoder,
    "flowid3d": FlowIDEncoder,
    "tradeid": TradeIDEncoder,
    "planb_onehot": PlanBOneHotEncoder,
    "planA_linear": nn.Linear,
}
dataset_attrs = {
    "embedding": ["n_tokens"],
    "textcond": ["vocab_size"],
    "linear": ["d_input"],  # TODO make this d_data?
    "planA_linear": [],
    "class": ["n_classes"],
    "time": ["n_tokens_time"],
    "onehot": ["n_tokens"],
    "conv1d": ["d_input"],
    "patch2d": ["d_input"],
    "tsindex_embedding": ["n_ts"],
    "cfbag": ["n_comms"],
    "cfid": ["n_comms"],
    "cfstate": ["n_comms"],
    "scid": ["n_states", "n_comms"],
    "flowid": [],
    "tradeid": ["n_states", "n_comms", "n_flows"],
    "planb_onehot": ["n_states", "n_comms", "n_flows"],
}
model_attrs = {
    "embedding": ["d_model"],
    "textcond": ["d_model"],
    "linear": ["d_model"],
    "lazy_linear": ["d_model"],
    "position": ["d_model"],
    "class": ["d_model"],
    "time": ["d_model"],
    "onehot": ["d_model"],
    "conv1d": ["d_model"],
    "patch2d": ["d_model"],
    "eegage": ["d_model"],
    "timestamp_embedding": ["d_model"],
    "tsindex_embedding": ["d_model"],
    "layer": ["d_model"],
}


def _instantiate(encoder, dataset=None, model=None):
    """Instantiate a single encoder"""
    if encoder is None:
        return None
    if isinstance(encoder, str):
        name = encoder
    else:
        name = encoder["_name_"]

    # Extract dataset/model arguments from attribute names
    dataset_args = utils.config.extract_attrs_from_obj(
        dataset, *dataset_attrs.get(name, [])
    )
    model_args = utils.config.extract_attrs_from_obj(model, *model_attrs.get(name, []))

    # Instantiate encoder
    obj = utils.instantiate(registry, encoder, *dataset_args, *model_args)
    return obj


def instantiate(encoder, dataset=None, model=None):
    encoder = utils.to_list(encoder)
    return U.PassthroughSequential(
        *[_instantiate(e, dataset=dataset, model=model) for e in encoder]
    )
