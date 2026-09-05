"""
Positional encoding for the per-series transformer.

The generic MultiHeadAttention / CrossAttentionModule / FiLMModule blocks that
used to live here were deleted: transformer.py uses nn.TransformerEncoderLayer,
and the cross_attention / film_attention variants they served are gone.

Only PositionalEncoding remains; see the note above.
"""

import torch
import torch.nn as nn
import math
class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer models.

    Adds position information to input embeddings using sinusoidal functions.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout rate (kept for compatibility, not used)
        """
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        # Register as buffer (not a parameter, but part of state)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor
                Shape: (batch_size, seq_len, d_model)

        Returns:
            Output tensor with positional encoding added
                Shape: (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, : x.size(1), :]
