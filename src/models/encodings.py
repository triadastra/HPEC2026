"""
Encoding strategies for categorical features in time series forecasting.

Implements 4 variants:
1. OneHotEncoding - Basic one-hot encoding
2. EmbeddingEncoding - Learned dense embeddings
3. CrossAttentionEncoding - Cross-attention for combinations
4. FiLMAttentionEncoding - Cross-attention + FiLM conditioning
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional


class EncodingStrategy(nn.Module, ABC):
    """
    Abstract base class for encoding strategies.

    Inherits ``nn.Module`` so subclasses that hold ``nn.Embedding`` /
    ``nn.Linear`` attributes have those parameters registered and picked up
    by the parent model's ``parameters()`` / ``state_dict()`` / ``.to()``.
    Every concrete subclass MUST call ``super().__init__()`` as the first
    line of its own ``__init__`` — otherwise PyTorch's setattr machinery
    is not initialised and embeddings are silently frozen at random init.

    All encoding strategies implement the ``encode`` method which transforms
    categorical features into a format suitable for the model.
    """

    @abstractmethod
    def encode(
        self,
        x_numeric: torch.Tensor,
        state_ids: torch.Tensor,
        comm_ids: torch.Tensor,
        flow_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode categorical features along with numeric features.

        Args:
            x_numeric: Numeric time series features
                Shape: (batch_size, seq_len, num_numeric_features)
            state_ids: State indices
                Shape: (batch_size,) or (batch_size, seq_len)
            comm_ids: Commodity indices
                Shape: (batch_size,) or (batch_size, seq_len)
            flow_ids: Flow direction indices (0=import, 1=export)
                Shape: (batch_size,) or (batch_size, seq_len)

        Returns:
            Encoded features tensor
                Shape: (batch_size, seq_len, encoded_dim)
        """
        pass

    @abstractmethod
    def output_dim(self, numeric_input_dim: int) -> int:
        """
        Calculate output dimension after encoding.

        Args:
            numeric_input_dim: Number of numeric input features

        Returns:
            Total output dimension
        """
        pass


class OneHotEncoding(EncodingStrategy):
    """
    Variant 1: One-Hot Encoding

    Uses one-hot vectors for categorical features. Simple but increases
    dimensionality significantly.
    """

    def __init__(
        self,
        num_states: int,
        num_commodities: int,
        num_flows: int = 2,
    ):
        """
        Initialize one-hot encoding.

        Args:
            num_states: Number of unique states
            num_commodities: Number of unique commodities
            num_flows: Number of flow directions (default: 2 for import/export)
        """
        super().__init__()
        self.num_states = num_states
        self.num_commodities = num_commodities
        self.num_flows = num_flows

        # Total one-hot dimensions
        self.cat_dim = num_states + num_commodities + num_flows

    def encode(
        self,
        x_numeric: torch.Tensor,
        state_ids: torch.Tensor,
        comm_ids: torch.Tensor,
        flow_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode using one-hot vectors.

        Args:
            x_numeric: (batch_size, seq_len, num_numeric)
            state_ids: (batch_size,)
            comm_ids: (batch_size,)
            flow_ids: (batch_size,)

        Returns:
            Combined features: (batch_size, seq_len, num_numeric + cat_dim)
        """
        batch_size, seq_len, _ = x_numeric.shape

        # Ensure IDs are on same device
        device = x_numeric.device
        state_ids = state_ids.to(device)
        comm_ids = comm_ids.to(device)
        flow_ids = flow_ids.to(device)

        # One-hot encode
        state_onehot = nn.functional.one_hot(state_ids, self.num_states).float()  # (B, num_states)
        comm_onehot = nn.functional.one_hot(comm_ids, self.num_commodities).float()  # (B, num_comm)
        flow_onehot = nn.functional.one_hot(flow_ids, self.num_flows).float()  # (B, num_flows)

        # Concatenate
        cat_features = torch.cat([state_onehot, comm_onehot, flow_onehot], dim=-1)  # (B, cat_dim)

        # Expand to sequence length
        cat_features = cat_features.unsqueeze(1).expand(-1, seq_len, -1)  # (B, seq_len, cat_dim)

        # Combine with numeric features
        combined = torch.cat([x_numeric, cat_features], dim=-1)  # (B, seq_len, num_numeric + cat_dim)

        return combined

    def output_dim(self, numeric_input_dim: int) -> int:
        """Calculate output dimension."""
        return numeric_input_dim + self.cat_dim


class EmbeddingEncoding(EncodingStrategy):
    """
    Variant 2: Learned Dense Embeddings

    Uses learned embeddings for categorical features. More efficient
    than one-hot for high cardinality features.
    """

    def __init__(
        self,
        num_states: int,
        num_commodities: int,
        num_flows: int = 2,
        state_embed_dim: int = 8,
        comm_embed_dim: int = 32,
        flow_embed_dim: int = 2,
    ):
        """
        Initialize embedding encoding.

        Args:
            num_states: Number of unique states
            num_commodities: Number of unique commodities
            num_flows: Number of flow directions
            state_embed_dim: State embedding dimension
            comm_embed_dim: Commodity embedding dimension
            flow_embed_dim: Flow embedding dimension
        """
        super().__init__()
        self.num_states = num_states
        self.num_commodities = num_commodities
        self.num_flows = num_flows

        # Create embedding layers
        self.state_embed = nn.Embedding(num_states, state_embed_dim)
        self.comm_embed = nn.Embedding(num_commodities, comm_embed_dim)
        self.flow_embed = nn.Embedding(num_flows, flow_embed_dim)

        # Repo-wide embedding standard: dims (state=8, comm=32, flow=2), normal(0.02)
        nn.init.normal_(self.state_embed.weight, std=0.02)
        nn.init.normal_(self.comm_embed.weight, std=0.02)
        nn.init.normal_(self.flow_embed.weight, std=0.02)

        # Total embedding dimension
        self.embed_dim = state_embed_dim + comm_embed_dim + flow_embed_dim

    def encode(
        self,
        x_numeric: torch.Tensor,
        state_ids: torch.Tensor,
        comm_ids: torch.Tensor,
        flow_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode using learned embeddings.

        Args:
            x_numeric: (batch_size, seq_len, num_numeric)
            state_ids: (batch_size,)
            comm_ids: (batch_size,)
            flow_ids: (batch_size,)

        Returns:
            Combined features: (batch_size, seq_len, num_numeric + embed_dim)
        """
        batch_size, seq_len, _ = x_numeric.shape

        # Ensure IDs are on same device
        device = x_numeric.device
        state_ids = state_ids.to(device)
        comm_ids = comm_ids.to(device)
        flow_ids = flow_ids.to(device)

        # Get embeddings
        state_emb = self.state_embed(state_ids)  # (B, state_embed_dim)
        comm_emb = self.comm_embed(comm_ids)  # (B, comm_embed_dim)
        flow_emb = self.flow_embed(flow_ids)  # (B, flow_embed_dim)

        # Concatenate embeddings
        cat_features = torch.cat([state_emb, comm_emb, flow_emb], dim=-1)  # (B, embed_dim)

        # Expand to sequence length
        cat_features = cat_features.unsqueeze(1).expand(-1, seq_len, -1)  # (B, seq_len, embed_dim)

        # Combine with numeric features
        combined = torch.cat([x_numeric, cat_features], dim=-1)  # (B, seq_len, num_numeric + embed_dim)

        return combined

    def output_dim(self, numeric_input_dim: int) -> int:
        """Calculate output dimension."""
        return numeric_input_dim + self.embed_dim
class EncodingFactory:
    """Factory for creating encoding strategies."""

    @staticmethod
    def create(variant: str, **kwargs) -> EncodingStrategy:
        """
        Create an encoding strategy based on variant name.

        Args:
            variant: One of "onehot", "embeddings"
            **kwargs: Additional arguments passed to encoding strategy

        Returns:
            EncodingStrategy instance

        Raises:
            ValueError: If variant is unknown
        """
        variants = {
            "onehot": OneHotEncoding,
            "embeddings": EmbeddingEncoding,
        }

        if variant not in variants:
            raise ValueError(
                f"Unknown encoding variant: {variant}. "
                f"Must be one of {list(variants.keys())}"
            )

        return variants[variant](**kwargs)
