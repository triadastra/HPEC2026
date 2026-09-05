"""
Base model class and factory for time series forecasting models.

All models must inherit from BaseModel and implement the required methods.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np

from .encodings import EncodingFactory, EncodingStrategy


# Grid/axial variant names that share the combo encoder spec. Without this the
# encoder factory raises on the unknown name and every run of that family dies
# at construction. Current names plus the legacy spellings they replaced:
# "cafa_*" borrowed the authors' MODEL name (CaFA) for a local reimplementation
# of their FA operator, and "cross_attention_*" described self-attention.
# ONE source of truth. This list used to be transcribed in three places --
# here, scripts/train.py's --variant choices, and its _axial combo test -- and
# adding fa_sm to only this one left the arm rejected by argparse at launch and
# routed down the FLAT path if it ever got past. Two of those three copies are
# load-bearing for correctness, and nothing compared them. Import the constant.
COMBO_GRID_STEMS = ("asa", "aca", "fa_local", "fa", "fa_sm", "grid",
                    "cross_attention", "cafa", "authors_cafa")
COMBO_GRID_VARIANTS = tuple(
    f"{stem}_{d}d" for d in (2, 3, 4) for stem in COMBO_GRID_STEMS
)
_COMBO_GRID_VARIANTS = frozenset(COMBO_GRID_VARIANTS)



class _NullEncoding(EncodingStrategy):
    """Identity encoder for combo/grid variants.

    Those models take the raw (B, L, G, F) window and replace ``self.encoder``
    with ``nn.Identity()`` in ``_build_model``; this exists so nothing is built
    only to be discarded. Parameter-free, so it cannot affect a cost figure.
    """

    def encode(self, x_numeric, state_ids, comm_ids, flow_ids):
        return x_numeric

    def output_dim(self, numeric_input_dim: int) -> int:
        return numeric_input_dim


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all forecasting models.

    All models must inherit from this class and implement:
    - forward(): Forward pass
    - predict(): Generate predictions
    - Optional: configure() for model-specific configuration
    """

    def __init__(
        self,
        variant: str,
        num_numeric_features: int,
        num_states: int,
        num_commodities: int,
        num_flows: int = 2,
        **kwargs
    ):
        """
        Initialize base model.

        Args:
            variant: Encoding variant name
                Options: "onehot", "embeddings", "cross_attention", "film_attention"
            num_numeric_features: Number of numeric input features
            num_states: Number of unique states
            num_commodities: Number of unique commodities
            num_flows: Number of flow directions (default: 2)
            **kwargs: Additional model-specific parameters
        """
        super().__init__()

        self.variant = variant
        self.num_numeric_features = num_numeric_features
        self.num_states = num_states
        self.num_commodities = num_commodities
        self.num_flows = num_flows

        # Create encoding strategy
        self.encoder = self._create_encoder(variant, **kwargs)

        # Calculate input dimension after encoding
        self.input_dim = self.encoder.output_dim(num_numeric_features)

        # Model-specific initialization
        self._build_model(**kwargs)

    @abstractmethod
    def _build_model(self, **kwargs) -> None:
        """
        Build model architecture.

        Called by __init__ to construct model-specific layers.
        Subclasses must implement this to define their architecture.

        Args:
            **kwargs: Model-specific parameters
        """
        pass

    @abstractmethod
    def forward(
        self,
        x_numeric: torch.Tensor,
        state_ids: torch.Tensor,
        comm_ids: torch.Tensor,
        flow_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x_numeric: Numeric time series features
                Shape: (batch_size, seq_len, num_numeric_features)
            state_ids: State indices
                Shape: (batch_size,)
            comm_ids: Commodity indices
                Shape: (batch_size,)
            flow_ids: Flow direction indices
                Shape: (batch_size,)

        Returns:
            Predictions
                Shape: (batch_size, output_dim)
        """
        pass

    def predict(
        self,
        x_numeric: torch.Tensor,
        state_ids: torch.Tensor,
        comm_ids: torch.Tensor,
        flow_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate predictions (alias for forward).

        This method can be overridden for more complex prediction logic
        (e.g., autoregressive decoding).

        Args:
            x_numeric: Numeric time series features
                Shape: (batch_size, seq_len, num_numeric_features)
            state_ids: State indices
                Shape: (batch_size,)
            comm_ids: Commodity indices
                Shape: (batch_size,)
            flow_ids: Flow direction indices
                Shape: (batch_size,)

        Returns:
            Predictions
                Shape: (batch_size, output_dim)
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x_numeric, state_ids, comm_ids, flow_ids)
        return predictions

    def _create_encoder(self, variant: str, **kwargs) -> EncodingStrategy:
        """
        Create encoding strategy based on variant.

        Args:
            variant: Encoding variant name
            **kwargs: Additional encoder parameters

        Returns:
            EncodingStrategy instance
        """
        # Combo/grid variants consume the raw (B, L, G, F) window and set
        # self.encoder = nn.Identity() in _build_model. They used to be remapped
        # onto the "cross_attention" encoder spec purely so EncodingFactory
        # would accept the name -- which built a full categorical encoder,
        # sized by num_commodities, on EVERY combo run and threw it away. Skip
        # the construction instead.
        if variant in _COMBO_GRID_VARIANTS:
            return _NullEncoding()

        # Extract encoder parameters
        encoder_kwargs = {
            "num_states": self.num_states,
            "num_commodities": self.num_commodities,
            "num_flows": self.num_flows,
        }

        # Add variant-specific parameters
        if variant == "onehot":
            pass  # No additional params needed
        elif variant == "embeddings":
            encoder_kwargs.update({
                "state_embed_dim": kwargs.get("state_embed_dim", 8),
                "comm_embed_dim": kwargs.get("comm_embed_dim", 32),
                "flow_embed_dim": kwargs.get("flow_embed_dim", 2),
            })
        return EncodingFactory.create(variant, **encoder_kwargs)

    def encode_features(
        self,
        x_numeric: torch.Tensor,
        state_ids: torch.Tensor,
        comm_ids: torch.Tensor,
        flow_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode features using the configured encoding strategy.

        This is a convenience method that can be used by subclasses
        to encode features before processing.

        Args:
            x_numeric: Numeric time series features
                Shape: (batch_size, seq_len, num_numeric_features)
            state_ids: State indices
                Shape: (batch_size,)
            comm_ids: Commodity indices
                Shape: (batch_size,)
            flow_ids: Flow direction indices
                Shape: (batch_size,)

        Returns:
            Encoded features
                Shape: (batch_size, seq_len, encoded_dim)
        """
        return self.encoder.encode(x_numeric, state_ids, comm_ids, flow_ids)

    def get_num_params(self) -> int:
        """
        Get total number of trainable parameters.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "variant": self.variant,
            "num_numeric_features": self.num_numeric_features,
            "num_states": self.num_states,
            "num_commodities": self.num_commodities,
            "num_flows": self.num_flows,
        }
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location="cpu")
        self.load_state_dict(checkpoint["model_state_dict"])


class ModelFactory:
    """
    Factory for creating model instances.

    Provides a unified interface for creating models by name and variant.
    """

    _models = {}

    @classmethod
    def register(cls, name: str, model_class: type) -> None:
        """
        Register a model class.

        Args:
            name: Model name (e.g., "lstm", "gru", "transformer")
            model_class: Model class (must inherit from BaseModel)
        """
        cls._models[name] = model_class

    @classmethod
    def create(
        cls,
        name: str,
        variant: str,
        num_numeric_features: int,
        num_states: int,
        num_commodities: int,
        num_flows: int = 2,
        **kwargs
    ) -> BaseModel:
        """
        Create a model instance.

        Args:
            name: Model name
            variant: Encoding variant
            num_numeric_features: Number of numeric input features
            num_states: Number of unique states
            num_commodities: Number of unique commodities
            num_flows: Number of flow directions
            **kwargs: Additional model-specific parameters

        Returns:
            BaseModel instance

        Raises:
            ValueError: If model name is unknown
        """
        if name not in cls._models:
            raise ValueError(
                f"Unknown model: {name}. "
                f"Available models: {list(cls._models.keys())}"
            )

        model_class = cls._models[name]

        return model_class(
            variant=variant,
            num_numeric_features=num_numeric_features,
            num_states=num_states,
            num_commodities=num_commodities,
            num_flows=num_flows,
            **kwargs
        )

    @classmethod
    def list_models(cls) -> list:
        """List all registered models."""
        return list(cls._models.keys())


# Convenience function for creating models
def create_model(
    name: str,
    variant: str,
    num_numeric_features: int,
    num_states: int,
    num_commodities: int,
    num_flows: int = 2,
    **kwargs
) -> BaseModel:
    """
    Create a model instance.

    This is a convenience wrapper around ModelFactory.create().

    Args:
        name: Model name (e.g., "lstm", "gru", "transformer")
        variant: Encoding variant
        num_numeric_features: Number of numeric input features
        num_states: Number of unique states
        num_commodities: Number of unique commodities
        num_flows: Number of flow directions
        **kwargs: Additional model-specific parameters

    Returns:
        BaseModel instance
    """
    return ModelFactory.create(
        name=name,
        variant=variant,
        num_numeric_features=num_numeric_features,
        num_states=num_states,
        num_commodities=num_commodities,
        num_flows=num_flows,
        **kwargs
    )
