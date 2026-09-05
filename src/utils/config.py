"""
Configuration management utilities.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import copy


def load_config(config_path: str, base_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load YAML configuration file and merge with base config.

    Args:
        config_path: Path to YAML config file
        base_config: Optional base config to merge with

    Returns:
        Merged configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Start with empty dict or base config
    if base_config is None:
        merged = {}
    else:
        merged = copy.deepcopy(base_config)

    # Deep merge
    _deep_merge(merged, config)

    return merged


def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> None:
    """
    Deep merge update dict into base dict (modifies base in-place).

    A ``None`` value in ``update`` does NOT overwrite ``base[key]``. This is
    deliberate: variant YAML files often contain comment-only sections (e.g.
    ``data:`` followed only by comments) which YAML parses as ``data: None``.
    Without this guard, those empty sections would wipe out the base config's
    real ``data:`` block during merging.
    """
    for key, value in update.items():
        if value is None:
            # Skip — empty/comment-only YAML sections must not blank out base.
            continue
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        save_path: Path to save YAML file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def override_config(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Override config values with new values (supports nested keys with dot notation).

    Args:
        config: Base configuration
        overrides: Override dictionary (supports dot notation for nested keys)

    Returns:
        Updated configuration

    Example:
        config = {"model": {"hidden_size": 128}}
        overrides = {"model.hidden_size": 256}
        result = {"model": {"hidden_size": 256}}
    """
    result = copy.deepcopy(config)

    for key, value in overrides.items():
        if "." in key:
            # Nested key with dot notation
            parts = key.split(".")
            current = result
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            # Top-level key
            result[key] = value

    return result


def get_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract model configuration from full config.

    Args:
        config: Full configuration dictionary

    Returns:
        Model configuration dictionary
    """
    return config.get("model", {})


def get_data_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract data configuration from full config.

    Args:
        config: Full configuration dictionary

    Returns:
        Data configuration dictionary
    """
    return config.get("data", {})


def get_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract training configuration from full config.

    Args:
        config: Full configuration dictionary

    Returns:
        Training configuration dictionary
    """
    return config.get("training", {})
