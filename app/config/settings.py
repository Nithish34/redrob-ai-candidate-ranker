"""
Configuration loader for the ranking system.
Loads weights and settings from weights.yaml.
"""

import os
import yaml
from typing import Any


_CONFIG_CACHE: dict | None = None


def load_config(config_path: str | None = None) -> dict:
    """Load configuration from YAML file. Results are cached."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "weights.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        _CONFIG_CACHE = yaml.safe_load(f)

    return _CONFIG_CACHE


def get_layer_config(layer_name: str) -> dict:
    """Get configuration for a specific layer."""
    config = load_config()
    return config.get(layer_name, {})


def get_ranking_weights() -> dict:
    """Get the final ranking layer weights."""
    return load_config().get("ranking", {})


def reset_config():
    """Reset the config cache (useful for testing)."""
    global _CONFIG_CACHE
    _CONFIG_CACHE = None
