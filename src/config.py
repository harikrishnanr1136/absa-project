"""
Configuration loader for ABSA Telecom project.
Reads config.yaml and provides a dict-based configuration interface.
"""

import os
import logging
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Default config path (project root)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.yaml")


def load_config(config_path: str = None) -> dict:
    """
    Load project configuration from a YAML file.

    Args:
        config_path: Path to config.yaml. Defaults to project root config.yaml.

    Returns:
        Dictionary containing all configuration values.

    Raises:
        FileNotFoundError: If config file does not exist.
        yaml.YAMLError: If config file has invalid YAML syntax.
    """
    path = config_path or _DEFAULT_CONFIG_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Configuration loaded from: {path}")
    return config


def get_project_root() -> str:
    """Return the absolute path to the project root directory."""
    return _PROJECT_ROOT


def resolve_path(relative_path: str) -> str:
    """
    Resolve a relative path from config.yaml to an absolute path.

    Args:
        relative_path: Path relative to project root (e.g., "data/cleaned.csv")

    Returns:
        Absolute path string.
    """
    return os.path.join(_PROJECT_ROOT, relative_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    print("=" * 60)
    print("CONFIGURATION LOADED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nProject root: {_PROJECT_ROOT}")
    print(f"\nData paths:")
    for key, val in config["data"].items():
        print(f"  {key}: {resolve_path(val)}")
    print(f"\nLabels:")
    print(f"  Aspects ({len(config['labels']['aspects'])}): {config['labels']['aspects']}")
    print(f"  Sentiments: {config['labels']['sentiments']}")
    print(f"\nSplit ratios: {config['split']}")
    print(f"Random seed: {config['seed']}")
    print("=" * 60)
