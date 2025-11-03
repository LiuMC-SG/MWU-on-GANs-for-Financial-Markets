import os
import json

from common.backtest_common import BacktestConfig

def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_config(config_file: str) -> BacktestConfig:
    """Load configuration from JSON file"""
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
    return BacktestConfig(**config_dict)


def save_config(config: BacktestConfig, config_file: str):
    """Save configuration to JSON file"""
    from dataclasses import asdict
    with open(config_file, 'w') as f:
        json.dump(asdict(config), f, indent=2)
