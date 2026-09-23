"""Instantiate the model target declared in a typed recipe."""
import importlib

from configs import ModelConfig


def get_obj_from_str(string: str):
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module), cls)


def instantiate_from_config(config: ModelConfig):
    if not config.target:
        raise KeyError("Expected 'target' to instantiate")
    return get_obj_from_str(config.target)(**config.params)
