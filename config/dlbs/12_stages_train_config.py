"""Hyperparameter configuration for 12 stages."""

from config.dlbs import default as default_config

def get_config():
    """Get the training hyperparameter configuration."""

    config = default_config.get_config()
    config.stages = 12
    return config

def metrics():
    return []