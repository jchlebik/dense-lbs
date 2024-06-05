"""Hyperparameter configuration for 6 stages."""

from config.dlbs import default as default_config

def get_config():
    """Get the training hyperparameter configuration."""

    config = default_config.get_config()
    config.stages = 6
    #config.validate_every_n_epochs  = 1 
    return config

def metrics():
    return []