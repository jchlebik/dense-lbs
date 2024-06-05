"""Default Hyperparameter configuration for generating acoustic mnist dataset."""

import ml_collections
import jax


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()
    
    config.image_size = 128
    config.pml_size = 16
    
    config.sos_range = (1.0, 3.0)
    config.rho_range = (1.0, 2.0)
    config.labels_filter = (1,)
    #config.sos_range = (15.0, 35.0)
    #config.rho_range = (10.0, 20.0)
    
    #config.sos_range = (1524.0, 3515.0)
    #config.rho_range = ( 993.3, 1908.0)
    # config.max_sos = 2.0
    # config.max_rho = 1.9
    
    config.omega = 1.0
    config.amp = 10.0
    config.src_pos = (32, 32)

   
    config.num_samples = 6000
    config.target=jax.numpy.complex64
    #config.target=jax.numpy.float32
    return config



def metrics():
    return []