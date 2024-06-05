import flax.linen as nn

from jax import numpy as jnp
from jwave import FourierSeries
from jwave.acoustics.time_harmonic import born_series
from jwave.geometry import Domain, Medium

from utils.key_mapper import KeyMapper as km


class CBS(nn.Module):
    stages: int = 12

    def setup(self):
        keys_mapping = km.get_dataset_to_model_mapping()
        
        self.sos_key = keys_mapping[km.get_sound_speed_key()]
        self.pml_key = keys_mapping[km.get_pml_key()]
        self.source_key = keys_mapping[km.get_source_key()]
        self.keys = [self.sos_key, self.pml_key, self.source_key]
            
    def get_keys(self):
        return self.keys   

    @nn.compact
    def __call__(self, **kwargs) -> jnp.ndarray:
        
        sos = kwargs[self.sos_key]
        pml = kwargs[self.pml_key]
        src = kwargs[self.source_key]
    
        # Strip batch dimension and channel dimension
        sos = sos[0, ..., 0]
        src = src[0, ..., 0]

        # Setup domain
        image_size = sos.shape[1]
        N = tuple([image_size] * 2)
        dx = (1.0, 1.0)
        domain = Domain(N, dx)

        # Define fields
        sound_speed = FourierSeries(sos, domain)
        src = FourierSeries(src, domain)

        # Make model
        medium = Medium(domain, sound_speed)

        # Predict
        predicted = born_series(
            medium, src, max_iter=self.stages, k0=0.79056941504209483299972338610818
        ).on_grid

        # Expand dims
        predicted = jnp.expand_dims(predicted, 0)

        return predicted