from typing import Callable

import flax.linen as nn
from jax import numpy as jnp

from utils.key_mapper import KeyMapper as km

from .bno import FourierStage
from .utils import get_grid

class DBNO(nn.Module):
    r"""
    Fourier Neural Operator for 2D signals.

    Implemented from
    https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py

    Attributes:
      modes1: Number of modes in the first dimension.
      modes2: Number of modes in the second dimension.
      width: Number of channels to which the input is lifted.
      depth: Number of Fourier stages
      channels_last_proj: Number of channels in the hidden layer of the last
        2-layers Fully Connected (channel-wise) network
      activation: Activation function to use
      out_channels: Number of output channels, >1 for non-scalar fields.
    """
    stages: int = 4         #depth: int = 4
    channels: int = 8       #width: int = 32
    dtype: jnp.dtype = jnp.complex64
    last_proj: int = 128    #channels_last_proj: int = 128
    use_nonlinearity: bool = True
    use_grid: bool = True
    padding: int = 32  # Padding for non-periodic inputs
    activation: Callable = nn.gelu 
    
    def setup(self):
        keys_mapping = km.get_dataset_to_model_mapping()
        
        self.sos_key = keys_mapping[km.get_sound_speed_key()]
        self.density_key = keys_mapping[km.get_density_key()]
        self.pml_key = keys_mapping[km.get_pml_key()]
        self.source_key = keys_mapping[km.get_source_key()]
        self.keys = [self.sos_key, self.density_key, self.pml_key, self.source_key]
        
        if self.dtype == jnp.complex64:
            self.out_channels = 2
        else:
            self.out_channels = 2
    
    def get_keys(self):
        return self.keys        
    
    @nn.compact
    def __call__(self, **kwargs) -> jnp.ndarray:        
        sos = kwargs[self.sos_key]
        density = kwargs[self.density_key]
        pmls = kwargs[self.pml_key]
        src = kwargs[self.source_key]

        # Pad input
        if self.padding > 0:
            src = jnp.pad(
                src,
                ((0, 0), (0, self.padding), (0, self.padding), (0, 0)),
                mode="constant",
            )
            sos = jnp.pad(
                sos, ((0, 0), (0, self.padding), (0, self.padding), (0, 0)), mode="edge"
            )
            density = jnp.pad(
                density, ((0, 0), (0, self.padding), (0, self.padding), (0, 0)), mode="edge"
            )

        # Generate coordinate grid, and append to input channels
        if self.use_grid:
            grid = get_grid(src)
            context = jnp.concatenate([sos, density, grid], axis=-1)
        else:
            context = jnp.concatenate([sos, density], axis=-1)
        
        # Lift the input to a higher dimension
        x = nn.Dense(self.channels)(src) * 0.0
        x_new = nn.Dense(self.channels)(src)

        # Apply Fourier stages, last one has no activation
        # (can't find this in the paper, but is in the original code)
        for depthnum in range(self.stages):
            activation = self.activation if depthnum < self.stages - 1 else lambda x: x
            x_new = nn.remat(FourierStage)(
                out_channels=self.channels,
                activation=activation,
                use_nonlinearity=self.use_nonlinearity,
            )(x_new, context)
            x = x_new + x

        # Unpad
        if self.padding > 0:
            x = x[:, : -self.padding, : -self.padding, :]

        # Project to the output channels
        x = nn.Dense(self.last_proj)(x)
        x = self.activation(x)
        x = nn.Dense(self.out_channels)(x)

        return jnp.expand_dims(x[..., 0] + 1j * x[..., 1], -1)