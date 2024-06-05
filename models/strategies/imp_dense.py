from typing import Callable

import flax.linen as nn
from jax import numpy as jnp
from utils.key_mapper import KeyMapper as km

from .utils import get_grid


class Impedance_Predictor(nn.Module):

    stages: int = 4
    channels: int = 8
    dtype: jnp.dtype = jnp.complex64
    last_proj: int = 128
    use_nonlinearity: bool = False
    use_grid: bool = False
    activation: Callable = nn.gelu
    
    out_channels: int = 2
    padding: int = 0

    def setup(self):
        keys_mapping = km.get_dataset_to_model_mapping()
        
        self.sos_key = keys_mapping[km.get_sound_speed_key()]
        self.density_key = keys_mapping[km.get_density_key()]
        self.pml_key = keys_mapping[km.get_pml_key()]
        self.source_key = keys_mapping[km.get_source_key()]
        self.sos_field_key = keys_mapping[km.get_sos_field_key()]
        self.keys = [self.sos_key, self.density_key, self.pml_key, self.source_key, self.sos_field_key]

        
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
        pml = kwargs[self.pml_key]
        src = kwargs[self.source_key]
        sos_field = kwargs[self.sos_field_key]
        training = kwargs.get('training', False)

        # Pad input
        if self.padding > 0:   
            src = jnp.pad(
                src,
                ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)),
                mode="constant",
            )
            sos = jnp.pad(
                sos, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode="edge"
            )
            density = jnp.pad(
                density, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode="edge"
            )

        #self.is_training = training
        # Generate coordinate grid, and append to input channels
        x = jnp.concatenate([src, sos, density], axis=-1)
        
        if self.use_grid:
            grid = get_grid(src)
            #grid = jnp.ones_like(sos)
            x = jnp.concatenate([x, grid], axis=-1)
        
        skips = []

        for i in range(self.stages):
            skip, x = self.downsample_block(x,
                                            channels=self.channels * 2 ** i,
                                            kernel_size=(3,3),
                                            strides=1,
                                            is_training = training)
            skips.append(skip)

        x = self.bottleneck_block(x, 
                                  channels=self.channels * 2 ** self.stages, 
                                  kernel_size=(3,3), 
                                  strides=1, 
                                  is_training = training)

        for i in reversed(range(self.stages)):
            x = self.upsample_block(x, 
                                    skip=skips[i], 
                                    channels=self.channels * 2 ** i,
                                    kernel_size=(3,3),
                                    strides=1,
                                    is_training = training)

        # Final convolution to get the output
        #x = nn.Conv(self.out_channels, kernel_size=(1,1))(x)
        if self.padding > 0:
            x = x[:, self.padding: -self.padding, self.padding: -self.padding, :]
            
        x = nn.Dense(self.last_proj)(x)
        x = self.activation(x)
        x = nn.Dense(self.out_channels)(x)

             

        return jnp.expand_dims(x[..., 0] + 1j * x[..., 1], -1)

    def downsample_block(self, x, channels, kernel_size, strides, padding = 'SAME', is_training=True):
        skip = self.double_conv(x, middle_channels=channels * 2, out_channels=channels, kernel_size=kernel_size, 
                                strides=strides, padding=padding, is_training=is_training)
        #skip = self.activation(skip)
        
        skip = self.activation(skip)
        #skip = nn.PReLU()(skip)
        pool_down = nn.max_pool(skip, window_shape=(2, 2), strides=(2, 2))
        return skip, pool_down
    
    def upsample_block(self, x, skip, channels, kernel_size, strides, padding = 'SAME', is_training=True):
        x = nn.ConvTranspose(channels, kernel_size=kernel_size, strides=(2,2), padding=padding)(x)
        x = jnp.concatenate([x, skip], axis=-1)
        x = nn.InstanceNorm()(x)
        #x = nn.BatchNorm(use_running_average=(not is_training))(x)
        x = self.activation(x)
        x = self.double_conv(x, middle_channels=channels * 2, out_channels=channels, kernel_size=kernel_size, 
                             strides=strides, padding=padding, is_training=is_training)
       
        x = self.activation(x)
        return x
    
    def bottleneck_block(self, x, channels, kernel_size, strides, padding='SAME', is_training=True):
        x = self.double_conv(x, middle_channels=channels * 2, out_channels=channels, kernel_size=kernel_size, 
                             strides=strides, padding=padding, is_training=is_training)
        x = self.activation(x)
        return x
        
    def double_conv(self, x, middle_channels, out_channels, kernel_size=(3, 3), strides=1, padding='SAME', is_training=True):
        #x = nn.Conv(middle_channels, kernel_size = kernel_size, strides = strides, padding = padding)(x)
        x = nn.Dense(middle_channels)(x)
        #x = nn.InstanceNorm()(x)
        #x = nn.BatchNorm(use_running_average=(not is_training))(x)
        x = self.activation(x)
        #x = nn.Conv(out_channels, kernel_size = kernel_size, strides = strides, padding = padding)(x)
        x = nn.Dense(out_channels)(x)
        #x = nn.InstanceNorm()(x)
        #x = nn.BatchNorm(use_running_average=(not is_training))(x)
        return x
