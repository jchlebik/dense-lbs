from typing import Callable

import flax.linen as nn
from jax import numpy as jnp

from jaxdf import FourierSeries
from jaxdf.geometry import Domain

from utils.key_mapper import KeyMapper as km

from .utils import get_grid, constant

class TunableGreens(nn.Module):
    out_channels: int = 32

    @nn.compact
    def __call__(self, x):
        # Initialize parameters
        in_channels = x.shape[-1]

        k0 = self.param(
            "k0",
            constant(1.0, jnp.float32),
            (in_channels, self.out_channels, 1, 1),
            jnp.float32,
        )
        amplitude = self.param(
            "amplitude",
            constant(1.0, jnp.float32),
            (in_channels, self.out_channels, 1, 1),
            jnp.float32,
        )

        # Keep them positive
        k0 = nn.softplus(k0)
        amplitude = nn.softplus(amplitude)

        # Get frequency axis squared
        _params = (
            jnp.zeros(
                list(x.shape[1:3])
                + [
                    1,
                ]
            )
            + 0j
        )
        field = FourierSeries(_params, Domain(x.shape[1:3], (1, 1)))
        freq_grid = field._freq_grid
        p_sq = amplitude * jnp.sum(freq_grid**2, -1)

        # Apply mixing Green's function
        g_fourier = 1.0 / (p_sq - k0 - 1j)  # [ch, ch, h, w]
        u_fft = jnp.fft.fftn(x, axes=(1, 2))
        u_filtered = jnp.einsum("bijc,coij->bijo", u_fft, g_fourier)

        Gu = jnp.fft.ifftn(u_filtered, axes=(1, 2)).real

        return Gu

class Project(nn.Module):
    in_channels: int = 32
    out_channels: int = 32
    activation: Callable = lambda x: jnp.exp(-(x**2))  # nn.gelu

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.in_channels)(x)
        x = self.activation(x)
        x = nn.Dense(self.in_channels)(x)
        x = self.activation(x)
        x = nn.Dense(self.out_channels)(x)
        return x


def D(x, W):
    q = jnp.einsum("bijc,bijco->bijo", x, W)
    return q


class FourierStage(nn.Module):
    out_channels: int = 32
    activation: Callable = nn.gelu
    use_nonlinearity: bool = True

    def gamma_reshaper(self, tensor_input):
        return jnp.reshape(
            tensor_input,
            (
                tensor_input.shape[0],
                tensor_input.shape[1],
                tensor_input.shape[2],
                self.out_channels,
                self.out_channels,
            )
        )

    @nn.compact
    def __call__(self, x, sos_context, density_context):
        gamma1 = Project(self.out_channels, self.out_channels**2)(sos_context)
        gamma2 = Project(self.out_channels, self.out_channels**2)(sos_context)
        gamma3 = Project(self.out_channels, self.out_channels**2)(jnp.concatenate([sos_context, density_context], axis=-1))
        gamma4 = Project(self.out_channels, self.out_channels**2)(density_context)
        gamma5 = Project(self.out_channels, self.out_channels**2)(density_context)
        
        G = TunableGreens(self.out_channels)

        # G = SpectralConv2d(
        #  out_channels=self.out_channels,
        #  modes1=self.modes1,
        #  modes2=self.modes2
        # )

        gamma1 = self.gamma_reshaper(gamma1)
        gamma2 = self.gamma_reshaper(gamma2)
        gamma3 = self.gamma_reshaper(gamma3)
        gamma4 = self.gamma_reshaper(gamma4)
        gamma5 = self.gamma_reshaper(gamma5)


        a = D(x, gamma2)
        b = D(x, gamma4)
        
        kern = G( jnp.concatenate([a, b], axis=-1) )
        
        c = G(kern)
        
        precond = D(c, gamma1)
        x_fourier = D(precond, gamma5)
        
        #x_fourier = D(     D(     G(   D(a, gamma4)   )    , gamma1) , gamma5)
        
        #x_fourier = D( G( D(x, gamma2) ), gamma1 )
        x_local = D(x, gamma3)
        out = x_fourier + x_local

        # Apply nonlinearity
        if self.use_nonlinearity:
            out = nn.Dense(out.shape[-1])(out)
            out = self.activation(out)

        out = nn.Dense(out.shape[-1])(out)
        return out

class DBNO_FStage_Ext(nn.Module):
    r"""
    Fourier Neural Operator for 2D signals.

    Implemented from
    https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py

    Attributes:
      modes1: Number of modes in the first dimension.
      modes2: Number of modes in the second dimension.
      channels: Number of channels to which the input is lifted.
      stages: Number of Fourier stages
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
        src = kwargs[self.source_key]
        sos = kwargs[self.sos_key]
        density = kwargs[self.density_key]
        pmls = kwargs[self.pml_key]
        
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
            sos_context = jnp.concatenate([sos, grid], axis=-1)
            density_context = jnp.concatenate([density, grid], axis=-1)
        else:
            sos_context = sos
            density_context = density

        # Lift the input to a higher dimension
        x = nn.Dense(self.channels)(src) * 0.0
        #x = nn.Dense(32)(src) * 0.0
        x_new = nn.Dense(self.channels)(src)

        # Apply Fourier stages, last one has no activation
        # (can't find this in the paper, but is in the original code)
        for depthnum in range(self.stages):
            activation = self.activation if depthnum < self.stages - 1 else lambda x: x
            x_new = nn.remat(FourierStage)(
                out_channels=self.channels,
                activation=activation,
                use_nonlinearity=self.use_nonlinearity,
            )(x_new, sos_context, density_context)
            x = x_new + x

        # Unpad
        if self.padding > 0:
            x = x[:, : -self.padding, : -self.padding, :]

        # Project to the output channels
        x = nn.Dense(self.last_proj)(x)
        x = self.activation(x)
        x = nn.Dense(self.out_channels)(x)

        return jnp.expand_dims(x[..., 0] + 1j * x[..., 1], -1)