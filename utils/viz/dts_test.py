import sys
sys.path.append('/home/xchleb07/dev/dlbs')

from jwave import FourierSeries
from jwave.acoustics.time_harmonic import helmholtz, helmholtz_solver
from jwave.geometry import Domain, Medium

import tqdm

import jax
from jax import numpy as jnp
from matplotlib import pyplot as plt

from dataset_generator.backend.mnist_downloader import MNISTDownloader

image_size = 128
pml_size = 16
sound_speed_lims = (1.500, 2.800)
density_lims = (1.000, 1.850)
labels_filter = (1, 7)
omega = 1 # * 2 * jnp.pi
amplitude = 10.0
dx = (1, 1)
src_pos = (32, 32)
num_samples = 6000
dtype = jnp.float32

sos_fields = []
density_fields = []
full_fields  = []
sound_speeds  = []
densities  = []
sources  = []
pml = []
out_path = "."

mnist_images = MNISTDownloader("tensorflow").get(num_samples, labels_filter)
N = tuple([image_size + 2 * pml_size] * 2)
domain = Domain(N, dx)

for i, image in enumerate(tqdm.tqdm(mnist_images)):
    im = jax.image.resize(image, (image_size, image_size), "nearest")
    im = jnp.pad(im, pml_size, mode="edge")
    im = jnp.expand_dims(im, -1)

    # Fixing range
    sos = im * (sound_speed_lims[1] - sound_speed_lims[0]) + sound_speed_lims[0]
    rho = im * (density_lims[1] - density_lims[0]) + density_lims[0]
        
    speed_of_sound = FourierSeries(sos, domain)
    density = FourierSeries(rho, domain)
    
    full_medium = Medium(domain=domain, sound_speed=speed_of_sound, density=density, pml_size=pml_size)
    
    src_field = jnp.zeros(N).astype(jnp.complex64)
    src_field = src_field.at[src_pos[0], src_pos[1]].set(amplitude)
    src = FourierSeries(jnp.expand_dims(src_field, -1), domain) * omega
        
    full_field = helmholtz_solver(full_medium, omega, -src)
    
    plt.imshow(jnp.real(full_field.on_grid), cmap="inferno" )
    plt.colorbar()
    plt.savefig("full_field.png")
    plt.close()
    
    plt.imshow(sos, cmap="inferno" )
    plt.colorbar()
    plt.savefig("sos.png")
    plt.close()
    
    plt.imshow(rho, cmap="inferno" )
    plt.colorbar()
    plt.savefig("rho.png")
    plt.close()
    
    full_fields.append(full_field)
    sound_speeds.append(sos)
    densities.append(density)
    sources.append(src)
    #pml.append(0.0)