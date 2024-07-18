import sys
sys.path.append('/home/xchleb07/dev/dlbs')

from jwave import FourierSeries
from jwave.acoustics.time_harmonic import helmholtz, helmholtz_solver
from jwave.geometry import Domain, Medium

import tqdm

import jax
from jax import numpy as jnp
from matplotlib import pyplot as plt

from dataset_generator.dataset_generator import MNISTHelmholtz

dataset_file_path = "/mnt/proj1/open-28-36/chlebik/datasets/1_7/128_16_(1.5, 2.8)_(1.0, 1.85)_1_6000_complex64_95bf8365c4b3dc206630fbbb8ccf3f30.npz"
samples_per_epoch = 1000

dataset = MNISTHelmholtz.create_obj_from_finished_dataset(dataset_file_path).load(samples_per_epoch)

dts = dataset.collate(0, 1000)

full_field = dts[MNISTHelmholtz.get_full_field_key()]

plt.imshow(jnp.real(full_field[835]), cmap="inferno")
plt.colorbar()
plt.savefig("ff.png")
plt.close()

