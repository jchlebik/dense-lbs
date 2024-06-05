import os
import pickle
from functools import partial, partialmethod
from hashlib import md5
import sys
from typing import Any

import flax.jax_utils
from jwave import FourierSeries
from jwave.acoustics.time_harmonic import helmholtz, helmholtz_solver
from jwave.geometry import Domain, Medium

import tqdm
from absl import app, flags
from ml_collections import config_flags

import jax
from jax import numpy as jnp

if __name__ == '__main__':
    from backend.mnist_downloader import MNISTDownloader
else:
    from .backend.mnist_downloader import MNISTDownloader

# hashable struct to have named parameters
class SimParams:
    """
    Class representing simulation parameters.

    Attributes:
        N (int): Number of simulations.
        dx (float): Step size.
        domain (str): Simulation domain.
    """

    def __init__(self, N, dx, domain) -> None:
        self.N = N
        self.dx = dx
        self.domain = domain


class MNISTHelmholtz:
    image_size: int = 128
    pml_size: int = 32
    sound_speed_lims: tuple[float, float] = (1.0, 1.3)
    density_lims: tuple[float, float] = (1.0, 1.9)
    labels_filter: tuple[int] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    omega: float = 1.0
    amplitude: float = 10.0
    src_pos: tuple[int, int] = (32, 32)
    num_samples: int = 10000
    dtype: jnp.dtype = jnp.float32
    sos_fields: list[Any] = []
    density_fields: list[Any] = []
    full_fields: list[Any] = []
    sound_speeds: list[Any] = []
    densities: list[Any] = []
    sources: list[Any] = []
    pml: list[Any] = []
    out_path: str = "."
    
    def __init__(
        self,
        image_size: int = 128,
        pml_size: int = 32,
        sound_speed_lims: tuple[float, float] = (1.0, 1.3),
        density_lims: tuple[float, float] = (1.0, 1.9),
        labels_filter: tuple[int] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        omega: float = 1.0,
        amplitude: float = 10.0,
        src_pos: tuple[int, int] = (32, 32),
        num_samples: int = 10000,
        dtype: jnp.dtype = jnp.float32,
        out_path: str = ".",
    ) -> None:
        self.image_size = image_size
        self.pml_size = pml_size
        self.sound_speed_lims = sound_speed_lims
        self.density_lims = density_lims
        self.labels_filter = labels_filter
        self.omega = omega
        self.amplitude = amplitude
        self.src_pos = src_pos
        self.num_samples = num_samples
        self.dtype = dtype
        self.out_path = out_path
    

    @classmethod
    def create_obj_from_finished_dataset(cls, filepath: str):
        with open(filepath, "rb") as f:
            dict_to_load = pickle.load(f)
            num_samples = dict_to_load["num_samples"]
            pml = dict_to_load["data"]["pml"]
            image_size = dict_to_load["image_size"]
            pml_size = dict_to_load["pml_size"]
            sound_speed_lims = dict_to_load["sound_speed_lims"]
            density_lims = dict_to_load["density_lims"]
            omega = dict_to_load["omega"]
            dtype = dict_to_load["dtype"]

        return cls(
            image_size=image_size,
            pml_size=pml_size,
            sound_speed_lims=sound_speed_lims,
            density_lims=density_lims,
            omega=omega,
            num_samples=num_samples,
            out_path=os.path.dirname(filepath),
            dtype=dtype,
        )

    @property
    def filepath(self):
        filename = f"{self.image_size}_{self.pml_size}_{self.sound_speed_lims}_{self.density_lims}_{self.omega}_{self.num_samples}_{jnp.dtype(self.dtype)}"
        hashed_name = md5(filename.encode()).hexdigest()
        # hashed_name = sha256(filename.encode()).hexdigest()
        return f"{self.out_path}/{filename}_{hashed_name}.npz"

    def __len__(self):
        return len(self.sos_fields)

    def __getitem__(self, idx):
        # Take absolute value if dtype is float32
        if self.dtype == jnp.float32:
            sos_field = jnp.abs(self.sos_fields[idx])
            density_field = jnp.abs(self.density_fields[idx])
            full_field = jnp.abs(self.full_fields[idx])
        else:
            sos_field = self.sos_fields[idx]
            density_field = self.density_fields[idx]
            full_field = self.full_fields[idx]
        return {
            "sound_speed_field": sos_field,
            "density_field": density_field,
            "full_field": full_field,
            "sound_speed": self.sound_speeds[idx],
            "density": self.densities[idx],
            "pml": self.pml,
            "source": self.sources[idx],
        }

    def save(self):
        """
        Saves the dataset to a file.
        """
        dict_to_save = {
            "data": {
                "sound_speed_fields": self.sos_fields,
                "density_fields": self.density_fields,
                "full_fields": self.full_fields,
                "sound_speeds": self.sound_speeds,
                "densities": self.densities,
                "sources": self.sources,
                "pml": self.pml,
                "src_position": self.src_pos,
                "amplitude": self.amplitude,
            },
            "image_size": self.image_size,
            "pml_size": self.pml_size,
            "sound_speed_lims": self.sound_speed_lims,
            "density_lims": self.density_lims,
            "omega": self.omega,
            "num_samples": self.num_samples,
            "dtype": self.dtype,
            "label_filter": self.labels_filter,
        }
        with open(self.filepath, "wb") as f:
            pickle.dump(dict_to_save, f)

    def load(self, num_samples=None):
        with open(self.filepath, "rb") as f:
            dict_to_load = pickle.load(f)
            if (num_samples is not None) and (num_samples > dict_to_load["num_samples"]):
                print("Requested number of samples larger than the dataset, defaulting ...")

            if (num_samples is None) or (num_samples >= dict_to_load["num_samples"]):
                self.full_fields = dict_to_load["data"]["full_fields"]
                self.sos_fields = dict_to_load["data"]["sound_speed_fields"]
                self.density_fields = dict_to_load["data"]["density_fields"]
                self.sound_speeds = dict_to_load["data"]["sound_speeds"]
                self.densities = dict_to_load["data"]["densities"]
                self.sources = dict_to_load["data"]["sources"]
                self.num_samples = dict_to_load["num_samples"]
            else:
                self.full_fields = dict_to_load["data"]["full_fields"][:num_samples, :]
                self.sos_fields = dict_to_load["data"]["sound_speed_fields"][:num_samples, :]
                self.density_fields = dict_to_load["data"]["density_fields"][:num_samples, :]
                self.sound_speeds = dict_to_load["data"]["sound_speeds"][:num_samples, :]
                self.densities = dict_to_load["data"]["densities"][:num_samples, :]
                self.sources = dict_to_load["data"]["sources"][:num_samples, :]
                self.num_samples = num_samples

            self.pml = dict_to_load["data"]["pml"]
            self.image_size = dict_to_load["image_size"]
            self.pml_size = dict_to_load["pml_size"]
            self.sound_speed_lims = dict_to_load["sound_speed_lims"]
            self.density_lims = dict_to_load["density_lims"]
            self.omega = dict_to_load["omega"]
            self.dtype = dict_to_load["dtype"]
            self.labels_filter = dict_to_load["label_filter"]
        return self

    def collate(self, start_idx, end_idx):
        """
        Collates the dataset into a batch.

        Args:
        - offset (int): The offset index.
        - batch_size (int): The batch size.

        Returns:
        - dict: The collated batch.
        """
        return {
            MNISTHelmholtz.get_full_field_key(): self.full_fields[start_idx:end_idx],
            MNISTHelmholtz.get_sos_field_key(): self.sos_fields[start_idx:end_idx],
            MNISTHelmholtz.get_density_field_key(): self.density_fields[start_idx:end_idx],
            MNISTHelmholtz.get_sound_speed_key(): self.sound_speeds[start_idx:end_idx],
            MNISTHelmholtz.get_density_key(): self.densities[start_idx:end_idx],
            MNISTHelmholtz.get_source_key(): self.sources[start_idx:end_idx],
            MNISTHelmholtz.get_pml_key(): jnp.stack([ self.pml for _ in range(start_idx, end_idx)]),
            "image_size": self.image_size,
            "pml_size": self.pml_size,
            "sound_speed_lims": self.sound_speed_lims,
            "density_lims": self.density_lims,
            "omega": self.omega,
            "num_samples": self.num_samples,
            # "pml": jnp.stack([item for item in self.pml], axis=0),
        }

    @staticmethod
    def get_collate_keys():
        return [
            MNISTHelmholtz.get_full_field_key(),
            MNISTHelmholtz.get_sos_field_key(),
            MNISTHelmholtz.get_density_field_key(),
            MNISTHelmholtz.get_sound_speed_key(),
            MNISTHelmholtz.get_density_key(),
            MNISTHelmholtz.get_source_key(),
            MNISTHelmholtz.get_pml_key()
        ]
    
    @staticmethod
    def get_sound_speed_key():
        return "sound_speeds"

    @staticmethod
    def get_density_key():
        return "densities"
    
    @staticmethod
    def get_source_key():
        return "sources"
    
    @staticmethod
    def get_pml_key():
        return "pmls"
    
    @staticmethod
    def get_sos_field_key():
        return "sos_fields"
    
    @staticmethod
    def get_density_field_key():
        return "density_fields"
    
    @staticmethod
    def get_full_field_key():
        return "full_fields"
        

    # Resize the images
    @partial(jax.jit, static_argnums=(0,))
    def _transform_image(self, image, limits):
        # Resizing
        im = jax.image.resize(image, (self.image_size, self.image_size), "nearest")
        im = jnp.pad(im, self.pml_size, mode="edge")
        im = jnp.expand_dims(im, -1)

        # Fixing range
        im = im * (limits[1] - limits[0]) + limits[0]
        return im

    def _download_mnist(self):
        return MNISTDownloader("tensorflow").get(self.num_samples, self.labels_filter)

    def _mnist_acoustic_transform(self, mnist_images):
        @jax.jit
        def __mnist_acoustic_transform_scan_loop_body__(carry, x):
            out = (
                self._transform_image(x, self.sound_speed_lims),
                self._transform_image(x, self.density_lims),
            )
            return (carry, out)

        (carry, stack) = jax.lax.scan(__mnist_acoustic_transform_scan_loop_body__, None, mnist_images)
        return (stack[0], stack[1])

    def _get_simulation_preliminaries(self):
        # Defining simulation parameters
        N = tuple([self.image_size + 2 * self.pml_size] * 2)
        dx = (1.0, 1.0)
        domain = Domain(N, dx)
        return SimParams(N, dx, domain)

    @partial(jax.jit, static_argnums=(0, 4))
    def _helmholtz_simulate(self, speed_of_sound, density, src, domain):
        """
        Simulates the Helmholtz equation.

        Args:
        - speed_of_sound: The speed of sound.
        - density: The density.
        - src: The source.
        - domain: The domain.

        Returns:
        - The simulated result.
        """
        speed_of_sound = FourierSeries(speed_of_sound, domain)
        density = FourierSeries(density, domain)
        
        sos_only_medium = Medium(domain=domain, sound_speed=speed_of_sound, pml_size=self.pml_size)
        density_only_medium = Medium(domain=domain, density=density, pml_size=self.pml_size)
        full_medium = Medium(domain=domain, sound_speed=speed_of_sound, density=density, pml_size=self.pml_size)
        
        sos_only_field = helmholtz_solver(sos_only_medium, self.omega, -src)
        density_only_field = helmholtz_solver(density_only_medium, self.omega, -src)
        full_field = helmholtz_solver(full_medium, self.omega, -src)
        return sos_only_field, density_only_field, full_field

    def _get_source(self, sim_params, position, amp):
        src_field = jnp.zeros(sim_params.N).astype(jnp.complex64)
        src_field = src_field.at[position[0], position[1]].set(amp)
        return (
            FourierSeries(jnp.expand_dims(src_field, -1), sim_params.domain)
            * self.omega
        )

    def _get_pml(self, sim_params, src):
        medium = Medium(
            domain=sim_params.domain,
            sound_speed=self.sound_speed_lims[1],
            density=self.density_lims[1],
            pml_size=self.pml_size,
        )
        sim_params = helmholtz.default_params(src, medium, omega=self.omega)
        pml = sim_params["pml_on_grid"][0].on_grid
        return jnp.stack(
            [pml[..., 0].real, pml[..., 0].imag, pml[..., 1].real, pml[..., 1].imag], -1
        )

    def _run_simulations(self, sim_params, sound_speeds, densities):
        """
        Run acoustic simulations for given sound speeds and densities.

        Args:
            sim_params (SimParams): Simulation parameters.
            sound_speeds (list): List of sound speeds.
            densities (list): List of densities.

        Returns:
            tuple: A tuple containing the simulated outfields, sources, sound speeds, and densities.
        """
        @jax.jit
        def __run_simulations_scan_loop_body__(carry, y):
            src = self._get_source(sim_params, position = self.src_pos, amp = self.amplitude)
            # jit
            sos_only_field, density_only_field, full_field = self._helmholtz_simulate(y[0], y[1], src, sim_params.domain)

            out = (
                self._crop_edges(sos_only_field.on_grid, self.pml_size),
                self._crop_edges(density_only_field.on_grid, self.pml_size),
                self._crop_edges(full_field.on_grid, self.pml_size),
                self._crop_edges(src.on_grid, self.pml_size).real,
                self._crop_edges(y[0], self.pml_size),
                self._crop_edges(y[1], self.pml_size),
            )
            return (carry, out)

        ys = jnp.stack([sound_speeds, densities], axis = 1)
        (_, stack) = jax.lax.scan(__run_simulations_scan_loop_body__, None, ys)
        return (stack[0], stack[1], stack[2], stack[3], stack[4], stack[5])
        
    def generate(self, batched=False):
        """
        Generates the dataset.

        Args:
            batched (bool, optional): Whether to generate the dataset in batches for multiple devices. 
                                        Defaults to False.

        Returns:
            str: The filepath of the generated dataset.
        """
        r"""Generates the dataset"""

        ## Download MNIST dataset
        mnist_images = self._download_mnist() # (num_samples, 28, 28)

        ## Simulation preparations
        print("Preliminaries")
        initial_sim_params = self._get_simulation_preliminaries()

        if batched:
            def __create_batch_iterator(data, global_batch_size, num_devices):
                for i in range(0, len(data), global_batch_size):
                    batch = data[i : i + global_batch_size] # (global_batch_size, 28, 28)
                    batch = batch.reshape((num_devices, -1) + data.shape[1:] ) # (num_devices, local_batch_size, 28, 28)
                    yield batch
            
            num_devices = jax.local_device_count()
            global_batch_size = 50 * num_devices
            
            assert mnist_images.shape[0] % global_batch_size == 0, "Number of samples must be divisible by the global batch size."
            
            steps = mnist_images.shape[0] // global_batch_size
            
            batched_mnist_it = __create_batch_iterator(mnist_images, global_batch_size, num_devices)
            batched_mnist_it = flax.jax_utils.prefetch_to_device(batched_mnist_it, 2)
            
            f_transform_pmap = jax.pmap(partial(MNISTHelmholtz._mnist_acoustic_transform, self))
            f_run_simulations_pmap = jax.pmap(partial(MNISTHelmholtz._run_simulations, self, initial_sim_params))
        
            progress_bar = tqdm.trange(0, steps, desc="Generating", unit="batch", disable=os.environ.get("DISABLE_TQDM", False))
    
            for i, batch in zip(progress_bar, batched_mnist_it):
                (sound_speeds, densities) = f_transform_pmap(batch)
                (sos_fields, density_fields, full_fields, sources, sound_speeds, densities) \
                    = f_run_simulations_pmap(sound_speeds, densities)
                
                self.sos_fields.append(self._unshard_data(sos_fields))
                self.density_fields.append(self._unshard_data(density_fields))
                self.full_fields.append(self._unshard_data(full_fields))
                self.sources.append(self._unshard_data(sources))
                self.sound_speeds.append(self._unshard_data(sound_speeds))
                self.densities.append(self._unshard_data(densities))
            
            self.sos_fields = jnp.concatenate(self.sos_fields, axis=0)
            self.density_fields = jnp.concatenate(self.density_fields, axis=0)
            self.full_fields = jnp.concatenate(self.full_fields, axis=0)
            self.sources = jnp.concatenate(self.sources, axis=0) 
            self.sound_speeds = jnp.concatenate(self.sound_speeds, axis=0)
            self.densities = jnp.concatenate(self.densities, axis=0)
                        
        else:
            (sound_speeds, densities) = self._mnist_acoustic_transform(mnist_images)
            (
                self.sos_fields,
                self.density_fields,
                self.full_fields,
                self.sources,
                self.sound_speeds,
                self.densities,
            ) = self._run_simulations(initial_sim_params, sound_speeds, densities)

        ## Get the pml
        self.pml = self._crop_edges(
            field = self._get_pml(initial_sim_params, self._get_source(initial_sim_params, position = self.src_pos, amp = self.amplitude)),
            edge_size = self.pml_size,
        )

        # Save the dataset
        print("Saving dataset")
        self.save()
        return self.filepath

    def _unshard_data(self, data):
        """
        Reshapes the sharded data into a single array.

        Args:
            data: The sharded data array.

        Returns:
            The reshaped data array.
        """
        return jnp.reshape(data, (data.shape[0] * data.shape[1], *data.shape[2:]))

    def _crop_edges(self, field, edge_size):
        """
        Crop the edges of a given field by a specified edge size.

        Args:
            field (numpy.ndarray): The input field to be cropped.
            edge_size (int): The size of the edges to be cropped.

        Returns:
            numpy.ndarray: The cropped field.
        """
        return field[
            edge_size:-edge_size,
            edge_size:-edge_size,
        ]


#python dataset_generator.py --config "config/dataset_generation_config.py" --workdir "/tmp/"
if __name__ == '__main__':
    FLAGS = flags.FLAGS

    flags.DEFINE_string('workdir', None, 'Directory to store dataset data.')
    config_flags.DEFINE_config_file(
        'config',
        None,
        'File path to the configuration.',
        lock_config=True,
    )    
    flags.mark_flags_as_required(['config', 'workdir'])
    
    FLAGS(sys.argv)
    config = FLAGS.config
        
    print("Creating object.")
    dataset = MNISTHelmholtz(
        image_size=config.image_size,
        pml_size=config.pml_size,
        sound_speed_lims=config.sos_range,
        density_lims=config.rho_range,
        labels_filter=config.labels_filter,
        num_samples=config.num_samples,
        omega=config.omega,
        amplitude=config.amp,
        src_pos=config.src_pos,
        out_path=FLAGS.workdir,
        dtype=config.target,
    )
    print("Generating")
    dataset.generate(batched=True)
    print("Done")
