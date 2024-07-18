from absl import logging
from flax import jax_utils
from jax import numpy as jnp
import jax

from torch import Generator
from torch.utils.data import DataLoader, random_split

from input_pipeline.backend.input_pipeline_interface import IInputPipeline
from dataset_generator.dataset_generator import MNISTHelmholtz

class InputPipeline_Torch(IInputPipeline):

    def create_input_iter(self, config):
        """
        Create input iterators for training and validation datasets.

        Args:
            config: A configuration object containing the following attributes:
                - dataset_file_path: The file path of the dataset.
                - samples_per_epoch: The number of samples per epoch.
                - train_ratio: The ratio of training data.
                - validation_ratio: The ratio of validation data.
                - per_device_batch_size: The batch size per device.
                - num_devices: The number of devices.

        Returns:
            A tuple containing the following:
            - iterators: A dictionary with keys 'train' and 'val', each containing a dictionary with keys 'iter' and 'size'.
            'iter' is the input iterator and 'size' is the size of the dataset.
            - dataset_metadata: A dictionary containing metadata about the dataset, including 'image_size' and 'pml_size'.
        """
        
        logging.debug("Loading train and validation datasets from file %s", config.dataset_file_path)
        dataset = MNISTHelmholtz.create_obj_from_finished_dataset(config.dataset_file_path).load(config.samples_per_epoch)
        logging.debug("Dataset loaded with %d samples", len(dataset))

        train_size = int(config.train_ratio * len(dataset))
        val_size = int(config.validation_ratio * len(dataset))
        test_size = len(dataset) - train_size - val_size

        train_ds, val_ds, test_ds = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=Generator().manual_seed(config.seed),
        )

        @jax.jit
        def collate_fn(batch):
            # in: batch[0:batch_size]["field" | "sound_speed" | "density" | "pml" | "source"]
            #      batch[0]["sound_speed"].shape == (1, 128, 128, 1*)
            sound_speeds = []
            densities = []
            sources = []
            pmls = []
            fields = []

            for item in batch:
                sound_speeds.append(item["sound_speed"])
                densities.append(item["density"])
                sources.append(item["source"])
                pmls.append(item["pml"])
                fields.append(item["full_fields"])


            #batch = jnp.stack([sound_speeds, densities, sources, pmls, fields], axis=1)
            transposed_batch = (
                jnp.array(sound_speeds), jnp.array(densities), 
                jnp.array(sources), jnp.array(pmls), jnp.array(fields),
            )

            # reshape (host_batch_size, height, width, 1*) to (local_devices, device_batch_size, height, width, 1*)
            return jax.tree_util.tree_map(
                lambda x: x.reshape((config.num_devices, -1) + x.shape[1:]),
                transposed_batch,
            )

        logging.debug("Creating input iterators")
        train_it = DataLoader(
            train_ds,
            batch_size=config.per_device_batch_size * config.num_devices,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
        )
        train_it = jax_utils.prefetch_to_device(train_it, 2)

        val_it = DataLoader(
            val_ds,
            batch_size=config.per_device_batch_size * config.num_devices,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
        )
        val_it = jax_utils.prefetch_to_device(val_it, 2)

        iterators = {
            "train": {"iter": train_it, "size": train_size},
            "val": {"iter": val_it, "size": val_size},
        }

        dataset_metadata = {
            "image_size": dataset.image_size,
            "pml_size": dataset.pml_size,
            "pml": dataset.pml,
        }
        return iterators, dataset_metadata
