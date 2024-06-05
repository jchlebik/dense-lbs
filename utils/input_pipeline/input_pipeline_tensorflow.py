from absl import logging
from flax import jax_utils
import jax

import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')
# ^ Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make it unavailable to JAX.

from utils.input_pipeline.input_pipeline_interface import IInputPipeline
from dataset_generator.dataset_generator import MNISTHelmholtz

from utils.key_mapper import KeyMapper as km

class InputPipeline_Tensorflow(IInputPipeline):

    def _create_split(
        self,
        dts : MNISTHelmholtz,
        train_ratio : float,
        val_ratio : float,
        train_batch_size : int,
        val_batch_size : int,
        use_cache : bool = False,
        shuffle_buffer_size : int = 2000,
        prefetch_num : int = 1,
        keys = None
    ) -> tuple[tf.data.Dataset, int, tf.data.Dataset, int]:
        # Splitting dataset
        train_size = int(train_ratio * len(dts))
        val_size = int(val_ratio * len(dts))
        test_size = len(dts) - train_size - val_size
        
        #"full_fields"
        #"sos_fields"
        #"density_fields"

        model_dts_mapping = km.get_model_to_dataset_mapping()           

        
        # prop_index_map = {
        #     'sound_speeds' : 0,
        #     'densities' : 1,
        #     'sources' : 2,
        #     'pmls' : 3,
        #     'full_fields' : 4,
        #     'sos_fields' : 5
        # }

        prop_index_map = {}
        
        if train_size == 0:
            train_ds = None
        else:
            train_dts_jnp = dts.collate(0, train_size)

            batch_data = []
            for i, key in enumerate(keys):
                if key not in model_dts_mapping:
                    raise ValueError(f"Invalid key: {key}")
                batch_data.append(train_dts_jnp[model_dts_mapping[key]])
                prop_index_map[model_dts_mapping[key]] = i
                

            train_ds = tf.data.Dataset.from_tensor_slices(batch_data)

            if use_cache:
                train_ds = train_ds.cache()
                
            if shuffle_buffer_size == -1:
                shuffle_buffer_size = train_ds.cardinality()

            train_ds = train_ds.shuffle(shuffle_buffer_size, seed = 0)
            train_ds = train_ds.repeat()
            train_ds = train_ds.batch(train_batch_size, drop_remainder=True)
            train_ds = train_ds.prefetch(prefetch_num)
    
        if val_size == 0:
            val_ds = None
        else:
            val_dts_jnp = dts.collate(train_size, train_size + val_size)

            batch_data = []
            for i, key in enumerate(keys):
                if key not in model_dts_mapping:
                    raise ValueError(f"Invalid key: {key}")
                batch_data.append(val_dts_jnp[model_dts_mapping[key]])
                prop_index_map[model_dts_mapping[key]] = i

            val_ds = tf.data.Dataset.from_tensor_slices(batch_data)

            if use_cache:
                val_ds = val_ds.cache()

            val_ds = val_ds.repeat()
            val_ds = val_ds.batch(val_batch_size, drop_remainder=True)
            val_ds = val_ds.prefetch(prefetch_num)

        return train_ds, train_size, val_ds, val_size, prop_index_map

    def prepare_tf_data(self, xs):
        """Convert an input batch from tf Tensors to numpy arrays."""
        local_device_count = jax.local_device_count()

        def _prepare_(x):
            # Use _numpy() for zero-copy conversion between TF and NumPy.
            x = x._numpy()  # pylint: disable=protected-access

            # reshape (host_batch_size, height, width, *) to (local_devices, device_batch_size, height, width, *)
            x = x.reshape((local_device_count, -1) + x.shape[1:])
            return jax.numpy.rot90(x, k = 1, axes=(2, 3))
        
        out = jax.tree_util.tree_map(_prepare_, xs)
        return out

    def create_input_iter(self, config, model_keys = None):
        logging.info("Creating input iterators...")
        logging.debug("Loading train and validation datasets from file %s", config.dataset_file_path)
        
        dataset = MNISTHelmholtz.create_obj_from_finished_dataset(config.dataset_file_path).load(config.samples_per_epoch)
        
        logging.debug("Dataset loaded with %d samples", len(dataset))
        logging.debug("Creating input iterators")
        train_ds, train_size, val_ds, val_size, prop_index_map = self._create_split(
            dataset,
            train_ratio         = config.train_ratio,
            val_ratio           = config.validation_ratio,
            train_batch_size    = config.per_device_batch_size * config.num_devices,
            val_batch_size      = config.per_device_batch_size * config.num_devices,
            use_cache           = config.use_cache,
            shuffle_buffer_size = config.shuffle_buffer_size,
            prefetch_num        = config.prefetch_num,
            keys                = model_keys
        )

        if train_ds is not None:
            train_it = map(self.prepare_tf_data, train_ds)
            train_it = jax_utils.prefetch_to_device(train_it, 2)
        else:
            train_it = None

        if val_ds is not None:
            val_it = map(self.prepare_tf_data, val_ds)
            val_it = jax_utils.prefetch_to_device(val_it, 2)
        else:
            val_it = None

        iterators = {
            'train' :  { 'iter' : train_it, 'size' : train_size },
            'val'   :  { 'iter' : val_it, 'size' : val_size },
        }

        dataset_metadata = {
            'image_size'    : dataset.image_size,
            'pml_size'      : dataset.pml_size,
            'prop_index_map': prop_index_map,
        }
        del dataset

        return iterators, dataset_metadata
