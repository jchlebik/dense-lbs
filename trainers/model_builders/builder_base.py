from abc import abstractmethod, ABC
from typing import Any
import jax
from jax import numpy as jnp
from flax.training.train_state import TrainState
from utils.key_mapper import KeyMapper as km


class BuilderBase(ABC):
    """
    Abstract base class for builders.

    This class defines the common methods and attributes that all builders should have.
    """

    def __init__(self):
        self.model_keys = None
        self.dts_to_model_mapping = km.get_dataset_to_model_mapping()
        self.model_to_dts_mapping = km.get_model_to_dataset_mapping()

    def kwargs_builder(self, key_index_map: dict, batch: tuple[jnp.ndarray]) -> dict:
        """
        Build keyword arguments for the model using KeyMapper to map keys from the dataset to model. 

        Args:
            key_index_map (dict): A dictionary mapping keys to their corresponding indices from the dataloader.
            batch (tuple[jnp.ndarray]): A tuple of input arrays.

        Returns:
            dict: A dictionary of keyword arguments for the model.
        """
        kwargs = {}
        for key in self.model_keys:
            kwargs[key] = batch[key_index_map[self.model_to_dts_mapping[key]]]
        return kwargs
            
    def setup_train_state(self, config, model, optimizer) -> TrainState:
        """
        Set up the training state.

        Args:
            config: The configuration for the training.
            model: The model to be trained.
            optimizer: The optimizer for the training.

        Returns:
            TrainState: The initialized training state.
        """
        self.model_keys =  model.get_keys()
        empty_batch = {}
        for key in self.model_keys:
            if key in [ self.dts_to_model_mapping[km.get_sound_speed_key()], self.dts_to_model_mapping[km.get_density_key()]]:
                empty_batch[key] =  jnp.ones((1, config.image_size, config.image_size, 1))        
            elif key == self.dts_to_model_mapping[km.get_pml_key()]:
                empty_batch[key] = jnp.ones((1, config.image_size, config.image_size, 4))
            elif key in [self.dts_to_model_mapping[km.get_source_key()], self.dts_to_model_mapping[km.get_sos_field_key()], \
                         self.dts_to_model_mapping[km.get_full_field_key()], self.dts_to_model_mapping[km.get_density_field_key()]]:
                empty_batch[key] =  jnp.ones((1, config.image_size, config.image_size, 1), dtype=config.target)
            else:
                raise ValueError(f"Invalid key: {key}")
                    
        #kwargs = self.kwargs_builder(key_index_map, empty_batch)
        key, subkey = jax.random.split(jax.random.PRNGKey(config.seed))
        model_params = model.init(subkey, **empty_batch)
        return TrainState.create(apply_fn=model.apply, params=model_params['params'], tx=optimizer)

    @abstractmethod
    def create_model(self, config):
        """
        Create the model.

        Args:
            config: The configuration for the model.

        Returns:
            The created model.
        """
        pass
        
    @abstractmethod
    def create_optimizer(self, config):
        """
        Create the optimizer.

        Args:
            config: The configuration for the optimizer.

        Returns:
            The created optimizer.
        """
        pass