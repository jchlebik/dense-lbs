import os
from functools import partial
from typing import TYPE_CHECKING, Iterable, Any

import jax
import jax.numpy as jnp
from flax import jax_utils
from flax.training.train_state import TrainState
import ml_collections
import tqdm
from absl import logging

if TYPE_CHECKING:
    from logger.base_logger import BaseLogger
else:
    BaseLogger = Any
    
from logger.field_plotting_settings import FieldPlottingSettings
import utils
from utils.key_mapper import KeyMapper as km
from .default_trainer import DefaultTrainer

class DefaultTester:   
    config : ml_collections.ConfigDict | ml_collections.FrozenConfigDict 
    summary_writer : BaseLogger | None
    property_index_map : dict
    keys_to_batch_map : dict
    
    def __init__(self, model: str):
        if model not in ['lbs', 'dlbs', 'corrector']:
            raise ValueError(f"Invalid model: {model}")
        
        self.trainer = DefaultTrainer(model)
        
        self.steps_per_testing = 0
        self.summary_writer = None
        self.property_index_map = {}
        
    def start_testing(self, config):
        self.config = config
        self.trainer.set_config(config)
        
        model = self.trainer.model_builder.create_model(config)
        optimizer = self.trainer.model_builder.create_optimizer(config)

        iterators, self.config, self.property_index_map = self.trainer.setup_iterators(model.get_keys())
        self.steps_per_testing = self.trainer.steps_per_train_epoch + self.trainer.steps_per_validation
               
        t_state, self.config, _ = \
            utils.checkpointer.CheckpointManager(self.config.checkpoint_path).restore_from_checkpoint(
                self.trainer.model_builder.setup_train_state(self.config, model, optimizer)
            )

        self.summary_writer, _, _ = utils.initialize_logging_utilities(
            self.config.tensorboard_dir, 
            None, 
            enable_checkpointing=False, 
            enable_profiling=False
        )
        
        #self.summary_writer.write_hparams(hparams = dict(self.config))
        t_state = jax_utils.replicate(t_state)  # Replicate the network across all devices for ddp

        logging.info("Starting the training process...")
        
        t_state = self.testing_loop(t_state, iterators["train"]["iter"], iterators["val"]["iter"])
        
        self.summary_writer.flush()
        self.summary_writer.close()
        logging.info("Training done, waiting for computations to finish")

        # Wait until computations are done before exiting
        jax.random.normal(jax.random.key(0), ()).block_until_ready()

        return jax_utils.unreplicate(t_state)
    
    def testing_loop(self, t_state: TrainState, train_it: Iterable[Any], val_it: Iterable[Any]):

        train_losses_container = []
        validation_losses_container = []
        
        #s_batch_to_log = None

        progress_bar = tqdm.trange(0, self.steps_per_testing, desc="Testing", unit="batch", position=0, leave=False, 
                                disable=os.environ.get("DISABLE_TQDM", False))
        
        for step, s_batch in enumerate(train_it):
            # batch.shape == (4, n_devices, batch_size_per_device, img_size1, img_size2, 1)
            kwargs = self.trainer.model_builder.kwargs_builder(self.property_index_map, s_batch)
            kwargs["ground_truths_batch"] = s_batch[self.property_index_map[km.get_full_field_key()]]

            losses = self.test_batch(t_state, kwargs)
            flat_losses = losses.flatten()

            progress_bar.set_postfix({"Loss": flat_losses.mean()})
            train_losses_container += flat_losses.tolist()
            progress_bar.update(1)
            if step == self.trainer.steps_per_train_epoch:
                break
        

        min_loss_dict = {
            'loss' : 1e9,
            'batch': None,
            'device_i': 0,
            'sample_i': 0
        }

        max_loss_dict = {
            'loss' : 0,
            'batch': None,
            'device_i': 0,
            'sample_i': 0
        }

        for step, s_batch in enumerate(val_it):
            # batch.shape == (S, n_devices, batch_size_per_device, img_size1, img_size2, 1)
            kwargs = self.trainer.model_builder.kwargs_builder(self.property_index_map, s_batch)
            kwargs["ground_truths_batch"] = s_batch[self.property_index_map[km.get_full_field_key()]]

            losses = self.test_batch(t_state, kwargs)
            flat_losses = losses.flatten()

            min_batch_loss = jnp.min(flat_losses)
            if min_batch_loss < min_loss_dict['loss']:
                indexes = jnp.argwhere(losses == min_batch_loss) #argwhere returns jnp array of arrays. i.e. Array([[4,5]], dtype=int32)
                min_loss_dict['device_i'] = int(indexes[0][0])
                min_loss_dict['sample_i'] = int(indexes[0][1])
                min_loss_dict['loss'] = min_batch_loss
                min_loss_dict['batch'] = s_batch
                #min_loss_dict['sample'] = [s_batch[i][device_i][sample_i] for i in range(len(s_batch))]

            max_batch_loss = jnp.max(flat_losses)
            if max_batch_loss > max_loss_dict['loss']:
                indexes = jnp.argwhere(losses == max_batch_loss) #argwhere returns jnp array of arrays. i.e. Array([[4,5,0]], dtype=int32)
                max_loss_dict['device_i'] = int(indexes[0][0])
                max_loss_dict['sample_i'] = int(indexes[0][1])
                max_loss_dict['loss'] = max_batch_loss
                max_loss_dict['batch'] = s_batch
                #max_loss_dict['sample'] = [s_batch[i][device_i][sample_i] for i in range(len(s_batch))]

            progress_bar.set_postfix({"Loss": flat_losses.mean()})
            validation_losses_container += flat_losses.tolist()
            progress_bar.update(1)
            if step == self.trainer.steps_per_validation:
                break

        if self.summary_writer is not None:
            self.summary_writer.write_boxplot(
                0,
                [train_losses_container, validation_losses_container],
                ["Training", "Validation"]
            )

            kwargs = self.trainer.model_builder.kwargs_builder(self.property_index_map, min_loss_dict['batch'])
            kwargs["ground_truths_batch"] = min_loss_dict['batch'][self.property_index_map[km.get_full_field_key()]]
            best_field = self.trainer.predict(t_state, kwargs)
            true_field = min_loss_dict['batch'][self.property_index_map[km.get_full_field_key()]][min_loss_dict['device_i']][min_loss_dict['sample_i']]
            fields_to_log = self.summary_writer.prepare_sample_comparison_for_logging(best_field[min_loss_dict['device_i']][min_loss_dict['sample_i']], true_field, "Best Sample", "Testing/")

            kwargs = self.trainer.model_builder.kwargs_builder(self.property_index_map, max_loss_dict['batch'])
            kwargs["ground_truths_batch"] = max_loss_dict['batch'][self.property_index_map[km.get_full_field_key()]]
            worst_field = self.trainer.predict(t_state, kwargs)
            true_field = max_loss_dict['batch'][self.property_index_map[km.get_full_field_key()]][max_loss_dict['device_i']][max_loss_dict['sample_i']]
            fields_to_log += self.summary_writer.prepare_sample_comparison_for_logging(worst_field, true_field, "Worst Sample", "Testing/")
        
            self.summary_writer.log_acoustic_fields(0, fields_to_log)

        return t_state
            
            
    @partial(jax.pmap, static_broadcasted_argnums=(0), in_axes=(None, 0, 0), axis_name="data")
    def test_batch(self, t_state: TrainState, kwargs_dict):
        #scaled_ground_truths_batch = self.config.training_scale_factor * kwargs_dict["ground_truths_batch"]
        loss_vals, pred_field_batch = self.trainer.loss_fn(t_state, t_state.params, **kwargs_dict)

        scaled_ground_truths_batch = self.config.training_scale_factor * kwargs_dict["ground_truths_batch"]

        pred_field_batch = t_state.apply_fn({"params": t_state.params}, **kwargs_dict)
        
        diff = jnp.abs(pred_field_batch - scaled_ground_truths_batch)

        losses = {
            "l2": jnp.mean(diff ** 2, axis=(1, 2, 3)) if self.config.loss_fun == "l2" else None,
            "linf": jnp.amax(diff, axis=(1, 2)) if self.config.loss_fun == "linf" else None,
            "l1": jnp.mean(diff, axis=(1, 2, 3)) if self.config.loss_fun == "l1" else None,
        }
    
        if self.config.loss_fun not in losses:
            raise ValueError(f"Invalid loss function: {self.config.loss_fun}")
        
        return losses[self.config.loss_fun]