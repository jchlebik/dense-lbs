import os
from functools import partial
from statistics import mean
from typing import TYPE_CHECKING, Iterable, Any

import jax
import jax.numpy as jnp
from flax import jax_utils, core
from flax.training.train_state import TrainState
import ml_collections
import tqdm
from absl import logging

if TYPE_CHECKING:
    from logger.base_logger import BaseLogger
    from utils.checkpointer import CheckpointManager
else:
    BaseLogger = Any
    CheckpointManager = Any
    
from input_pipeline import InputPipeline
import utils
from utils.key_mapper import KeyMapper as km

class DefaultTrainer:   
    config : ml_collections.ConfigDict
    steps_per_train_epoch : int
    steps_per_validation : int
    passed_epochs : int
    summary_writer : BaseLogger
    tracer : Any
    checkpointer : CheckpointManager
    property_index_map : dict

    def __init__(self, model: str):
        if model not in ['lbs', 'dlbs', 'corrector']:
            raise ValueError(f"Invalid model: {model}")
        
        if model == 'lbs':
            from models.builders.lbs_builder import LbsBuilder
            self.model_builder = LbsBuilder()
        elif model == 'dlbs':
            from models.builders.dlbs_builder import DlbsBuilder
            self.model_builder = DlbsBuilder()
        elif model == 'corrector':
            from models.builders.corrector_builder import CorrectorBuilder
            self.model_builder = CorrectorBuilder()
            
        self.best_validation_loss = float("inf")
        self.v_batch_to_log = None
        self.steps_per_train_epoch = 0
        self.steps_per_validation = 0
        self.passed_epochs = 0
        self.summary_writer = None
        self.tracer = None
        self.checkpointer = None
        self.property_index_map = {}

    def set_config(self, config):
        self.config = config
    
    def setup_iterators(self, model_keys):
        """
        Set up iterators for training and validation.

        Args:
            model_keys (list): List of model keys.

        Returns:
            tuple: A tuple containing the following elements:
                - iterators (dict): A dictionary containing the training and validation iterators.
                - config (object): The configuration object.
                - prop_index_map (dict): A dictionary mapping property names to their corresponding indices in the dataset.

        """
        iterators, dataset_metadata = InputPipeline('tensorflow').create_input_iter(self.config, model_keys)
        self.config.image_size = dataset_metadata["image_size"]
        
        if iterators["val"]["size"] == 0:
            logging.warning("Validation dataset is empty. Using cross-validation with training dataset.")
            iterators["val"]["size"] = int(0.2 * iterators["train"]["size"])
            iterators["val"]["iter"] = iterators["train"]["iter"]
        
        self.steps_per_train_epoch = iterators["train"]["size"] // (self.config.per_device_batch_size * self.config.num_devices)
        self.steps_per_validation = iterators["val"]["size"] // (self.config.per_device_batch_size * self.config.num_devices)
        return iterators, self.config, dataset_metadata["prop_index_map"]
    
    def start_training(self, config) -> TrainState:
        """
        Starts the training process.

        Args:
            config: The configuration object containing training parameters.

        Returns:
            The final training state.

        """
        self.config = config
        
        model = self.model_builder.create_model(config)
        optimizer = self.model_builder.create_optimizer(config)

        iterators, self.config, self.property_index_map = self.setup_iterators(model.get_keys())
        
        t_state = self.model_builder.setup_train_state(self.config, model, optimizer)

        t_state, self.config, self.passed_epochs = utils.handle_checkpoint_restoration(self.config, t_state)
        
        self.summary_writer, self.tracer, self.checkpointer = utils.initialize_logging_utilities(self.config.tensorboard_dir, 
                                                                                                 self.config.checkpoint_dir, 
                                                                                                 enable_profiling= False)
        self.summary_writer.write_hparams(hparams = dict(self.config))
        t_state = jax_utils.replicate(t_state)  # Replicate the network across all devices for ddp

        logging.info("Starting the training process...")
        
        best_validation_losses = [1e9] * 3
        ############# START OF TRAINING LOOP #############
        for epoch in tqdm.trange(self.passed_epochs, config.num_epochs, desc="Epoch", position=0, disable=os.environ.get("DISABLE_TQDM", False)):
            t_state = self.training_loop(t_state, iterators["train"]["iter"], epoch)
            if (epoch + 1) % config.validate_every_n_epochs == 0:
                #continue
                best_validation_losses = self.validation_loop(t_state, iterators["val"]["iter"], epoch, best_validation_losses)
        ############## END OF TRAINING LOOP ##############
        
        self.summary_writer.flush()
        self.summary_writer.close()
        logging.info("Training done, waiting for computations to finish")

        # Wait until computations are done before exiting
        jax.random.normal(jax.random.key(0), ()).block_until_ready()

        return jax_utils.unreplicate(t_state)
        
    @partial(jax.pmap, static_broadcasted_argnums=(0))
    def predict(self, t_state: TrainState, kwargs_dict):
        """
        Perform prediction using the given TrainState and a batch.

        Args:
            t_state (TrainState): The TrainState object containing the necessary parameters.
            kwargs_dict (dict): The batch - a dictionary of keyword arguments.

        Returns:
            The result of applying the prediction function to the TrainState and additional arguments.

        """
        return t_state.apply_fn({"params": t_state.params}, **kwargs_dict)
     
    @partial(jax.pmap, static_broadcasted_argnums=(0), in_axes=(None, 0, 0), axis_name="data")
    def train_batch(self, t_state: TrainState, kwargs_dict):
        """
        Trains a t_state model on a batch of data. JAX PMAPed for multi-gpu.

        Args:
            t_state (TrainState): The current training state.
            kwargs_dict (dict): The batch - a dictionary of keyword arguments.

        Returns:
            tuple: A tuple containing the loss and gradients.
        """
        self_loss_fn = partial(DefaultTrainer.loss_fn, self)
        val_and_grad_fn = jax.value_and_grad(self_loss_fn, argnums=1, has_aux=True)
        (loss, pred_field_batch), grads = val_and_grad_fn(t_state, t_state.params, **kwargs_dict)

        all_reduce_metrics = {
            'loss': loss,
            'grads': grads
        }
        
        # Combine the gradient across all devices by taking their mean.
        all_reduce_metrics = jax.lax.pmean(all_reduce_metrics, axis_name="data")
        return all_reduce_metrics['loss'], all_reduce_metrics['grads']
        
    @partial(jax.pmap, static_broadcasted_argnums=(0), in_axes=(None, 0, 0), axis_name="data")
    def validate_batch(self, t_state: TrainState, kwargs_dict):
        """
        Performs validation using given t_state model and a batch of data and calculates the loss and accuracy.
        JAX PMAPed for multi-gpu.

        Args:
            t_state (TrainState): The current training state.
            kwargs_dict (dict): The batch - a dictionary of keyword arguments.

        Returns:
            tuple: A tuple containing the loss and accuracy values.

        """
        scaled_ground_truths_batch = self.config.training_scale_factor * kwargs_dict["ground_truths_batch"]
        loss_val, pred_field_batch = self.loss_fn(t_state, t_state.params, **kwargs_dict)
            
        error_field = jnp.abs(pred_field_batch - scaled_ground_truths_batch)
        max_val = jnp.max(jnp.abs(scaled_ground_truths_batch), axis=(1, 2, 3))
        acc = 1.0 - jnp.mean(error_field / max_val)
        acc = acc * 100.0
        
        all_reduce_metrics = {
            'loss': loss_val,
            'acc': acc
        }    
        # All-reduce the loss mean.
        all_reduce_metrics = jax.lax.pmean(all_reduce_metrics, axis_name="data")
        return all_reduce_metrics['loss'], all_reduce_metrics['acc']
     
    @partial(jax.pmap, static_broadcasted_argnums=(0), in_axes=(None, 0, 0), axis_name="data")
    def test_batch(self, t_state: TrainState, kwargs_dict):
        scaled_ground_truths_batch = self.config.training_scale_factor * kwargs_dict["ground_truths_batch"]
        loss_val, pred_field_batch = self.loss_fn(t_state, t_state.params, **kwargs_dict)
            
        error_field = jnp.abs(pred_field_batch - scaled_ground_truths_batch)
        max_val = jnp.max(jnp.abs(scaled_ground_truths_batch), axis=(1, 2, 3))
        acc = 1.0 - jnp.mean(error_field / max_val)
        acc = acc * 100.0
        
        return loss_val, acc
    
    @partial(jax.jit, static_argnums=(0))
    def loss_fn(self, t_state: TrainState, params: core.FrozenDict[str, Any], **kwargs):
        """
        Runs the model and calculates the loss function and parameters. JAX JITed for speed.

        Args:
            t_state (TrainState): The current training state.
            params (core.FrozenDict[str, Any]): The parameters for the loss function.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple: A tuple containing the loss value and the predicted field batch.
        """
        scaled_ground_truths_batch = self.config.training_scale_factor * kwargs["ground_truths_batch"]
        del kwargs["ground_truths_batch"]

        pred_field_batch = t_state.apply_fn({"params": params}, **kwargs)
        
        diff = jnp.abs(pred_field_batch - scaled_ground_truths_batch)
        
        losses = {
            "l2": jnp.mean(diff ** 2) if self.config.loss_fun == "l2" else None,
            "linf": jnp.mean(jnp.amax(diff, axis=(1, 2))) if self.config.loss_fun == "linf" else None,
            "l1": jnp.mean(diff) if self.config.loss_fun == "l1" else None,
        }
    
        if self.config.loss_fun not in losses:
            raise ValueError(f"Invalid loss function: {self.config.loss_fun}")
        
        return losses[self.config.loss_fun], pred_field_batch
    
    @partial(jax.pmap, static_broadcasted_argnums=(0), in_axes=(None, 0, 0))
    def update_model(self, t_state: TrainState, parameter_gradients) -> Any:
        """
        Updates the model with the given parameter gradients. JAX PMAPed for multi-gpu.

        Args:
            t_state (TrainState): The current training state.
            parameter_gradients: The gradients of the model parameters.

        Returns:
            Any: The result of applying the gradients to the training state.
        """
        return t_state.apply_gradients(grads=parameter_gradients)
    
    def training_loop(self, t_state: TrainState, train_it: Iterable[Any], epoch: int):
        """
        Executes the training loop for a single epoch.

        Args:
            t_state (TrainState): The current training state.
            train_it (Iterable[Any]): The training data iterator.
            epoch (int): The current epoch number.

        Returns:
            TrainState: The updated training state after the epoch.
        """
        disable_tracing = os.environ.get("DISABLE_TRACING", False) or self.tracer is None
        progress_bar = tqdm.trange(0, self.steps_per_train_epoch, desc="Training", unit="batch", position=1, leave=False, 
                                disable=os.environ.get("DISABLE_TQDM", False))
        train_losses = []
        for step, t_batch in zip(progress_bar, train_it):
            # batch.shape == (6, n_devices, batch_size_per_device, img_size1, img_size2, *)
            kwargs = self.model_builder.kwargs_builder(self.property_index_map, t_batch)
            kwargs["ground_truths_batch"] = t_batch[self.property_index_map["full_fields"]]


            #loss = pmap_test(kwargs["sound_speeds_batch"], kwargs["densities_batch"])
            loss, grads = self.train_batch(t_state, kwargs)
            t_state = self.update_model(t_state, grads)

            train_losses.append(float(loss[0]))
            progress_bar.set_postfix({"Batch loss": loss[0]})

            if not disable_tracing and epoch == 0:
                self.tracer(step)

        train_loss = jnp.mean(jnp.array(train_losses))

        logging.debug("epoch:% 3d, train_loss: %.4f" % (epoch, train_loss))
        if self.summary_writer is not None:
            self.summary_writer.write_scalars(epoch, {"Loss/train": train_loss})
            #summary_writer.write_histograms(epoch, {f"Grads_{epoch//20}" : jnp.concatenate([g.flatten() for g in jax.tree.leaves(grads)])})
        return t_state
    
    def validation_loop(self, t_state: TrainState, val_it: Iterable[Any], epoch: int, best_validation_losses: float):
        """
        Perform the validation loop for a given epoch.

        Args:
            t_state (TrainState): The current training state.
            val_it (Iterable[Any]): The validation iterator.
            epoch (int): The current epoch number.

        Returns:
            float: The best validation loss.
        """
        validation_losses_container = []
        validation_accuracy_container = []
        #best_validation_loss = self.best_validation_loss
        
        v_batch_to_log = None

        if self.steps_per_validation == 0:
            logging.warning("Validation dataset is empty.")
            return best_validation_losses

        progress_bar = tqdm.trange(0, self.steps_per_validation, desc="Validation", unit="batch", position=2, leave=False, 
                                disable=os.environ.get("DISABLE_TQDM", False))
        for step, v_batch in zip(progress_bar, val_it):
            # batch.shape == (4, n_devices, batch_size_per_device, img_size1, img_size2, 1)
            kwargs = self.model_builder.kwargs_builder(self.property_index_map, v_batch)
            kwargs["ground_truths_batch"] = v_batch[self.property_index_map[km.get_full_field_key()]]

            loss, acc = self.validate_batch(t_state, kwargs)

            progress_bar.set_postfix({"Loss": loss[0], "Accuracy": acc[0]})
            validation_losses_container.append(float(loss[0]))
            validation_accuracy_container.append(float(acc[0]))
            
            if v_batch_to_log is None:
                v_batch_to_log = v_batch

        validation_loss = float(jnp.mean(jnp.array(validation_losses_container)))
        validation_accuracy = float(jnp.mean(jnp.array(validation_accuracy_container)))
        #logging.debug("validation_loss: %.4f, best_validation_loss: %.4f" % (validation_loss, best_validation_loss))

        if self.summary_writer is not None:
            kwargs = self.model_builder.kwargs_builder(self.property_index_map, v_batch_to_log)

            #it is faster to run on multigpu and just take one result than unreplicating the entire train state
            predicted_field_batch = self.predict(t_state, kwargs)

            fields_to_log = self.summary_writer.prepare_acoustic_properties_for_logging(
                v_batch_to_log, 
                predicted_field_batch / self.config.training_scale_factor,
                self.property_index_map,
                self.model_builder.get_model_keys(),
                skip_keys_substr=["pml", "source"]
            )
            
            self.summary_writer.log_acoustic_fields(
                step_n = epoch, 
                fields = fields_to_log
            )

            self.summary_writer.write_scalars(epoch, {"Loss/validation": validation_loss})
            self.summary_writer.write_scalars(epoch, {"Accuracy": validation_accuracy})

        ####### CHECKPOINT #######
        for i in range(len(best_validation_losses)):
            if validation_loss < best_validation_losses[i]:
                
                tmp, best_validation_losses[i] = best_validation_losses[i], validation_loss
                if i < len(best_validation_losses) - 1:
                    for j in range(i + 1, len(best_validation_losses)):
                        tmp, best_validation_losses[j] = best_validation_losses[j], tmp
                    
                if self.config.enable_checkpointing and epoch >= self.config.checkpointing_warmup:
                    self.checkpointer.save_checkpoint(jax_utils.unreplicate(t_state), dict(self.config), epoch + 1, validation_loss)
                break
            
        return best_validation_losses