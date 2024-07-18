from absl import logging
from clu import periodic_actions
import ml_collections
from flax.training.train_state import TrainState
from logger.base_logger import BaseLogger
from utils import checkpointer, create_output_folder_structure


def handle_checkpoint_restoration(passed_cfg, t_state: TrainState):
    """
    Handles the restoration of training state from a checkpoint.

    Args:
        passed_cfg: The configuration object containing the training parameters.
        t_state: The current training state to serve as a template for restoration.

    Returns:
        A tuple containing the updated training state, the configuration object, and the number of passed epochs.
    """
    passed_epochs = 0

    if passed_cfg.get("use_checkpoint", False):
        ckpt = checkpointer.CheckpointManager(passed_cfg.checkpoint_path)
        t_state, config, passed_epochs = ckpt.restore_from_checkpoint(t_state)
        if passed_cfg.get("restart_from_checkpoint", False):
            config.output_dir, config.checkpoint_dir, config.tensorboard_dir = create_output_folder_structure(passed_cfg.workdir)
            config.num_epochs = passed_cfg.num_epochs
            config.loss_fun = passed_cfg.loss_fun
            config.validate_every_n_epochs = passed_cfg.validate_every_n_epochs
            config.training_scale_factor = passed_cfg.training_scale_factor
            passed_epochs = 0

        #utils.advance_iterator(iterators["train"]["iter"], passed_epochs * steps_per_train_epoch)
        logging.info(f"Restored training state from epoch {passed_epochs}")
    else:
        passed_cfg.output_dir, passed_cfg.checkpoint_dir, passed_cfg.tensorboard_dir \
            = create_output_folder_structure(passed_cfg.workdir)
        config = passed_cfg
    
    config = ml_collections.FrozenConfigDict(config)
    return t_state, config, passed_epochs

def initialize_logging_utilities(tensorboard_dir: str, checkpoint_dir: str, enable_checkpointing: bool = True, enable_profiling: bool = False):
    """
    Initializes logging utilities for tensorboard, profiling, and checkpointing.

    Args:
        tensorboard_dir (str): The directory to store tensorboard logs.
        checkpoint_dir (str): The directory to store checkpoints.

    Returns:
        tuple: A tuple containing the summary writer, tracer, and checkpointer objects.
    """
    summary_writer = BaseLogger("tensorboard")(tensorboard_dir)
    logging.info(f"Tensorboard file at {tensorboard_dir}")
    
    if enable_profiling:
        tracer = periodic_actions.Profile(logdir=tensorboard_dir, num_profile_steps=5, first_profile=5)
    else:
        tracer = None

    if enable_checkpointing:
        checkpoint_manager = checkpointer.CheckpointManager(checkpoint_dir)
    else:
        checkpoint_manager = None

    return summary_writer, tracer, checkpoint_manager