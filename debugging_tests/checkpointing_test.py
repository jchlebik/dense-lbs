import os
import sys
sys.path.append('/home/ichlebik/dev/dlbs')

import jax
from jax import tree_util
from flax.training.train_state import TrainState
from flax import jax_utils
import orbax.checkpoint

from absl import app
import ml_collections
import tqdm

from config import train_config
import utils
import train

def main(argv):
    config = train_config.get_config()
    config.dataset_file_path = os.path.abspath("128_16_(1.0, 3.0)_(1.0, 2.0)_1.0_20000_complex64_3fef0e92ef34efac438cf60335020057.npz")

    generate = False

    if generate:
        config.output_dir, config.checkpoint_dir, config.tensorboard_dir = utils.create_output_folder_structure(os.path.abspath("logdir/"))

        config = ml_collections.FrozenConfigDict(config)
        
        #checkpoint_manager = get_checkpoint_manager(config.checkpoint_dir)
        checkpoint_manager = utils.checkpointer.CheckpointManager(config.checkpoint_dir)
        #checkpoint_manager_raw = get_checkpoint_manager(config.tensorboard_dir)

        iterators, dataset_metadata = utils.InputPipeline('tensorflow').create_input_iter(config)
        train_it = iterators["train"]["iter"]
        global_batch_size = config.per_device_batch_size * config.num_devices
        steps_per_train_epoch = iterators["train"]["size"] // global_batch_size

        t_state = jax_utils.replicate(train.create_train_state(config, dataset_metadata["image_size"]))

        step = 0
        t_state = train.training_loop(config, t_state, train_it, None, None, steps_per_train_epoch, step)
        
        checkpoint_manager.save_checkpoint(jax_utils.unreplicate(t_state), config, step + 1, 20.0)
    else:
        config.checkpoint_dir = "/home/ichlebik/dev/dlbs/logdir/2024-03-28_13-11-25/checkpoints"
        config = ml_collections.FrozenConfigDict(config)
        checkpoint_manager = utils.checkpointer.CheckpointManager(config.checkpoint_dir)

        iterators, dataset_metadata = utils.InputPipeline('tensorflow').create_input_iter(config)
        train_it = iterators["train"]["iter"]
        global_batch_size = config.per_device_batch_size * config.num_devices
        steps_per_train_epoch = iterators["train"]["size"] // global_batch_size

        # Make "empty" training state
        t_state = train.create_train_state(config, dataset_metadata["image_size"])

        #t_state, config, epochs_finished = restore(t_state, checkpoint_manager)
        t_state, config, epochs_finished = checkpoint_manager.restore_from_checkpoint(t_state)
        utils.advance_iterator(train_it, epochs_finished * steps_per_train_epoch)
        t_state = jax_utils.replicate(t_state)

        #for step in tqdm.trange(epochs_finished, epochs_finished + 2):
        step = epochs_finished
        t_state = train.training_loop(config, t_state, train_it, None, None, steps_per_train_epoch, step)

        print(t_state, config, step)

if __name__ == '__main__':
    app.run(main)