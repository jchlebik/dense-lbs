import sys
sys.path.append('/home/xchleb07/dev/dlbs')
import os
#os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = ".90"
os.environ['DISABLE_TQDM'] = "True"

import jax
from flax import jax_utils
from absl import flags, app
import ml_collections
from ml_collections import config_flags
import tqdm

import utils
import train

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
flags.DEFINE_string('dataset_file_path', None, 'Path to the .npz MNISTHelmholtz generated file.')
#flags.DEFINE_enum('launch_option', None, ['train', 'test'], 'How to launch the model.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=False
)

def main(argv):
    FLAGS = flags.FLAGS
    FLAGS.config.dataset_file_path = os.path.abspath(FLAGS.dataset_file_path)

    FLAGS.config.output_dir, FLAGS.config.checkpoint_dir, FLAGS.config.tensorboard_dir \
        = utils.create_output_folder_structure(FLAGS.workdir)

    FLAGS.config.enable_checkpointing = True
    FLAGS.config.checkpointing_warmup = 0

    config = ml_collections.FrozenConfigDict(FLAGS.config)

    iterators, dataset_metadata =  utils.InputPipeline('tensorflow').create_input_iter(config)
    
    global_batch_size = config.per_device_batch_size * config.num_devices
    steps_per_train_epoch = iterators["train"]["size"] // global_batch_size
    steps_per_validation = iterators["val"]["size"] // global_batch_size

    summary_writer, tracer, checkpointer = utils.init_logging_utilities(config.tensorboard_dir, config.checkpoint_dir)

    summary_writer.write_hparams(dict(config))

    t_state = jax_utils.replicate(train.create_train_state(config, dataset_metadata["image_size"]))
    starting_epoch = 0
        
    best_v_loss = float("inf")
    #for i in tqdm.trange(starting_epoch, starting_epoch + 5):
    t_state = train.training_loop(config, t_state, iterators["train"]["iter"], None, None, steps_per_train_epoch, 0)
    best_v_loss = train.validation_loop(config, t_state, iterators["val"]["iter"], None, checkpointer, best_v_loss, steps_per_validation, 0)
    
    summary_writer.flush()
    summary_writer.close()
    
    jax.random.normal(jax.random.key(0), ()).block_until_ready()
    
    t_state = jax_utils.unreplicate(t_state)
    print(best_v_loss)
    

if __name__ == '__main__':
  flags.mark_flags_as_required(['config', 'workdir', 'dataset_file_path'])
  app.run(main)

# --config "config/train_config.py" 
# --workdir "logdir/" 
# --dataset_file_path "./128_16_(1.0, 3.0)_(1.0, 2.0)_1.0_20000_complex64_3fef0e92ef34efac438cf60335020057.npz"
