
from matplotlib import pyplot as plt
import sys
sys.path.append('/home/ichlebik/dev/dlbs')


from jax import numpy as jnp
from absl import flags, app
import ml_collections
from ml_collections import config_flags


import utils

#flags.DEFINE_string('workdir', None, 'Directory to store model data.')
#flags.DEFINE_enum('launch_option', None, ['train', 'test'], 'How to launch the model.')
config_flags.DEFINE_config_file(
    'config',
    '/home/ichlebik/dev/dlbs/config/unet_train_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False
)

def main(argv):
    FLAGS = flags.FLAGS
    FLAGS.config.dataset_file_path =  "/mnt/Share/chlebik/128_16_(1.0, 3.0)_(1.0, 2.0)_1.0_10000_complex64_340899afbdf825aed66e10008a65dee2.npz"
    FLAGS.config.num_samples = 50
    #FLAGS.config.output_dir, FLAGS.config.checkpoint_dir, FLAGS.config.tensorboard_dir \
    #    = utils.create_output_folder_structure(FLAGS.workdir)

    #FLAGS.config.enable_checkpointing = True
    #FLAGS.config.checkpointing_warmup = 0

    config = ml_collections.FrozenConfigDict(FLAGS.config)

    iterators, dataset_metadata =  utils.InputPipeline('tensorflow').create_input_iter(config)
    
    t_batch = next(iterators["train"]["iter"])
    # batch.shape == (4, n_devices, batch_size_per_device, img_size1, img_size2, *)
    # t_batch: [sound_speed, density, source, pml, full_field, sos_field]
    
    full_field = t_batch[4][0]
    sos_field = t_batch[5][0]
    src = t_batch[2][0]	
    
    #a = jnp.rot90(full_field, k = 1, axes=(1, 2))
    #src = jnp.rot90(src, k = 3, axes=(1, 2))
    sos_field = jnp.rot90(sos_field, k = 3, axes=(1, 2))
    
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(jnp.real(sos_field[3]), cmap="seismic")
    ax[1].imshow(src[3], cmap="seismic")
    ax[0].set_title("SoS Field")
    ax[1].set_title("Source")
    
    
    # mrpe = 100 * (jnp.abs(full_field - sos_field) / jnp.max(jnp.abs(full_field)))
    # #mape = 100 * ((jnp.abs(full_field - sos_field)) / ( jnp.abs(full_field) ) + 1e-8)
    
    # to_plot = [jnp.real(sos_field), jnp.real(full_field), mrpe]
    # names = ["SoS Field", "Rho + SoS Field", "Relative Difference %"]
    # cmaps = ["seismic", "seismic", "inferno"]
    # fig, ax = plt.subplots(1, len(to_plot), figsize=(20, 5))
    # for i in range(len(to_plot)):
    #     raster = ax[i].imshow(to_plot[i], cmap=cmaps[i])
    #     ax[i].set_title(names[i])
    #     fig.colorbar(raster, ax=ax[i])
        
    fig.tight_layout()
    fig.savefig("rotation.png")
        
    
if __name__ == '__main__':
    flags.mark_flags_as_required(['config'])
    app.run(main)