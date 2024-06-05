import sys
sys.path.append('/home/xchleb07/dev/dlbs')

from matplotlib import pyplot as plt
import utils
import ml_collections

import jax
import jax.numpy as jnp
from mpl_toolkits import axes_grid1

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


config = ml_collections.ConfigDict()

config.dataset_file_path = "/mnt/proj1/open-28-36/chlebik/datasets/1_7/128_16_(1.0, 3.0)_(1.0, 2.0)_1.0_6000_complex64_a20a468e0c30498ef6a52fabd69eb716.npz"
config.samples_per_epoch = 1000
config.train_ratio = 0.8
config.validation_ratio = 0.1
config.per_device_batch_size = 16
config.num_devices = jax.local_device_count()
config.use_cache = True
config.shuffle_buffer_size = -1
config.prefetch_num = jax.local_device_count()

prop_map = {  
    "sos": 0,
    "density": 1,
    "sources": 2,
    "pmls": 3,
    "full_fields": 4,
    "sos_fields": 5
}
# batch.shape == (6, n_devices, batch_size_per_device, img_size1, img_size2, *)
# t_batch: [sound_speed, density, source, pml, full_field, sos_field]

iterators, dataset_metadata = utils.InputPipeline('tensorflow').create_input_iter(config)

device = 0
sample = 0
for batch in iterators['train']['iter']:
    sos = batch[prop_map["sos"]][device][sample]
    density = batch[prop_map["density"]][device][sample]
    field = batch[prop_map["full_fields"]][device][sample]
    src = batch[prop_map["sources"]][device][sample]
    
    full_field = batch[prop_map["full_fields"]][device]
    sos_field = batch[prop_map["sos_fields"]][device]
    
    max_val = jnp.max(jnp.abs(full_field[sample]))
    #error_field = 100 * error_field / max_val

    mrpe = 100 * (jnp.abs(full_field[sample] - sos_field[sample]) / jnp.max(jnp.abs(full_field[sample])))
    mape = 100 * ((jnp.abs(full_field[sample] - sos_field[sample])) / ( jnp.abs(full_field[sample]) ) + 1e-8)
        
        
    plt.imshow(jnp.real(full_field[sample]), cmap="inferno" )
    plt.colorbar()
    plt.savefig("gt.png")
    plt.close()
    
    plt.imshow(jnp.real(sos_field[sample]), cmap="inferno" )
    plt.colorbar()
    plt.savefig("sos.png")
    plt.close()
    
    plt.imshow(mrpe, cmap="inferno" )
    plt.colorbar()
    plt.title("Relative Difference %")
    plt.tight_layout()
    plt.savefig("mrpe.png")
    plt.close()
    
    plt.imshow(mape, cmap="inferno" )
    plt.colorbar()
    plt.savefig("mape.png")
    plt.close()
    
    fig, ax = plt.subplots(1, 4)
    full_field_raster = ax[0].imshow(jnp.real(full_field[sample]), cmap="inferno" )
    sos_field_raster = ax[1].imshow(jnp.real(sos_field[sample]), cmap="inferno" )
    mrpe_field_raster = ax[2].imshow(mrpe, cmap="inferno" )
    mape_field_raster = ax[3].imshow(mape, cmap="inferno" )
    
    ax[0].set_title("GT")
    ax[1].set_title("SoS")
    ax[2].set_title("MRPE")
    ax[3].set_title("MAPE")
    
    add_colorbar(full_field_raster)
    add_colorbar(sos_field_raster)
    add_colorbar(mrpe_field_raster)
    add_colorbar(mape_field_raster)
    
    ax[0].axis("off")
    ax[1].axis("off")
    ax[2].axis("off")
    ax[3].axis("off")
    #fig.tight_layout()
    fig.savefig("errors.png")
    
    # fig, ax = plt.subplots(1, 3)
    # sos_raster = ax[0].imshow(sos, cmap="inferno")
    # density_raster = ax[1].imshow(density)
    # field_raster = ax[2].imshow(field.real)
    
    # ax[0].set_title("sos")
    # ax[1].set_title("rho")
    # ax[2].set_title("field")
    
    # fig.colorbar(sos_raster, ax=ax[0])
    # fig.colorbar(density_raster, ax=ax[1])
    # fig.colorbar(field_raster, ax=ax[2])
    # fig.tight_layout()
                
    # fig.savefig("medium.png")

#train_dts_jnp['densities'], train_dts_jnp['sources'], train_dts_jnp['pmls'], train_dts_jnp['full_fields']