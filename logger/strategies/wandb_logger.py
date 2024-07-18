import io

import wandb
import jax.numpy as jnp
from matplotlib import pyplot as plt

from logger.field_plotting_settings import FieldPlottingSettings

class WandbLogger:
    '''
    Partial implementation, NOT TESTED AND NOT READY FOR USE
    '''
    
    def __init__(self, run):
        wandb.init(run)
        run_name = wandb.run.name

    def write_hparams(self, hparams):
        wandb.config.update(hparams)

    def log_acoustic_fields(self, step_n: int, fields: list[tuple[FieldPlottingSettings, jnp.ndarray]]):
        for (settings, field) in fields:
            fig, ax = plt.subplots()
            raster = ax.imshow(
                jnp.real(field) if settings.is_complex else field, 
                cmap=settings.cmap,
                vmin=settings.vmin,
                vmax=settings.vmax
            )
            ax.set_title(settings.logging_title)
            fig.colorbar(raster, ax=ax)
            fig.tight_layout()
            
            #self.run.log({settings.logging_title: fig})

        img = wandb.Image(plt)
        wandb.log({"Acoustic Fields": img}, step=step_n)
    
        plt.close('all')

    def write_scalars(self, step: int, mapping):
        wandb.log(mapping, step=step)
        
    def write_histogram(self, step: int, arrays, bins):
        pass

    def flush(self):
        pass

    def close(self):
        pass

