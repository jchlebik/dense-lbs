import io

from clu import metric_writers
import jax.numpy as jnp
from matplotlib import pyplot as plt

from logger.field_plotting_settings import FieldPlottingSettings

class AcousticFieldsTensorboardLogger:
    
    def __init__(self, log_dir):
        self.writer = metric_writers.SummaryWriter(logdir=log_dir)
        self.io_buf = io.BytesIO()
        
    def log_acoustic_fields(self, step_n: int, fields: list[tuple[FieldPlottingSettings, jnp.ndarray]]):
        """
        Log acoustic fields to TensorBoard.

        Args:
            step_n (int): The step number.
            fields (list[tuple[FieldPlottingSettings, jnp.ndarray]]): A list of tuples containing the field plotting settings
                and the corresponding field data. Use prepare_acoustic_properties_for_logging from BaseLogger class to prepare the fields.

        Returns:
            None
        """
        #io_buf = io.BytesIO()
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
            self.io_buf.seek(0)
            fig.savefig(self.io_buf)
            self.io_buf.seek(0)
            image = plt.imread(self.io_buf)
            self.writer.write_images(step_n, {settings.logging_title: image})
        
        #io_buf.close()
        self.io_buf.seek(0)
        plt.close('all')
        
    def write_scalars(self, step: int, mapping):
        self.writer.write_scalars(step, mapping)
        
    def write_hparams(self, hparams):
        self.writer.write_hparams(hparams)
        
    def write_boxplot(self, step_n, data, labels):
        #io_buf = io.BytesIO()
        
        fig, ax = plt.subplots()
        ax.boxplot(data, labels = labels, notch = True, patch_artist = True, vert = True, showfliers=False)
        ax.set_title('Training and Validation Losses')
        #ax.set_xlabel('Loss')
        
        fig.tight_layout()
        self.io_buf.seek(0)
        fig.savefig(self.io_buf)
        self.io_buf.seek(0)
        
        image = plt.imread(self.io_buf)
        self.writer.write_images(step_n, {'Training and Validation Losses': image})
        self.io_buf.seek(0)
        plt.close('all')

        fig, ax = plt.subplots()
        ax.boxplot(data[1], labels = [labels[1]], patch_artist = True, vert = True, showfliers=False)
        ax.set_title('Validation Losses')
        fig.tight_layout()
        self.io_buf.seek(0)
        fig.savefig(self.io_buf)
        self.io_buf.seek(0)
        
        image = plt.imread(self.io_buf)
        self.writer.write_images(step_n, {'Validation Losses': image})
        self.io_buf.seek(0)
        plt.close('all')
        
        
    def write_histogram(self, step: int, arrays, bins):
        self.writer.write_histograms(step, arrays, bins)
        
    def flush(self):
        self.writer.flush()
    
    def close(self):
        self.writer.close()
        self.writer = None
        
    def __del__(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            
        if self.io_buf is not None:
            self.io_buf.close()
            self.io_buf = None