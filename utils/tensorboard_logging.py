
import io
from matplotlib import pyplot as plt
from clu import metric_writers

import jax.numpy as jnp

class TensorboardSummaryWriter_Acoustics(metric_writers.SummaryWriter):
    def __init__(self, logdir: str):
        """
        Initializes a TensorboardSummaryWriter_Acoustics object.

        Args:
            logdir (str): The directory where the Tensorboard logs will be saved.
        """
        super().__init__(logdir)
              

    def log_acoustic_fields_tensorboard(self, step_n: int, sos, density, sos_field, true_field, pred_field, aux_fields: list[jnp.ndarray] = None, aux_field_names: list[str] = None):
        """
        Logs acoustic fields to Tensorboard.

        Args:
            step_n (int): The step number.
            sos: The speed of sound field.
            density: The density field.
            sos_field: The speed of sound field.
            true_field: The true field.
            pred_field: The predicted field.
            aux_fields (list[jnp.ndarray], optional): List of auxiliary fields. Defaults to None.
            aux_field_names (list[str], optional): List of names for the auxiliary fields. Defaults to None.
        """
        assert sos is not None and \
                density is not None and \
                true_field is not None and \
                pred_field is not None, "All fields must be provided."
        assert sos.shape == density.shape == true_field.shape == pred_field.shape, "All fields must have the same shape."
        assert aux_fields is None or len(aux_fields) == len(aux_field_names), "The number of aux fields must match the number of aux field names."

                
        io_buf = io.BytesIO()

        max_val = jnp.max(jnp.abs(true_field))

        mrpe = 100 * (jnp.abs(true_field - pred_field) / jnp.max(jnp.abs(true_field)))
        mape = 100 * ((jnp.abs(true_field - pred_field)) / ( jnp.abs(true_field) ) + 1e-8)
        
        to_plot = [sos, density, jnp.real(true_field), jnp.real(pred_field), jnp.real(sos_field), mrpe, mape]
        names = ["Speed of sound", "Density", "True Field", "Predicted Field", "SoS Field", "Relative Difference %", "Absolute Difference %"]
        cmaps = ["inferno", "inferno", "seismic", "seismic", "seismic", "inferno", "inferno"]
        limits = [(None, None), (None, None), (None, max_val), (None, max_val), (None, None), (None, None), (0, 50)]
        
        if aux_fields is not None:
            for i in range(len(aux_fields)):
                aux_field = aux_fields[i]
                aux_field_name = aux_field_names[i]
                to_plot.append(jnp.real(aux_field))
                names.append(aux_field_name)
                cmaps.append("inferno")
                limits.append((None, None))

        for i in range(len(to_plot)):
            fig, ax = plt.subplots()
            raster = ax.imshow(to_plot[i], cmap=cmaps[i], vmin=limits[i][0], vmax=limits[i][1])
            ax.set_title(names[i])
            fig.colorbar(raster, ax=ax)
            fig.tight_layout()
            io_buf.seek(0)
            fig.savefig(io_buf)
            io_buf.seek(0)
            image = plt.imread(io_buf)
            self.write_images(step_n, {names[i]: image})
        
        io_buf.close()
        plt.close('all')

