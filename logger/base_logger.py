from typing import Any
import jax.numpy as jnp

from utils.key_mapper import KeyMapper as km
from logger.field_plotting_settings import FieldPlottingSettings

class BaseLogger:
    """
    Base class for logging
    """
    def __init__(self, backend: str):
        """
        Initialize the logger

        Args:
            backend (str): The backend for the logger
        """
        assert backend in ["tensorboard", "wandb"], f"Backend {backend} is not supported"
        if backend == "tensorboard":
            from .strategies.tensorboard_logger import AcousticFieldsTensorboardLogger
            self.backend = AcousticFieldsTensorboardLogger
        elif backend == "wandb":
            from .strategies.wandb_logger import AcousticFieldsWandbLogger
            self.backend = AcousticFieldsWandbLogger
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.writter = self.backend(*args, **kwds)
        return self

    def log_acoustic_fields(self, *args, **kwargs):
        """
        Logs the acoustic fields.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.writter.log_acoustic_fields(*args, **kwargs)
        
    def write_scalars(self, *args, **kwargs):
        self.writter.write_scalars(*args, **kwargs)
        
    def write_histogram(self, **kwargs):
        self.writter.write_histogram(**kwargs)
        
    def write_hparams(self, *args, **kwargs):
        """
        Writes the hyperparameters to the logger.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.writter.write_hparams(*args, **kwargs)
        
    def write_boxplot(self, *args, **kwargs):
        """
        Writes a bar plot to the logger.

        Returns:
            None
        """
        self.writter.write_boxplot(*args, **kwargs)
        
    def flush(self):
        """
        Flushes the writer to ensure that all buffered data is written to the underlying stream.
        """
        self.writter.flush()
        
    def close(self):
        """
        Closes the logger and releases any resources used.
        """
        self.writter.close()
    
    @staticmethod
    def prepare_sample_comparison_for_logging(predicted_field, true_field, sample_description = "", prefix=""):
        """
        Prepares the acoustic properties for logging.

        Args:
            predicted_field (jnp.ndarray): The predicted field.
            true_field (jnp.ndarray): The true field.

        Returns:
            jnp.ndarray: The prediction delta.
        """
        prediction_delta = jnp.abs(true_field - predicted_field)
        fields_to_log = []
        fields_to_log.append((
            FieldPlottingSettings(
                "inferno",
                None, None,
                f"{prefix}Mean Relative Percentage Error {sample_description}",
                False
            ),
            100 * (prediction_delta / jnp.max(jnp.abs(true_field)) + 1e-8)
        ))
        fields_to_log.append((
            FieldPlottingSettings(
                "inferno",
                0, 100,
                f"{prefix}Mean Absolute Percentage Error {sample_description}",
                False
            ),
            100 * (prediction_delta / (jnp.abs(true_field)) + 1e-8)
        ))
        return fields_to_log
        
    
    @staticmethod
    def prepare_acoustic_properties_for_logging(batch_to_log, 
                                                predicted_field_batch, 
                                                property_index_map, 
                                                model_keys,
                                                skip_keys_substr=["pml", "source"],
                                                device=0,
                                                sample=0):
        """
        Prepares the acoustic properties for logging.

        Args:
            batch_to_log (list): The batch of data to log.
            predicted_field_batch (list): The batch of predicted fields.
            property_index_map (dict): A mapping of property keys to indices.
            model_keys (list): The list of model keys.
            skip_keys_substr (list, optional): A list of substrings to skip when processing model keys. Defaults to ["pml", "source"].

        Returns:
            list: A list of tuples containing the field plotting settings and corresponding data to log.
        """

        assert property_index_map is not None, f"Property index map is None"
        assert batch_to_log is not None, f"Batch to log is None"
        
        fields_to_log = []
        predicted_field = None
        true_field = None
        
        if predicted_field_batch is not None and batch_to_log is not None:
            predicted_field = predicted_field_batch[device][sample]
            fields_to_log.append((            
                FieldPlottingSettings(
                    "seismic",
                    None, None,
                    "predicted_fields",
                    jnp.iscomplexobj(predicted_field)
                ),
                predicted_field 
            ))
            
        if km.get_full_field_key() in property_index_map:            
            true_field = batch_to_log[property_index_map[km.get_full_field_key()]][device][sample]
            fields_to_log.append((
                FieldPlottingSettings(
                    "seismic",
                    None, None,
                    "true_fields",
                    jnp.iscomplexobj(true_field)
                ),
                true_field
            ))
            
        if predicted_field is not None and true_field is not None:
            prediction_delta = jnp.abs(true_field - predicted_field)
            #value_diff = jnp.abs(fields_to_log["true_fields"] - fields_to_log["predicted_fields"])

            fields_to_log.append((
                FieldPlottingSettings(
                    "inferno",
                    None, None,
                    "prediction_delta",
                    False
                ),
                prediction_delta
            ))

            fields_to_log.append((
                FieldPlottingSettings(
                    "inferno",
                    None, None,
                    "mrpe",
                    False
                ),
                100 * (prediction_delta / jnp.max(jnp.abs(true_field)) + 1e-8)
            ))

            fields_to_log.append((
                FieldPlottingSettings(
                    "inferno",
                    0, 100,
                    "mape",
                    False
                ),
                100 * (prediction_delta / (jnp.abs(true_field)) + 1e-8)
            ))
            
        model_to_dts_mapping = km.get_model_to_dataset_mapping()
        #skip_keys = ["pml", "source", "src"]
        for key in model_keys:
            if any(forbiden_key in key for forbiden_key in skip_keys_substr):
                continue
            dts_key = model_to_dts_mapping[key]
            data = batch_to_log[property_index_map[dts_key]][device][sample]
            is_complex = jnp.iscomplexobj(data)
            fields_to_log.append((
                FieldPlottingSettings(
                    "seismic" if is_complex else "inferno",
                    None, None,
                    dts_key,
                    is_complex
                ),
                data
            ))
            
        return fields_to_log