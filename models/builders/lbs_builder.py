import optax
from models.acoustic_model import AcousticModel
from models.builders.builder_base import BuilderBase

class LbsBuilder(BuilderBase):
    """
    A builder class for creating model and optimizers for the Learned Born Series.
    """

    def __init__(self):
        super().__init__()

    def create_model(self, config):
        """
        Create the LBS model with the specified configuration.

        Args:
            config (Config): The configuration object containing the model parameters.

        Returns:
            AcousticModel: The created acoustic model.

        """
        return AcousticModel("lbs")(
            stages=config.stages,
            channels=config.channels,
            dtype=config.target,
            last_proj=config.last_projection_channels,
            use_nonlinearity=config.use_nonlinearity,
            use_grid=config.use_grid,
        )
    
    def create_optimizer(self, config):
        """
        Creates an optimizer for the LBS.

        Args:
            config (dict): A dictionary containing the configuration parameters for the optimizer.

        Returns:
            optax.GradientTransformation: The created optimizer.

        """
        return optax.chain(
            optax.adaptive_grad_clip(config.clip_threshold),
            optax.adamw(learning_rate=config.lr),
        )
    
