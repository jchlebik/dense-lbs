import optax
from models.acoustic_model import AcousticModel
from trainers.model_builders.builder_base import BuilderBase

class DlbsBuilder(BuilderBase):                        
    """
    A builder class for creating model and optimizers for Dense Learned Born Series.
    """

    def __init__(self):
        super().__init__()

    def create_model(self, config):
        """
        Create an instance of the DLBS model with the specified configuration.

        Args:
            config (Config): The configuration object containing the model parameters.

        Returns:
            AcousticModel: An instance of the AcousticModel class.

        """
        return AcousticModel("dlbs")(
            stages=config.stages,
            channels=config.channels,
            dtype=config.target,
            last_proj=config.last_projection_channels,
            use_nonlinearity=config.use_nonlinearity,
            use_grid=config.use_grid,
        )
    
    def create_optimizer(self, config):
        """
        Creates an optimizer for the DLBS model.

        Args:
            config: A configuration object containing optimizer parameters.

        Returns:
            An instance of the optax optimizer.
        """
        return optax.chain(
            optax.adaptive_grad_clip(config.clip_threshold),
            optax.adamw(learning_rate=config.lr),
        )