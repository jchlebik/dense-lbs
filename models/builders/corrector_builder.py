import optax
from models.acoustic_model import AcousticModel
from models.builders.builder_base import BuilderBase

class CorrectorBuilder(BuilderBase):                        
    """
    A class that builds a corrector model and optimizer.
    """

    def __init__(self):
        super().__init__()
    
    def create_model(self, config):
        """
        Creates an instance of the AcousticModel for the corrector.

        Args:
            config: The configuration object containing model parameters.

        Returns:
            An instance of the AcousticModel for the corrector.
        """
        return AcousticModel("corrector")(
            stages=config.stages,
            channels=config.channels,
            dtype=config.target,
            last_proj=0,
            use_nonlinearity=False,
            use_grid=config.use_grid,
        )
    
    def create_optimizer(self, config):
        """
        Creates an optimizer for training the corrector model.

        Args:
            config: The configuration object containing optimizer parameters.

        Returns:
            An instance of the optax optimizer.
        """
        return optax.chain(
            optax.adaptive_grad_clip(config.clip_threshold),
            optax.adamw(learning_rate=config.lr),
        )