from jax import tree_util
from flax.training.train_state import TrainState
import ml_collections
import orbax.checkpoint

class CheckpointManager:
    #checkpoint_manager: orbax.checkpoint.CheckpointManager = None
    #https://github.com/google/flax/issues/3708 #deprecation warnign about SaveArgs.aggregate should not be of concern
    def __init__(self, checkpoint_dir: str) -> None:
        self.checkpoint_manager = orbax.checkpoint.CheckpointManager(
            checkpoint_dir,     
            item_handlers={
                "model": orbax.checkpoint.PyTreeCheckpointHandler(), 
                "config": orbax.checkpoint.JsonCheckpointHandler(), 
                "epoch": orbax.checkpoint.StandardCheckpointHandler(),
                "loss": orbax.checkpoint.StandardCheckpointHandler()
            },  
            options=orbax.checkpoint.CheckpointManagerOptions(max_to_keep=3, create=True, best_fn=lambda metrics: metrics['loss'], best_mode='min')
        )
    
    def get_checkpoint_manager(self) -> orbax.checkpoint.CheckpointManager:
        """
        Gets the checkpoint manager.

        Returns:
            orbax.checkpoint.CheckpointManager: The checkpoint manager.
        """
        return self.checkpoint_manager

    def save_checkpoint(self, t_state: TrainState, config: dict, step: int, loss: float):
        """
        Saves a checkpoint of the training state.

        Args:
            t_state (TrainState): The training state to be saved.
            config (ml_collections.FrozenConfigDict): The configuration dictionary.
            step (int): The current step of the training.

        Returns:
            str: The path to the saved checkpoint file.
        """
        return self.checkpoint_manager.save(
            step, 
            args=orbax.checkpoint.args.Composite(
                config=orbax.checkpoint.args.JsonSave(config),
                model=orbax.checkpoint.args.PyTreeSave(t_state),
                epoch=orbax.checkpoint.args.StandardSave(step),
                loss = orbax.checkpoint.args.StandardSave(loss)
            ),
            metrics={'loss': loss}
        )
    
    def restore_from_checkpoint(self, t_state: TrainState) -> tuple[TrainState, ml_collections.ConfigDict, int]:
        """
        Restores the training state from the latest checkpoint.

        Args:
            t_state (TrainState): The training state to be restored.

        Returns:
            tuple: A tuple containing the restored training state, configuration dictionary, and epoch.
        
        """
        restored_dict = self.checkpoint_manager.restore(
            self.checkpoint_manager.latest_step(),
        )

        restored_optimizer = tree_util.tree_unflatten(tree_util.tree_structure(t_state.opt_state), tree_util.tree_leaves(restored_dict['model']['opt_state']))

        t_state = t_state.replace(params=restored_dict['model']["params"], 
                                step=restored_dict['model']["step"], 
                                opt_state=restored_optimizer)  
    
        return t_state, ml_collections.ConfigDict(restored_dict['config']), restored_dict['epoch']

    def __del__(self):
        self.checkpoint_manager.close()
        self.checkpoint_manager = None