import os
from absl import flags, app
from ml_collections import config_flags

#from trainers.train import Trainer
from trainers.default_trainer import DefaultTrainer
from trainers.default_tester import DefaultTester

#os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = ".95"

flags.DEFINE_string("workdir", None, "Directory to store model data.")
flags.DEFINE_string("dataset_file_path", None, "Path to the .npz MNISTHelmholtz generated file.")
flags.DEFINE_boolean("use_checkpoint", False, "Whether to use a checkpoint.")
flags.DEFINE_string("checkpoint_file_path", None, "Path to the optax checkpoint.")
flags.DEFINE_boolean('restart_from_checkpoint', False, 'Whether to restart from the checkpoint. \
                                                        If true, restart from the checkpoint provided using new config \
                                                        If false, continue training from the checkpoint using original config')
flags.DEFINE_boolean("is_test", False, "Whether to run the model in test mode.")

config_flags.DEFINE_config_file("config", None, "File path to the training hyperparameter configuration.", lock_config=False,)


def handle_chkpt_dependencies(flags):
    """
    Handles the dependencies related to checkpoints.

    Args:
        flags: An object containing the flags and configurations.

    Raises:
        ValueError: If `use_checkpoint` is True but `checkpoint_file_path` is not set.
        ValueError: If `restart_from_checkpoint` is True but `use_checkpoint` is not True.
    """
    if flags.is_test:
        if flags.checkpoint_file_path is None:
            raise ValueError("If is_test is True, checkpoint_file_path must be set.")
        flags.config.checkpoint_path = os.path.abspath(flags.checkpoint_file_path)
        
    elif flags.use_checkpoint:
        if flags.checkpoint_file_path is None:
            raise ValueError("If use_checkpoint is True, checkpoint_file_path must be set.")            

        flags.config.use_checkpoint = True
        flags.config.checkpoint_path = os.path.abspath(flags.checkpoint_file_path)
        flags.config.restart_from_checkpoint = flags.restart_from_checkpoint
    
    elif flags.restart_from_checkpoint and not flags.use_checkpoint:
        raise ValueError("If restart_from_checkpoint is True, use_checkpoint must be True.")

def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    FLAGS = flags.FLAGS
    FLAGS.config.workdir = os.path.abspath(FLAGS.workdir)
    FLAGS.config.dataset_file_path = os.path.abspath(FLAGS.dataset_file_path)
    
    handle_chkpt_dependencies(FLAGS)
    
    if FLAGS.is_test:
        _ = DefaultTester(FLAGS.config.model).start_testing(FLAGS.config)
    else:
        _ = DefaultTrainer(FLAGS.config.model).start_training(FLAGS.config)
        
    return 0

if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir", "dataset_file_path"])
    app.run(main)
