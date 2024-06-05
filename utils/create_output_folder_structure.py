from datetime import datetime
import os

def create_output_folder_structure(workdir):
    """
    Creates the folder structure for output files.

    Args:
        workdir (str): The base directory where the output folder structure will be created.

    Returns:
        tuple: A tuple containing the paths to the output directory, checkpoint directory, and tensorboard directory.
    """
    output_dir = os.path.join(os.path.abspath(workdir), datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    tensorboard_dir = os.path.join(output_dir, 'tensorboard')
    os.makedirs(tensorboard_dir, exist_ok=True)

    return output_dir, checkpoint_dir, tensorboard_dir
