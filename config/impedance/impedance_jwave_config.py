"""Default Hyperparameter configuration for training."""

import ml_collections
import jax

def get_config():
    """Get the training hyperparameter configuration."""

    config = ml_collections.ConfigDict()
    
    # Checkpoint options
    config.restore_from_checkpoint  = False # if true, restore from path provided bellow
    config.checkpoint_path          = ""    # a path to the checkpoint we wish to restore
    config.enable_checkpointing     = True  # checkpoint the training process throught the run
    config.checkpointing_warmup     = 10    # Number of epochs before checkpointing starts

    # Training options
    config.seed                     = 0                         # seed for initial model parameters and rng
    config.num_devices              = jax.local_device_count()  # gpus available for training
    config.per_device_batch_size    = 16                        # how many samples we wish to process per device per batch, global batch size is then product of this number and the number of devices available
    config.samples_per_epoch        = 4000                     # No need to adjust to exact batch size as the dataloader is set to drop remainders
    config.num_epochs               = 1000
    config.lr                       = 1e-3                  # learning rate
    config.loss_fun                 = "l2"                  # "l1" # "linf" # "l2" # "msre"
    config.target                   = "complex64"           # jax.numpy.complex64 #using string instead of numpy dtype makes for easy serialization
    config.clip_threshold           = 1.0
    config.validate_every_n_epochs  = 1                     # validate after every 'n' training epochs
    config.training_scale_factor    = 10                    # scale the training data by this factor
    
    # Architecture options
    config.model                    = "impedance_jwave"
    config.channels                 = 8
    config.stages                   = 2
    config.use_grid                 = False

    config.omega                    = 1.0
    config.amp                      = 10.0
    config.src_pos                  = (32, 32)
    
    # Dataset loading options
    #config.dataset_file_path        = "/mnt/proj1/open-28-36/chlebik/128_16_(1.0, 3.0)_(1.0, 2.0)_1.0_20000_complex64_3fef0e92ef34efac438cf60335020057.npz"
    config.train_ratio              = 0.8
    config.validation_ratio         = 0.2
    
    # TF settings
    config.prefetch_num             = jax.local_device_count()       # number of batches to prefetch while training... we are not using TF to train, so im not sure of the effect this setting has
    config.use_cache                = False  # as far as i understand the tensorflow data pipeline, cache is only relevant when the dataset does not fit into the RAM
    config.shuffle_buffer_size      = -1     # Shuffle the entire dataset, this is fine as long as the dataset fits into the RAM
    
    return config

def metrics():
    return []