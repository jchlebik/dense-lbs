"""Default Hyperparameter configuration for training."""

import ml_collections
import jax

def get_config():
    """Get the training hyperparameter configuration."""

    config = ml_collections.ConfigDict()
    
    # config.use_checkpoint = False           # if true, we will use a checkpoint to restore the model and continue training
    # if config.use_checkpoint:
    #     config.checkpoint_path          = "/mnt/proj1/open-28-36/chlebik/barbora/runlogs/lbslinf/2024-04-22_11-40-23/checkpoints/" #/mnt/proj1/open-28-36/chlebik/barbora/runlogs/lbs/2024-04-21_18-04-01/checkpoints/"    # a path to the checkpoint we wish to restore
    #     config.restart_from_checkpoint  = True # if true, restart from the checkpoint provided above, if false, continue training from the checkpoint provided above
    
    config.enable_checkpointing     = True  # checkpoint the training process throught the run
    config.checkpointing_warmup     = 10    # Number of epochs before checkpointing starts

    # Training options
    config.seed                     = 9846351564531          # seed for initial model parameters and rng
    config.num_devices              = jax.local_device_count()  # gpus available for training
    config.per_device_batch_size    = 16                        # how many samples we wish to process per device per batch, global batch size is then product of this number and the number of devices available
    config.samples_per_epoch        = 6000                     # No need to adjust to exact batch size as the dataloader is set to drop remainders
    config.num_epochs               = 100000
    config.lr                       = 1e-3
    config.loss_fun                 = "linf"                # "l1" # "linf"
    config.target                   = "complex64"           # jax.numpy.complex64 #using string instead of numpy dtype makes for easy serialization
    config.clip_threshold           = 1.0
    config.validate_every_n_epochs  = 5                     # validate after every 'n' training epochs
    config.training_scale_factor    = 10                    # scale the training data by this factor
    
    # Architecture options
    config.model                    = "dlbs"
    config.channels                 = 8
    config.last_projection_channels = 128
    config.use_nonlinearity         = True
    config.use_grid                 = True
    config.stages                   = 1
    
    # Dataset loading options
    config.train_ratio              = 0.8
    config.validation_ratio         = 0.1
    
    # TF settings
    config.prefetch_num             = jax.local_device_count()       # number of batches to prefetch while training... we are not using TF to train, so im not sure of the effect this setting has
    config.use_cache                = False  # as far as i understand the tensorflow data pipeline, cache is only relevant when the dataset does not fit into the RAM
    config.shuffle_buffer_size      = -1     # Shuffle the entire dataset, this is fine as long as the dataset fits into the RAM
    
    return config

def metrics():
    return []