import ml_collections
import jax
import time
import tqdm
import sys
sys.path.append('/home/xchleb07/dev/dlbs')

import utils

num_tests = 50

@jax.pmap
def foo(sos, rho, src, pml, gt):
    key = jax.random.key(0)
    y = jax.numpy.multiply(sos, rho)
    y = jax.numpy.multiply(y, src)
    y = jax.numpy.multiply(y, gt)
    s = jax.numpy.sum(y)
    
    x = jax.random.normal(key, (1000000, 1000000)) * s
    return s

def benchmark_torch():    
    config = ml_collections.ConfigDict()
    config.seed                     = 123456789                     # seed for initial model parameters and rng
    config.num_devices              = jax.local_device_count()      # gpus available for training
    config.per_device_batch_size    = 16                            # how many samples we wish to process per device per batch, global batch size is then product of this number and the number of devices available
    config.samples_per_epoch        = 10000                         # No need to adjust to exact batch size as the dataloader is dropping remainders
    config.dataset_file_path        = "/mnt/proj1/open-28-36/chlebik/128_16_(1.0, 3.0)_(1.0, 2.0)_1.0_20000_complex64_3fef0e92ef34efac438cf60335020057.npz"
    config.train_ratio              = 0.8
    config.validation_ratio         = 0.1
    config = ml_collections.FrozenConfigDict(config)
    
    iterators, dataset_metadata = utils.InputPipeline('torch').create_input_iter(config)
    
    train_it = iterators["train"]["iter"]
    
    num_batches = config.samples_per_epoch // (config.per_device_batch_size * config.num_devices)
    
    times = []
    print("Starting benchmark")
    for test_i in tqdm.trange(0, num_tests):
        t0 = time.time()
        for step, batch in zip(range(0, num_batches), train_it):
            out = foo(batch[0], batch[1], batch[2], batch[3], batch[4])
            out.block_until_ready()
        t1 = time.time()

        times.append(t1-t0)
        
    elapsed_time = sum(times) / len(times)
    
    print('Execution time:', elapsed_time, 'seconds')

def benchmark_tensorflow():
    config = ml_collections.ConfigDict()
    config.seed                     = 123456789                     # seed for initial model parameters and rng
    config.num_devices              = jax.local_device_count()      # gpus available for training
    config.per_device_batch_size    = 16                            # how many samples we wish to process per device per batch, global batch size is then product of this number and the number of devices available
    config.samples_per_epoch        = 10000                         # No need to adjust to exact batch size as the dataloader is dropping remainders
    config.dataset_file_path        = "/mnt/proj1/open-28-36/chlebik/128_16_(1.0, 3.0)_(1.0, 2.0)_1.0_20000_complex64_3fef0e92ef34efac438cf60335020057.npz"
    config.train_ratio              = 0.8
    config.validation_ratio         = 0.1
    config.use_cache                = False
    config.shuffle_buffer_size      = -1
    config.prefetch_num             = jax.local_device_count()
    config = ml_collections.FrozenConfigDict(config)
    
    iterators, dataset_metadata = utils.InputPipeline('tensorflow').create_input_iter(config)
    
    train_it = iterators["train"]["iter"]
    
    num_batches = config.samples_per_epoch // (config.per_device_batch_size * config.num_devices)
    

    times = []
    print("Starting benchmark")
    for test_i in tqdm.trange(0, num_tests):
        t0 = time.time()
        for step, batch in zip(range(0, num_batches), train_it):
            out = foo(batch[0], batch[1], batch[2], batch[3], batch[4])
            out.block_until_ready()
        t1 = time.time()

        times.append(t1-t0)
        
    elapsed_time = sum(times) / len(times)
    
    print('Execution time:', elapsed_time, 'seconds')
    
if __name__ == '__main__':
    try:
        benchmark_tensorflow()
    except Exception as e:
        benchmark_torch()