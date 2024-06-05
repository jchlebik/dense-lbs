import numpy as np

import tensorflow_datasets as tfds
from tensorflow_datasets.core.utils import gcs_utils
gcs_utils._is_gcs_disabled = True   #we do not need google auth token for mnist

class TensorFlow_MNISTDownloader:
    def download(self, num_samples: int, labels_filter = None):
        """
        Downloads or loads the MNIST images.

        Returns:
        - The downloaded or loaded MNIST images.
        """
        # Download or load the MNIST images
        print("Getting MNIST images using TensorFlow")
        mnist = tfds.builder("mnist")
        mnist.download_and_prepare(download_dir="/tmp/tfds/mnist/")
        train_ds = tfds.as_numpy(mnist.as_dataset(split="train", batch_size=-1, shuffle_files=True))
        #train_ds_filter = jnp.where(train_ds["label"] in (0, ), 1, 0)
        if labels_filter is not None:
            train_ds_filter = np.isin(train_ds["label"], np.array(labels_filter))
            train_ds = train_ds['image'][train_ds_filter]
        else:
            train_ds = train_ds['image']
            
        train_ds = train_ds[:num_samples]

        mnist_images = np.squeeze(train_ds)

        mnist_images = np.float32(mnist_images) / 255.0
        return mnist_images