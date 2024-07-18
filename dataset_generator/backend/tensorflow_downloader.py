import numpy as np
from scipy.ndimage import rotate

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
            train_ds['image'] = train_ds['image'][train_ds_filter]
            train_ds['label'] = train_ds['label'][train_ds_filter]
        #else:
        #    train_ds['image'] = train_ds['image']
            
        train_ds['image'] = train_ds['image'][:num_samples]
        train_ds['label'] = train_ds['label'][:num_samples]

        train_ds['image'] = np.squeeze(train_ds['image'])

        #train_ds['image'], angles = self.random_rotate_images(np.float32(train_ds['image']) / 255.0, 360)
        #return train_ds, angles
        
        train_ds['image'] = np.float32(train_ds['image']) / 255.0
        return train_ds
    
    def random_rotate_images(self, images, max_angle):
        """
        Rotates each image in a batch by a random angle between -max_angle and +max_angle.

        Args:
        - images: A numpy array of shape (batch_size, height, width) or (batch_size, height, width, channels).
        - max_angle: The maximum rotation angle in degrees.

        Returns:
        - A new numpy array of rotated images with the same shape as the input.
        """
        # Generate random rotation angles (in degrees) for each image in the batch
        angles = np.random.uniform(-max_angle, max_angle, size=len(images))

        # Rotate each image by its corresponding angle
        rotated_images = np.empty_like(images)
        for i in range(len(images)):
            rotated_images[i] = rotate(images[i], angles[i], reshape=False)

        return rotated_images, angles