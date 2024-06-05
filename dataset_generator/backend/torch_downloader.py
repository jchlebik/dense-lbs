import jax
import jax.numpy as jnp

from torch.utils.data import Dataset
from torchvision.datasets import MNIST


class PyTorch_MNISTDownloader:    
    def _download_mnist(self, num_samples: int, labels_filter = None):
        """
        Downloads or loads the MNIST images.

        Returns:
        - The downloaded or loaded MNIST images.
        """
        # Download or load the MNIST images
        print("Getting MNIST images")
        # Load MNIST dataset
        mnist_dataset = MNIST("/tmp/chlebik/data/mnist_images_original", download=True)
        
        # Convert images to NumPy array
        mnist_images = mnist_dataset.data.numpy().astype(jnp.float32)
        if labels_filter is None:
            return jnp.array(mnist_images / 255.0)
        
        # Filter out the images based on the labels
        mnist_labels = mnist_dataset.targets.numpy()

        train_ds_filter = jnp.isin(mnist_labels["labels"], labels_filter)
        mnist_images = mnist_images[train_ds_filter]

        mnist_images = mnist_images / 255.0
        return jnp.array(mnist_images)