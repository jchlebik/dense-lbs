class MNISTDownloader:
    def __init__(self, backend: str):
        if backend not in ['torch', 'tf', 'pytorch', 'tensorflow']:
            raise ValueError(f"Invalid backend: {backend}")
        
        if backend == 'torch' or backend == 'pytorch':
            from .torch_downloader import PyTorch_MNISTDownloader
            self._backend = PyTorch_MNISTDownloader
        
        elif backend == 'tf' or backend == 'tensorflow':
            from .tensorflow_downloader import TensorFlow_MNISTDownloader
            self._backend = TensorFlow_MNISTDownloader
        
    def get(self, num_samples: int, labels_filter = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
        return self._backend().download(num_samples, labels_filter)
