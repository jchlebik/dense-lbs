from utils.key_mapper import KeyMapper as km

class InputPipeline:
    def __init__(self, implementation: str):
        """
        Initializes an InputPipeline object.

        Args:
            implementation (str): The implementation to use for the input pipeline. Must be either 'torch' or 'tensorflow'.
        
        Raises:
            ValueError: If the implementation is not 'torch' or 'tensorflow'.
        """
        if implementation not in ['torch', 'tensorflow']:
            raise ValueError(f"Invalid implementation: {implementation}")
        
        if implementation == 'torch':
            from input_pipeline.backend.input_pipeline_torch import InputPipeline_Torch
            self._strategy = InputPipeline_Torch()
        elif implementation == 'tensorflow':
            from input_pipeline.backend.input_pipeline_tensorflow import InputPipeline_Tensorflow
            self._strategy = InputPipeline_Tensorflow()
        
    def create_input_iter(self, config, keys=None):
            """
            Creates an input iterator for the given configuration and keys.

            Args:
                config (dict): The configuration for creating the input iterator.
                keys (list, optional): The keys to include in the input iterator. If None, all batch keys will be used.

            Returns:
                The input iterator.
            """
            dts_keys = km.get_dataset_to_model_mapping()
            if keys is None:
                keys = km.get_batch_keys()
            ground_truths_key = dts_keys[km.get_full_field_key()]
            if ground_truths_key not in keys:
                keys.append(ground_truths_key)
                
            return self._strategy.create_input_iter(config, keys)
