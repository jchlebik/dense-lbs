class InputPipeline:
    def __init__(self, implementation: str):
        if implementation not in ['torch', 'tensorflow']:
            raise ValueError(f"Invalid implementation: {implementation}")
        
        if implementation == 'torch':
            from .input_pipeline_torch import InputPipeline_Torch
            self._strategy = InputPipeline_Torch()
        elif implementation == 'tensorflow':
            from .input_pipeline_tensorflow import InputPipeline_Tensorflow
            self._strategy = InputPipeline_Tensorflow()
        
    def create_input_iter(self, config, keys = None):
        return self._strategy.create_input_iter(config, keys)