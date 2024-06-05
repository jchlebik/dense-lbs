from typing import Any


class AcousticModel:
    def __init__(self, model: str):
        if model not in ['lbs', 'dlbs', 'corrector', 'fno']:
            raise ValueError(f"Invalid model: {model}")
        
        if model == 'lbs':
            from models.strategies.bno import BNO
            self._strategy = BNO
        
        elif model == 'dlbs':
            from models.strategies.dbno import DBNO
            self._strategy = DBNO
        
        elif model == 'corrector':
            from models.strategies.density_corrector import Density_Corrector
            self._strategy = Density_Corrector

        elif model == 'fno':
            from models.strategies.fno import FNO2D
            self._strategy = FNO2D
        
    def __call__(self, *args, **kwargs) -> Any:
         return self._strategy(*args, **kwargs)
