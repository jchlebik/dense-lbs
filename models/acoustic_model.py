from typing import Any


class AcousticModel:
    def __init__(self, model: str):
        if model not in ['lbs', 'dlbs', 'corrector', 'fno']:
            raise ValueError(f"Invalid model: {model}")
        
        if model == 'lbs':
            from models.models.bno import BNO
            self._strategy = BNO
        
        elif model == 'dlbs':
            from models.models.dbno import DBNO
            self._strategy = DBNO
        
        elif model == 'corrector':
            from models.models.density_corrector import Density_Corrector
            self._strategy = Density_Corrector

        elif model == 'fno':
            from models.models.fno import FNO2D
            self._strategy = FNO2D
        
    def __call__(self, *args, **kwargs) -> Any:
         return self._strategy(*args, **kwargs)
