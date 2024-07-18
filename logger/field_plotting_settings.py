from dataclasses import dataclass

@dataclass
class FieldPlottingSettings:
    cmap = ""
    vmin = None
    vmax = None
    logging_title = ""
    is_complex = False

    def __init__(
        self,
        cmap: str,
        vmin: int,
        vmax: int,
        logging_title: str,
        is_complex: bool
    ):
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.logging_title = logging_title
        self.is_complex = is_complex