from typing import Any
from flax.training import train_state

class TrainState(train_state.TrainState):
    batch_stats: Any
