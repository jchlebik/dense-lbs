from itertools import islice
from typing import Any, Iterable

def advance_iterator(it: Iterable[Any], checkpointed_step: int) -> None:
    """
    Advances the given iterator to the specified checkpointed step.

    Parameters:
    it (Iterable[Any]): The iterator to be advanced.
    checkpointed_step (int): The step to which the iterator should be advanced.

    Returns:
    Iterator: The advanced iterator.
    """
    next(islice(it, checkpointed_step, checkpointed_step), None)
    #return it