from typing import Union, List, Tuple

import numpy as np

Vec = Union[
    List[float],
    List[int],
    Tuple[float, ...],
    Tuple[int, ...],
    np.ndarray,
]


def column_sum(vector: Vec):
    """
    returns the sum of all elements in a column vector
    """
    vector = np.atleast_2d(np.array(vector))
    if vector.shape[0] != 1:
        vector = vector.T
    return sum(vector[0])
