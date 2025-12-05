import numpy as np
from numba import njit
from numba.core.types import boolean, int64, Array, uint8

from numbarrow.core.configurations import default_jit_options


@njit(boolean(int64, Array(uint8, 1, "C")), **default_jit_options)
def is_null(index_: int, bitmap: np.ndarray) -> bool:
    byte_for_index = bitmap[index_ // 8]
    bit_position_in_byte = index_ % 8
    return not (byte_for_index >> bit_position_in_byte) % 2
