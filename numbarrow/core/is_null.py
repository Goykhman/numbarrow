"""
Null detection for Apache Arrow validity bitmaps.

Arrow uses a packed bitmap to track which elements in an array are valid (non-null).
Each bit corresponds to one element: bit=1 means valid, bit=0 means null.
Bits are packed LSB-first into uint8 bytes — element *i* lives at byte ``i // 8``,
bit position ``i % 8`` within that byte.
"""

import numpy as np
from numba import njit
from numba.core.types import boolean, int64, Array, uint8

from numbarrow.core.configurations import default_jit_options


@njit(boolean(int64, Array(uint8, 1, "C")), **default_jit_options)
def is_null(index_: int, bitmap: np.ndarray) -> bool:
    """Check whether element *index_* is null according to *bitmap*.

    Arrow validity bitmaps store one bit per element, packed LSB-first into
    uint8 bytes. A set bit (1) means valid; a cleared bit (0) means null.

    :param index_: zero-based element index
    :param bitmap: uint8 array containing the packed validity bitmap
    :returns: True if the element is null (bit is 0), False if valid (bit is 1)
    """
    # Locate the byte containing the bit for this element
    byte_for_index = bitmap[index_ // 8]
    # Isolate the specific bit within that byte (LSB-first order)
    bit_position_in_byte = index_ % 8
    return not (byte_for_index >> bit_position_in_byte) % 2
