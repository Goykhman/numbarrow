"""
Null detection for Apache Arrow validity bitmaps.

Arrow uses a packed bitmap to track which elements in an array are valid (non-null).
Each bit corresponds to one element: bit=1 means valid, bit=0 means null.
Bits are packed LSB-first into uint8 bytes — element *i* lives at byte ``i // 8``,
bit position ``i % 8`` within that byte.
"""

import numpy as np
from numba import njit
from numba.core.types import boolean, int64, Array, uint8, bool_

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


@njit(Array(bool_, 1, "C")(int64, int64, Array(uint8, 1, "C")), **default_jit_options)
def unpack_booleans(offset: int, length: int, packed_data: np.ndarray) -> np.ndarray:
    """Unpack bit-packed boolean data into a boolean array.

    :param offset: bit offset into packed_data to start reading
    :param length: number of boolean values to extract
    :param packed_data: uint8 array containing LSB-first packed bits
    :returns: boolean array of *length* elements
    """
    result = np.empty(length, dtype=np.bool_)
    for i in range(length):
        byte_idx = (offset + i) // 8
        bit_idx = (offset + i) % 8
        result[i] = (packed_data[byte_idx] >> bit_idx) & 1
    return result


# Two-layer struct nullability inspired by Awkward Array's BitMaskedArray(RecordArray)
# design. See: https://awkward-array.org/doc/main/reference/generated/ak.contents.BitMaskedArray.html


@njit(**default_jit_options)
def is_null_struct(index_, struct_bitmap, field_bitmap):
    """Check whether a struct field value is null at either the struct or field layer.

    Arrow StructArrays carry a validity bitmap for the struct itself (is this
    entire row null?) independent of each child field's bitmap (is this
    particular field null within a non-null row?).  A value is null if either
    layer marks it as null.

    :param index_: zero-based element index
    :param struct_bitmap: uint8 packed bitmap for struct-level validity, or None
    :param field_bitmap: uint8 packed bitmap for field-level validity, or None
    :returns: True if null at either layer
    """
    if struct_bitmap is not None and is_null(index_, struct_bitmap):
        return True
    if field_bitmap is not None and is_null(index_, field_bitmap):
        return True
    return False
