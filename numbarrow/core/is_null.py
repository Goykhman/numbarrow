"""Functions, compiled with numba, that read Arrow validity bitmaps and bit-packed boolean values.

An Arrow validity bitmap holds one bit for each element of an array: 1 if the element is valid, 0 if it is null. The
bits are packed into uint8 bytes from the least significant bit up, so element ``i`` is bit ``i % 8`` of byte
``i // 8``. :func:`is_null` reads the bit of one element, and :func:`is_null_struct` reads the two bitmaps that
together decide whether a value of a struct's field is null. Arrow packs the values of a boolean array in the same
way, and :func:`unpack_booleans` expands them into a numpy boolean array. numba compiles all three functions when this
module is imported, with the options of :mod:`numbarrow.core.configurations`. Python code and jitted code can both
call them.
"""

import numpy as np
from numba.core.types import boolean, int64, Array, Optional, uint8, bool_

from numbarrow.core.configurations import jit_with_options


@jit_with_options(boolean(int64, Array(uint8, 1, "C", readonly=True)))
def is_null(index_: int, bitmap: np.ndarray) -> bool:
    """Return True if element ``index_`` is null in the validity bitmap ``bitmap``, and False if it is valid.

    ``bitmap`` must be a one-dimensional, C-contiguous uint8 array, read-only or writable, as the adapters return it.
    Called from Python with anything else, None included, the function raises ``TypeError``. An adapter returns None
    for an array without a validity buffer, whose elements are all valid, so test for None before the call.

    ``index_`` must satisfy ``0 <= index_ < 8 * len(bitmap)``. numba compiles without bounds checks by default, so an
    index past the end reads memory outside the bitmap: the answer is meaningless, and the read can crash the process.
    With bounds checks, the function raises ``IndexError`` instead. ``NUMBARROW_JIT_OPTIONS='{"boundscheck": true}'``
    turns them on, and caching off, because the object replaces the default ``{"cache": true}``. With caching on,
    through ``{"cache": true, "boundscheck": true}`` or numba's ``NUMBA_BOUNDSCHECK=1``, numba still loads code that it
    cached without bounds checks, so point ``NUMBA_CACHE_DIR`` at a new directory as well.

    A negative index down to ``-8 * len(bitmap)`` counts back from the end, as a numpy index does: it reads the bit of
    element ``8 * len(bitmap) + index_``. That read stays inside the bitmap, so bounds checks do not catch it. A lower
    index reads outside the bitmap, and bounds checks catch it.

    :param index\\_: the zero-based index of the element
    :param bitmap: the packed validity bitmap
    :returns: True if the element's bit is 0, False if it is 1
    """
    byte_for_index = bitmap[index_ // 8]
    bit_position_in_byte = index_ % 8
    return not (byte_for_index >> bit_position_in_byte) % 2


@jit_with_options(Array(bool_, 1, "C")(int64, int64, Array(uint8, 1, "C", readonly=True)))
def unpack_booleans(offset: int, length: int, packed_data: np.ndarray) -> np.ndarray:
    """Return a new array of the ``length`` booleans packed in ``packed_data``, from bit ``offset`` on.

    Arrow packs the values of a boolean array as it packs a validity bitmap: one bit per value, from the least
    significant bit of each byte up. ``packed_data`` is a one-dimensional, C-contiguous uint8 array, read-only or
    writable.

    ``offset + length`` must not exceed ``8 * len(packed_data)``. numba compiles without bounds checks by default, so
    the bits past that end come from memory outside ``packed_data``, and the read can crash the process. With bounds
    checks, which ``NUMBARROW_JIT_OPTIONS='{"boundscheck": true}'`` or numba's ``NUMBA_BOUNDSCHECK=1`` turns on, the
    function raises ``IndexError`` instead. With caching on, numba still loads code that it cached without bounds
    checks, so point ``NUMBA_CACHE_DIR`` at a new directory as well.

    :param offset: the position of the first bit to read, counted from the first bit of ``packed_data``
    :param length: the number of booleans to return
    :param packed_data: the bytes that hold the packed bits
    :returns: a new array of ``length`` booleans
    """
    result = np.empty(length, dtype=np.bool_)
    for i in range(length):
        byte_idx = (offset + i) // 8
        bit_idx = (offset + i) % 8
        result[i] = (packed_data[byte_idx] >> bit_idx) & 1
    return result


# is_null_struct reads a struct's nulls in two layers, a design inspired by Awkward Array's BitMaskedArray(RecordArray):
# https://awkward-array.org/doc/main/reference/generated/ak.contents.BitMaskedArray.html
#
# The signature is explicit, so numba's on-disk cache holds one entry for is_null_struct. Without a signature, numba
# would cache an entry for each combination of argument types that callers pass. numba names a new entry's data file
# with the lowest number that the index it has just read does not use. Two processes that fill an empty cache at the
# same time can therefore write different code under one file name, and a later process can then load the wrong code.


@jit_with_options(boolean(int64, Optional(Array(uint8, 1, "C", readonly=True)),
                          Optional(Array(uint8, 1, "C", readonly=True))))
def is_null_struct(index_, struct_bitmap, field_bitmap):
    """Return True if element ``index_`` of a struct's field is null in the struct's bitmap or in the field's.

    A struct array has a validity bitmap of its own, which marks the rows that are null as a whole, and each field has
    one for its own values. A field's bitmap need not mark the null rows of the struct, so a value is null if either
    bitmap marks it. Pass None for a bitmap that the array does not have; if both are None, the value is valid.

    numba compiles the function once, when the module is imported, for an ``int64`` index and, for each bitmap, a
    read-only uint8 array or None. Every call uses that compilation: an index of another integer type converts to
    ``int64``, and a writable bitmap is accepted as well. The bounds on ``index_`` are those of :func:`is_null`.

    :param index\\_: the zero-based index of the element, converted to ``int64``
    :param struct_bitmap: the struct array's own validity bitmap, or None
    :param field_bitmap: the field's validity bitmap, or None
    :returns: True if either bitmap marks the value null
    """
    if struct_bitmap is not None and is_null(index_, struct_bitmap):
        return True
    if field_bitmap is not None and is_null(index_, field_bitmap):
        return True
    return False
