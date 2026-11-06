"""Tests of is_null and unpack_booleans, the jitted functions that read Arrow's packed bitmaps.

Arrow packs one bit per element, least significant bit first, and a set bit marks a valid element. The tests check
is_null on hand-built bitmaps of one and two bytes, and both functions on read-only arrays. ``test_offset.py`` tests
``is_null_struct``, and the adapter tests read the bitmaps that the adapters return.
"""
import numpy as np

from numbarrow.core.is_null import is_null, unpack_booleans


def test_is_null_1():
    """Check bits 0 to 6 of a one-byte bitmap, least significant bit first."""
    bitmap = np.array([int("1010001", base=2)], dtype=np.uint8)
    assert all([
        not is_null(0, bitmap),
        is_null(1, bitmap),
        is_null(2, bitmap),
        is_null(3, bitmap),
        not is_null(4, bitmap),
        is_null(5, bitmap),
        not is_null(6, bitmap)
    ])


def test_is_null_2():
    """Check a two-byte bitmap, where element 8 reads the lowest bit of the second byte."""
    bitmap = np.array([int("10100010", base=2), int("1", base=2)], dtype=np.uint8)
    assert all([
        is_null(0, bitmap),
        not is_null(1, bitmap),
        is_null(2, bitmap),
        is_null(3, bitmap),
        is_null(4, bitmap),
        not is_null(5, bitmap),
        is_null(6, bitmap),
        not is_null(7, bitmap),
        not is_null(8, bitmap),
        is_null(9, bitmap),
        is_null(10, bitmap),
    ])


def test_is_null_accepts_a_readonly_bitmap():
    # is_null declares its bitmap parameter read-only, so it accepts a read-only array, such as np.frombuffer over a
    # validity buffer read from an Arrow IPC stream. numba accepts a writable array for a read-only parameter too, so
    # a caller with a writable bitmap, such as one an adapter returns, gets the same answer.
    writable = np.array([int("1010001", base=2)], dtype=np.uint8)
    readonly = writable.copy()
    readonly.flags.writeable = False
    assert is_null(1, readonly) == is_null(1, writable)
    assert not is_null(0, readonly) and is_null(2, readonly)


def test_unpack_booleans_accepts_a_readonly_buffer():
    packed = np.array([int("10110", base=2)], dtype=np.uint8)
    packed.flags.writeable = False
    assert unpack_booleans(0, 5, packed).tolist() == [False, True, True, False, True]


if __name__ == "__main__":
    test_is_null_1()
    test_is_null_2()
