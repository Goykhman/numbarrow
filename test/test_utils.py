"""Tests of the viewers in numbarrow.utils.utils, the jitted functions that read memory at an address as a numpy array.

The tests read numpy arrays through viewers, and check that ``arrays_viewers`` compiles one viewer per key and keeps
it. Two tests check the names of the viewers for structured dtypes. numba names a function's cache files after
its qualified name and source line, and every viewer comes from one source line, so two viewers with one name would
share a cache index. ``test_cache.py`` checks the cache files themselves, across processes.
"""
import numpy as np
from numpy.testing import assert_equal
from numbarrow.utils.utils import arrays_viewers, numpy_array_from_ptr_factory


def test_int32_array_from_ptr_as_int():
    a = np.array([137, 314], dtype=np.int32)
    a_p = a.ctypes.data
    a_ = arrays_viewers[np.int32](a_p, len(a))
    assert_equal(a_, a)


def test_a_viewer_is_built_when_first_asked_for_and_kept():
    # arrays_viewers compiles a viewer through numpy_array_from_ptr_factory on the first request for a dtype, and
    # returns the same function on every later request.
    a = np.array([0.5, -1.0], dtype=np.float64)
    viewer = arrays_viewers[np.float64]
    assert viewer(a.ctypes.data, len(a)).tolist() == [0.5, -1.0]
    assert arrays_viewers[np.float64] is viewer


if __name__ == "__main__":
    test_int32_array_from_ptr_as_int()


def test_structured_dtypes_of_one_itemsize_get_viewers_of_their_own():
    # numpy names both dtypes void32. With one name for both viewers, a process that loads them from a cache that two
    # processes filled can run the first viewer's code for the second.
    first = numpy_array_from_ptr_factory(np.dtype([("a", "<i4")]))
    second = numpy_array_from_ptr_factory(np.dtype([("b", "<f4")]))
    assert first.__qualname__ != second.__qualname__
    ints = np.array([(7,), (-3,)], dtype=[("a", "<i4")])
    floats = np.array([(1.5,), (-2.5,)], dtype=[("b", "<f4")])
    assert first(ints.ctypes.data, 2).tolist() == [(7,), (-3,)]
    assert second(floats.ctypes.data, 2).tolist() == [(1.5,), (-2.5,)]


def test_a_structured_dtype_with_out_of_order_fields_gets_a_viewer():
    # numpy's multi-field indexing, rec[["b", "a"]], returns a dtype with out-of-order fields, and numpy refuses to
    # build dtype.descr for such a dtype. The factory names the viewer after a hash of the dtype's repr instead, which
    # numpy builds for any dtype.
    rec = np.zeros(2, dtype=[("a", "<f4"), ("b", "<i4")])
    rec["a"] = [1.5, 2.5]
    rec["b"] = [7, 8]
    reordered = rec[["b", "a"]]
    viewer = numpy_array_from_ptr_factory(reordered.dtype)
    assert viewer(reordered.ctypes.data, 2).tolist() == [(1.5, 7), (2.5, 8)]
    assert viewer.__qualname__ != numpy_array_from_ptr_factory(rec.dtype).__qualname__
