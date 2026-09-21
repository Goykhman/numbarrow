import numpy as np
from numpy.testing import assert_equal
from numbarrow.utils.utils import arrays_viewers


def test_int32_array_from_ptr_as_int():
    a = np.array([137, 314], dtype=np.int32)
    a_p = a.ctypes.data
    a_ = arrays_viewers[np.int32](a_p, len(a))
    assert_equal(a_, a)


def test_a_viewer_is_built_when_first_asked_for_and_kept():
    # arrays_viewers compiles a viewer through the factory on the first
    # request for a dtype and hands the same one back after.
    a = np.array([0.5, -1.0], dtype=np.float64)
    viewer = arrays_viewers[np.float64]
    assert viewer(a.ctypes.data, len(a)).tolist() == [0.5, -1.0]
    assert arrays_viewers[np.float64] is viewer


if __name__ == "__main__":
    test_int32_array_from_ptr_as_int()
