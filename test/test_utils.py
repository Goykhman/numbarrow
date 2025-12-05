import numpy as np
from numpy.testing import assert_equal
from numbarrow.utils.utils import arrays_viewers


def test_int32_array_from_ptr_as_int():
    a = np.array([137, 314], dtype=np.int32)
    a_p = a.ctypes.data
    a_ = arrays_viewers[np.int32](a_p, len(a))
    assert_equal(a_, a)


if __name__ == "__main__":
    test_int32_array_from_ptr_as_int()
