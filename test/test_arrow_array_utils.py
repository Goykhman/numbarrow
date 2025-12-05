import numpy as np
import pyarrow as pa
from numbarrow.utils.arrow_array_utils import (
    create_str_array, structured_array_adapter, uniform_arrow_array_adapter
)
from numbarrow.core.is_null import is_null


def test_create_str_array():
    pa_a = pa.array(["first", "second", None, "third", "fourth element", "f"], type=pa.string())
    np_a = create_str_array(pa_a)
    ref = ['first', 'second', '', 'third', 'fourth element', 'f']
    assert all([np_a_e == ref_e for np_a_e, ref_e in zip(np_a, ref)])


def test_is_null_1():
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


def test_structured_array_adapter():
    indices = pa.array([14, 89, None, 105], type=pa.int32())
    ratios = pa.array([1.41, None, 1.72, 9.99], type=pa.float64())
    struct_array = pa.StructArray.from_arrays([indices, ratios], ["indices", "ratios"])
    bitmap, data = structured_array_adapter(struct_array)
    indices_bitmap = bitmap["indices"]
    indices_data = data["indices"]
    assert len(indices_bitmap) == 1, "length of `indices` is less than 9"
    assert indices_bitmap[0] == int("1011", base=2)
    assert indices_data[0] == 14 and indices_data[1] == 89 and indices_data[3] == 105
    ratios_bitmap = bitmap["ratios"]
    ratios_data = data["ratios"]
    assert len(ratios_bitmap) == 1, "length of `ratios` is less than 9"
    assert ratios_bitmap[0] == int("1101", base=2)
    assert np.allclose([ratios_data[0], ratios_data[2], ratios_data[3]], [1.41, 1.72, 9.99])


def test_uniform_arrow_array_adapter():
    a = pa.array([141, None, 172, 314, 271], type=pa.int32())
    bitmap, data = uniform_arrow_array_adapter(a)
    assert data[0] == 141 and data[2] == 172 and data[3] == 314 and data[4] == 271
    assert (
        not is_null(0, bitmap) and is_null(1, bitmap) and not is_null(2, bitmap)
        and not is_null(3, bitmap) and not is_null(4, bitmap)
    )


if __name__ == "__main__":
    test_create_str_array()
    test_is_null_1()
    test_is_null_2()
    test_structured_array_adapter()
    test_uniform_arrow_array_adapter()
