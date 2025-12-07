import numpy as np

from numbarrow.core.is_null import is_null


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


if __name__ == "__main__":
    test_is_null_1()
    test_is_null_2()
