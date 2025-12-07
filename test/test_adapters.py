import numpy as np
import pyarrow as pa

from datetime import date, datetime
from dateutil import tz

from numbarrow.core.adapters import arrow_array_adapter
from numbarrow.core.is_null import is_null


def test_arrow_array_adapter_1():
    a = pa.array([True, None, False, True, False], type=pa.bool_())
    bitmap, data = arrow_array_adapter(a)
    assert data.dtype == np.bool_
    assert data[0] and not data[2] and data[3] and not data[4]
    assert (
        not is_null(0, bitmap) and is_null(1, bitmap) and not is_null(2, bitmap) and not is_null(3, bitmap)
    )


def test_arrow_array_adapter_2():
    d0 = date(2012, 7, 4)
    d1 = date(2012, 12, 6)
    d3 = date(2013, 6, 11)
    a = pa.array([d0, d1, None, d3], type=pa.date32())
    bitmap, data = arrow_array_adapter(a)
    assert data[0] == np.datetime64(d0, "D")
    assert data[1] == np.datetime64(d1, "D")
    assert data[3] == np.datetime64(d3, "D")
    assert (
        not is_null(0, bitmap) and not is_null(1, bitmap) and is_null(2, bitmap) and not is_null(3, bitmap)
    )


def test_arrow_array_adapter_3():
    d0 = np.datetime64("2012-07-04T10:11:06.123", "ms")
    d2 = np.datetime64("2012-12-06T13:08:09.97", "ms")
    d3 = np.datetime64("2013-06-11T23:45:11.1", "ms")
    a = pa.array([d0.astype(np.int64), None, d2.astype(np.int64), d3.astype(np.int64)], type=pa.date64())
    bitmap, data = arrow_array_adapter(a)
    assert data[0] == d0
    assert data[2] == d2
    assert data[3] == d3
    assert (
        not is_null(0, bitmap) and is_null(1, bitmap) and not is_null(2, bitmap) and not is_null(3, bitmap)
    )


def test_arrow_array_adapter_4():
    d0 = datetime(2012, 7, 4, 10, 11, 6, tzinfo=tz.tzutc())
    d1 = datetime(2012, 12, 6, 4, 6, 59, tzinfo=tz.tzutc())
    a = pa.array([d0, d1, None], type=pa.timestamp("us", "UTC"))
    bitmap, data = arrow_array_adapter(a)
    assert data[0] == np.datetime64("2012-07-04T10:11:06.000000")
    assert data[1] == np.datetime64("2012-12-06T04:06:59.000000")
    assert (
        not is_null(0, bitmap) and not is_null(1, bitmap) and is_null(2, bitmap)
    )


if __name__ == "__main__":
    test_arrow_array_adapter_1()
    test_arrow_array_adapter_2()
    test_arrow_array_adapter_3()
    test_arrow_array_adapter_4()
