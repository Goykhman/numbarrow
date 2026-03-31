import numpy as np
import pyarrow as pa

from numbarrow.core.adapters import arrow_array_adapter
from numbarrow.core.is_null import is_null
from numbarrow.utils.arrow_array_utils import (
    create_str_array, uniform_arrow_array_adapter
)


class TestUniformOffset:
    def test_int32_sliced(self):
        a = pa.array([10, 20, 30, 40, 50], type=pa.int32())
        s = a[2:]
        bitmap, data = uniform_arrow_array_adapter(s)
        assert len(data) == 3
        assert data[0] == 30
        assert data[1] == 40
        assert data[2] == 50

    def test_float64_sliced(self):
        a = pa.array([1.1, 2.2, 3.3, 4.4], type=pa.float64())
        s = a[1:]
        bitmap, data = uniform_arrow_array_adapter(s)
        assert len(data) == 3
        assert np.isclose(data[0], 2.2)
        assert np.isclose(data[1], 3.3)
        assert np.isclose(data[2], 4.4)

    def test_int32_sliced_with_nulls(self):
        a = pa.array([10, None, 30, None, 50], type=pa.int32())
        s = a[1:]  # [None, 30, None, 50], offset=1
        bitmap, data = arrow_array_adapter(s.cast(pa.int32()))
        assert len(data) == 4
        assert is_null(0, bitmap)
        assert not is_null(1, bitmap)
        assert data[1] == 30
        assert is_null(2, bitmap)
        assert not is_null(3, bitmap)
        assert data[3] == 50


class TestBooleanOffset:
    def test_bool_sliced(self):
        a = pa.array([True, False, True, True, False], type=pa.bool_())
        s = a[2:]  # [True, True, False]
        bitmap, data = arrow_array_adapter(s)
        assert len(data) == 3
        assert data[0] == True
        assert data[1] == True
        assert data[2] == False

    def test_bool_sliced_with_nulls(self):
        a = pa.array([True, None, False, True, None], type=pa.bool_())
        s = a[1:]  # [None, False, True, None]
        bitmap, data = arrow_array_adapter(s)
        assert len(data) == 4
        assert is_null(0, bitmap)
        assert not is_null(1, bitmap)
        assert data[1] == False
        assert not is_null(2, bitmap)
        assert data[2] == True
        assert is_null(3, bitmap)


class TestDateTimeOffset:
    def test_date32_sliced(self):
        from datetime import date
        d = [date(2020, 1, 1), date(2020, 6, 15), date(2020, 12, 31)]
        a = pa.array(d, type=pa.date32())
        s = a[1:]
        bitmap, data = arrow_array_adapter(s)
        assert len(data) == 2
        assert data[0] == np.datetime64(d[1], "D")
        assert data[1] == np.datetime64(d[2], "D")

    def test_date64_sliced(self):
        d0 = np.datetime64("2020-01-01T00:00:00.000", "ms")
        d1 = np.datetime64("2020-06-15T12:30:00.000", "ms")
        d2 = np.datetime64("2020-12-31T23:59:59.000", "ms")
        a = pa.array([d0.astype(np.int64), d1.astype(np.int64), d2.astype(np.int64)], type=pa.date64())
        s = a[1:]
        bitmap, data = arrow_array_adapter(s)
        assert len(data) == 2
        assert data[0] == d1
        assert data[1] == d2

    def test_timestamp_sliced(self):
        from datetime import datetime, timezone
        t0 = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        t1 = datetime(2020, 6, 15, 12, 30, 0, tzinfo=timezone.utc)
        t2 = datetime(2020, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        a = pa.array([t0, t1, t2], type=pa.timestamp("us", "UTC"))
        s = a[1:]
        bitmap, data = arrow_array_adapter(s)
        assert len(data) == 2
        assert data[0] == np.datetime64("2020-06-15T12:30:00.000000")
        assert data[1] == np.datetime64("2020-12-31T23:59:59.000000")


class TestStringOffset:
    def test_string_sliced(self):
        a = pa.array(["alpha", "beta", "gamma", "delta"], type=pa.string())
        s = a[1:]
        result = create_str_array(s)
        assert len(result) == 3
        assert result[0] == "beta"
        assert result[1] == "gamma"
        assert result[2] == "delta"

    def test_string_with_embedded_nul(self):
        a = pa.array(["ab\x00cd", "ef\x00\x00gh", "plain"], type=pa.string())
        result = create_str_array(a)
        assert result[0] == "ab\x00cd"
        assert result[1] == "ef\x00\x00gh"
        assert result[2] == "plain"


class TestStructOffset:
    def test_struct_sliced(self):
        indices = pa.array([10, 20, 30, 40], type=pa.int32())
        ratios = pa.array([1.1, 2.2, 3.3, 4.4], type=pa.float64())
        sa = pa.StructArray.from_arrays([indices, ratios], ["idx", "ratio"])
        s = sa[1:]  # offset=1
        bitmaps, datas = arrow_array_adapter(s)
        assert len(datas["idx"]) == 3
        assert datas["idx"][0] == 20
        assert datas["idx"][1] == 30
        assert datas["idx"][2] == 40
        assert np.isclose(datas["ratio"][0], 2.2)
        assert np.isclose(datas["ratio"][1], 3.3)
        assert np.isclose(datas["ratio"][2], 4.4)


class TestBitmapOffset:
    def test_bitmap_offset_across_byte_boundary(self):
        """Offset that crosses a byte boundary in the bitmap."""
        values = [None if i % 3 == 0 else i for i in range(16)]
        a = pa.array(values, type=pa.int32())
        s = a[9:]  # offset=9, crosses into byte 1
        bitmap, data = uniform_arrow_array_adapter(s)
        # Original: [None,10,11,None,13,14,None]
        assert is_null(0, bitmap)      # index 9 -> None
        assert not is_null(1, bitmap)  # index 10
        assert not is_null(2, bitmap)  # index 11
        assert is_null(3, bitmap)      # index 12 -> None
        assert not is_null(4, bitmap)  # index 13
        assert not is_null(5, bitmap)  # index 14
        assert is_null(6, bitmap)      # index 15 -> None
