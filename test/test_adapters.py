"""Tests of the handlers in numbarrow.core.adapters that arrow_array_adapter dispatches each array type to.

The tests cover the boolean, date32, date64 and timestamp handlers, empty and large string columns, and struct
columns whose fields are boolean, string, uint8, temporal or of a type with more than two ``buffers()``. They check
the values and the bitmaps, and that a temporal handler reads each value at its own unit and keeps a zero-length
column's bitmap. ``test_arrow_array_utils.py`` covers the adapters that these handlers call, and ``test_offset.py``
covers sliced columns.
"""
import numpy as np
import pyarrow as pa
import pytest

from datetime import date, datetime
from dateutil import tz

from numbarrow.core.adapters import arrow_array_adapter
from numbarrow.core.is_null import is_null


def test_arrow_array_adapter_1():
    """Check that a bool column unpacks to a numpy bool array, and that its bitmap marks the null."""
    a = pa.array([True, None, False, True, False], type=pa.bool_())
    bitmap, data = arrow_array_adapter(a)
    assert data.dtype == np.bool_
    assert data[0] and not data[2] and data[3] and not data[4]
    assert (
        not is_null(0, bitmap) and is_null(1, bitmap) and not is_null(2, bitmap) and not is_null(3, bitmap)
    )


def test_arrow_array_adapter_2():
    """Check that a date32 column adapts to ``datetime64[D]`` values, and that its bitmap marks the null."""
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
    """Check that a date64 column adapts to ``datetime64[ms]`` values, and that its bitmap marks the null."""
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
    """Check that a ``timestamp[us, UTC]`` column adapts to naive ``datetime64`` values in UTC, with its null marked."""
    d0 = datetime(2012, 7, 4, 10, 11, 6, tzinfo=tz.tzutc())
    d1 = datetime(2012, 12, 6, 4, 6, 59, tzinfo=tz.tzutc())
    a = pa.array([d0, d1, None], type=pa.timestamp("us", "UTC"))
    bitmap, data = arrow_array_adapter(a)
    assert data[0] == np.datetime64("2012-07-04T10:11:06.000000")
    assert data[1] == np.datetime64("2012-12-06T04:06:59.000000")
    assert (
        not is_null(0, bitmap) and not is_null(1, bitmap) and is_null(2, bitmap)
    )


def test_empty_str_array():
    bitmap, data = arrow_array_adapter(pa.array([], type=pa.string()))
    assert bitmap is None
    assert data.dtype == np.dtype("|U1") and len(data) == 0


def test_large_str_array():
    bitmap, data = arrow_array_adapter(pa.array(["large", "string"], type=pa.large_string()))
    assert bitmap is None
    assert data.dtype == np.dtype("|U6") and len(data) == 2


def test_bool_struct_child_every_bit_pattern():
    """Check every bit pattern of a boolean struct field, from one to twelve elements.

    A boolean field is bit-packed. A reader that takes one byte per element covers eight times the buffer, so it
    reads wrong values and runs past the end. Patterns that are False after the first element read correctly even
    then, so the test covers every pattern rather than a sample.
    """
    for n in range(1, 13):
        for pattern in range(1 << n):
            vals = [bool((pattern >> i) & 1) for i in range(n)]
            arr = pa.array([{"f": v} for v in vals], type=pa.struct([("f", pa.bool_())]))
            _, _, datas = arrow_array_adapter(arr)
            assert datas["f"].tolist() == vals, (n, pattern)


def test_bool_struct_child_matches_the_standalone_child():
    vals = [True, False, True, True, False, False, True, False, True]
    arr = pa.array([{"f": v} for v in vals], type=pa.struct([("f", pa.bool_())]))
    _, _, datas = arrow_array_adapter(arr)
    _, standalone = arrow_array_adapter(arr.field("f"))
    assert datas["f"].tolist() == standalone.tolist() == vals


def test_multi_buffer_struct_children_adapt_or_raise_a_typed_error():
    """Check that a struct field with more than two ``buffers()`` adapts or raises an error naming its type.

    Python's own "too many values to unpack" would name neither the field nor its type.
    """
    adapts = {"string": pa.string(), "large_string": pa.large_string()}
    refuses = {
        "binary": pa.binary(), "large_binary": pa.large_binary(),
        "nested_struct": pa.struct([("x", pa.int64())]),
        "list": pa.list_(pa.int64()), "large_list": pa.large_list(pa.int64()),
        "fixed_size_list": pa.list_(pa.int64(), 2),
        "map": pa.map_(pa.string(), pa.int64()),
    }
    for child_ty in adapts.values():
        arr = pa.array([None], type=pa.struct([("f", child_ty)]))
        _, bitmaps, datas = arrow_array_adapter(arr)
        assert "f" in datas and "f" in bitmaps
    for name, child_ty in refuses.items():
        arr = pa.array([None], type=pa.struct([("f", child_ty)]))
        with pytest.raises(NotImplementedError) as excinfo:
            arrow_array_adapter(arr)
        # The assertion looks for the type, not the field name: the letter f appears in almost any message, and every
        # message here names the type. An assertion that also accepted the quoted name 'f' would pass a message that
        # left the type out.
        assert str(child_ty) in str(excinfo.value), name


def test_struct_with_a_string_field():
    # PySpark sends a struct<name: string, id: bigint> column in this shape.
    arr = pa.array([{"name": "a", "id": 1}, {"name": "bb", "id": 2}],
                   type=pa.struct([("name", pa.string()), ("id", pa.int64())]))
    _, _, datas = arrow_array_adapter(arr)
    assert list(datas["name"]) == ["a", "bb"]
    assert datas["id"].tolist() == [1, 2]


def test_uint8_column_and_struct_child():
    arr = pa.array([3, 250, None], type=pa.uint8())
    bitmap, data = arrow_array_adapter(arr)
    assert data.tolist() == [3, 250, 0] or data[:2].tolist() == [3, 250]
    assert not is_null(0, bitmap) and is_null(2, bitmap)
    struct_arr = pa.array([{"f": 3}, {"f": 250}], type=pa.struct([("f", pa.uint8())]))
    _, _, datas = arrow_array_adapter(struct_arr)
    assert datas["f"].tolist() == [3, 250]


def test_zero_length_date_and_timestamp():
    for label, full in [
        ("date32", pa.array([1, 2], type=pa.int32()).cast(pa.date32())),
        ("date64", pa.array([86400000], type=pa.int64()).cast(pa.date64())),
        ("timestamp", pa.array([1], type=pa.int64()).cast(pa.timestamp("ms"))),
    ]:
        empty = full.slice(0, 0)
        _, data = arrow_array_adapter(empty)
        assert len(data) == 0, label
        _, _, datas = arrow_array_adapter(pa.StructArray.from_arrays([empty], ["f"]))
        assert len(datas["f"]) == 0, label


if __name__ == "__main__":
    test_arrow_array_adapter_1()
    test_arrow_array_adapter_2()
    test_arrow_array_adapter_3()
    test_arrow_array_adapter_4()
    test_empty_str_array()


def test_a_zero_length_temporal_column_keeps_its_bitmap_presence():
    """Check that a zero-length temporal column with a validity buffer gets an empty bitmap rather than None.

    A cast of a zero-length array drops the validity buffer, so the temporal handlers take the bitmap from the source
    column instead of the cast. The bitmap is None only for a column without a validity buffer.
    """
    for arrow_type, source_type in ((pa.date32(), pa.int32()), (pa.date64(), pa.int64()),
                                    (pa.timestamp("us", "UTC"), pa.int64())):
        source = pa.array([1, None], type=source_type).cast(arrow_type).slice(1, 0)
        assert source.buffers()[0] is not None
        bitmap, data = arrow_array_adapter(source)
        assert bitmap is not None and bitmap.dtype == np.uint8 and len(bitmap) == 0 and len(data) == 0


def test_the_64bit_date_view_refuses_a_unit_that_is_not_the_arrays_own():
    """Check that the date64 and timestamp view refuses a unit other than the array's own, and any other type.

    The function views the int64 values without rescaling them, so another unit would misplace every value:
    ``timestamp[ms]`` viewed as ``datetime64[s]`` lands in the year 51971.
    """
    from numbarrow.core.adapters import cast_64bit_date_arrow_to_numpy_array
    stamps = pa.array([datetime(2020, 1, 1, 12)], type=pa.timestamp("ms"))
    _, data = cast_64bit_date_arrow_to_numpy_array(stamps, np.dtype("datetime64[ms]"))
    assert data.tolist() == [datetime(2020, 1, 1, 12)]
    with pytest.raises(ValueError, match=r"ms instants.*not as datetime64\[s\]"):
        cast_64bit_date_arrow_to_numpy_array(stamps, np.dtype("datetime64[s]"))
    with pytest.raises(ValueError, match="not a date64 or timestamp"):
        cast_64bit_date_arrow_to_numpy_array(pa.array([1], type=pa.int64()), np.dtype("datetime64[s]"))
    # A time64 and a duration carry a unit too, but their values are a time of day and an elapsed span, not instants.
    # Viewed as datetime64, an hour of the day and a ninety-second span would land on 1970-01-01.
    for clocked in (pa.array([3_600_000_000], type=pa.time64("us")), pa.array([90_000], type=pa.duration("ms"))):
        with pytest.raises(ValueError, match="not a date64 or timestamp"):
            cast_64bit_date_arrow_to_numpy_array(clocked, np.dtype(f"datetime64[{clocked.type.unit}]"))


def test_a_timestamp_is_read_at_its_own_unit():
    """Check that a timestamp of each unit adapts to ``datetime64`` of that unit and reads back the same instant.

    The test covers all four units, because a handler that read every unit as microseconds would still pass at
    microseconds, while a millisecond column read that way lands in 1970.
    """
    instant = datetime(2020, 9, 13, 12, 26, 40)
    for unit in ("s", "ms", "us", "ns"):
        _, data = arrow_array_adapter(pa.array([instant], type=pa.timestamp(unit)))
        assert data.dtype == np.dtype(f"datetime64[{unit}]"), unit
        assert data.astype("datetime64[us]").tolist() == [instant], unit
