import numpy as np
import pyarrow as pa
import pytest

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


def test_empty_str_array():
    bitmap, data = arrow_array_adapter(pa.array([], type=pa.string()))
    assert bitmap is None
    assert data.dtype == np.dtype("|U1") and len(data) == 0


def test_large_str_array():
    bitmap, data = arrow_array_adapter(pa.array(["large", "string"], type=pa.large_string()))
    assert bitmap is None
    assert data.dtype == np.dtype("|U6") and len(data) == 2


def test_bool_struct_child_every_bit_pattern():
    # A boolean child is bit-packed. Read at one byte per element it spans
    # eight times its own buffer, so the values are wrong and the read runs
    # past the end. Every pattern up to twelve elements is checked because the
    # all-False-after-index-0 patterns happen to survive the wrong stride.
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
    # Every child type whose Arrow layout uses more than two buffers. Each must
    # either adapt or name itself in the error; "too many values to unpack"
    # names neither the field nor the type.
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
        # Not `"f" in message`: the letter f appears in almost any message.
        # Asserted without a disjunction on the field name: every message form
        # here carries the type, and `or "'f'" in ...` let the type be dropped
        # from the struct-field message with the suite staying green.
        assert str(child_ty) in str(excinfo.value), name


def test_struct_with_a_string_field():
    # The shape PySpark produces for struct<name: string, id: bigint>.
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
    # The temporal handlers took the bitmap from the cast, and a zero-length
    # cast drops the validity buffer, so the documented None-only-without-a-
    # buffer rule broke for date32, date64 and timestamp columns.
    for arrow_type, source_type in ((pa.date32(), pa.int32()), (pa.date64(), pa.int64()),
                                    (pa.timestamp("us", "UTC"), pa.int64())):
        source = pa.array([1, None], type=source_type).cast(arrow_type).slice(1, 0)
        assert source.buffers()[0] is not None
        bitmap, data = arrow_array_adapter(source)
        assert bitmap is not None and bitmap.dtype == np.uint8 and len(bitmap) == 0 and len(data) == 0
