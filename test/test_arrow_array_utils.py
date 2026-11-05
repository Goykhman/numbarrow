"""Tests of the array adapters in numbarrow.utils.arrow_array_utils, called directly and through arrow_array_adapter.

The tests cover string, struct, list-of-struct, fixed-width and boolean arrays: whole, sliced to an offset, and at
length zero. They check that the data comes back read-only and the bitmap as a copy the caller owns. They also check
that a view keeps its Arrow buffer alive after the caller drops the source, and that a refusal names an unsupported
array's type without its values. Three tests run a child interpreter: one under ``python -O``, and two where a defect
would crash the process. ``test_adapters.py`` checks the values of the boolean and temporal adapters,
``test_offset.py`` checks sliced columns, and ``test_is_null.py`` checks the bitmap readers.
"""
import gc
import os
import random
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from numbarrow.core.adapters import arrow_array_adapter
from numbarrow.utils.arrow_array_utils import (
    create_bitmap, create_str_array, structured_array_adapter,
    uniform_arrow_array_adapter
)
from numbarrow.core.is_null import is_null

# The checkout root on PYTHONPATH makes a child interpreter import the numbarrow under test. Without it, a child that
# starts in any other directory imports the installed numbarrow, which may be another copy.
CHILD_ENV = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parent.parent))


def owner_of(array):
    """Return the first pyarrow Buffer or Array in an array's base chain, or None if the first eight links hold none."""
    node = array
    for _ in range(8):
        node = node.obj if isinstance(node, memoryview) else getattr(node, "base", None)
        if node is None:
            return None
        if isinstance(node, (pa.Buffer, pa.Array)):
            return node
    return None


def test_create_str_array():
    """Check that create_str_array returns every value, an empty string at the null slot, and a bitmap marking it."""
    pa_a = pa.array(["first", "second", None, "third", "fourth element", "f"], type=pa.string())
    bitmap, np_a = create_str_array(pa_a)
    ref = ["first", "second", "", "third", "fourth element", "f"]
    assert is_null(2, bitmap) and all(not is_null(i, bitmap) for i in range(len(pa_a)) if i != 2)
    assert len(np_a) == len(ref)
    assert list(np_a) == ref


def test_create_str_array_long_with_null():
    """Check a string column of ten values, whose bitmap spans two bytes, at offset 0 and at offset 1.

    numpy raises ``ValueError`` on the truth value of an array of more than one element. Code that tests the bitmap
    with ``bool()`` therefore fails on these bitmaps, although it passes on a bitmap of one byte.
    """
    vals = ["s%d" % i for i in range(9)] + [None]
    bitmap, np_a = create_str_array(pa.array(vals, type=pa.string()))
    assert is_null(9, bitmap) and all(not is_null(i, bitmap) for i in range(9))
    assert np_a[0] == "s0" and np_a[8] == "s8" and np_a[9] == ""
    sliced = pa.array(vals + ["tail"], type=pa.string())[1:]
    bitmap_s, np_s = create_str_array(sliced)
    assert is_null(8, bitmap_s) and not is_null(9, bitmap_s)


def test_create_str_array_all_null_bitmap_is_a_copy():
    """Check that the bitmap of an all-null string column is a copy, not a view over the Arrow validity buffer.

    Every byte of an all-null bitmap is zero, so code that copies only a bitmap that tests true skips the copy here. A
    view reads freed memory once the caller drops the source array. The test checks the base: a copy has none, and a
    slice of the view has the view as its base.
    """
    bitmap, np_a = create_str_array(pa.array([None] * 8, type=pa.string()))
    assert all(is_null(i, bitmap) for i in range(8))
    assert bitmap.base is None


def test_structured_array_adapter():
    """Check a struct of an int32 and a float64 field: no struct bitmap, and each field's own bitmap and values."""
    indices = pa.array([14, 89, None, 105], type=pa.int32())
    ratios = pa.array([1.41, None, 1.72, 9.99], type=pa.float64())
    struct_array = pa.StructArray.from_arrays([indices, ratios], ["indices", "ratios"])
    struct_bitmap, bitmap, data = structured_array_adapter(struct_array)
    assert struct_bitmap is None  # no struct-level nulls in this test
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


def test_uniform_arrow_array_adapter_1():
    """Check the values and the bitmap of an int32 column with one null, read through the uniform adapter."""
    a = pa.array([141, None, 172, 314, 271], type=pa.int32())
    bitmap, data = uniform_arrow_array_adapter(a)
    assert data[0] == 141 and data[2] == 172 and data[3] == 314 and data[4] == 271
    assert (
        not is_null(0, bitmap) and is_null(1, bitmap) and not is_null(2, bitmap)
        and not is_null(3, bitmap) and not is_null(4, bitmap)
    )


def test_bitmap_at_offset_zero_does_not_alias_the_source():
    source = pa.array([1, 2, None, 4, 5, 6, 7, 8, 9], type=pa.int64())
    bitmap, _ = arrow_array_adapter(source)
    bitmap[0] = 0
    assert source.to_pylist() == [1, 2, None, 4, 5, 6, 7, 8, 9]


def test_short_offset_zero_slice_does_not_hand_back_the_parents_bits():
    parent = pa.array([1, 2, None] + list(range(4, 21)), type=pa.int64())
    bitmap, _ = arrow_array_adapter(parent.slice(0, 4))
    bitmap[0] = 0
    assert parent.to_pylist()[:4] == [1, 2, None, 4]


def test_bitmap_never_aliases_arrow_memory():
    # The test writes through the bitmap rather than walk its base chain. A view built over a raw address has no
    # pyarrow object in its base chain either, so a check of the chain cannot tell such a view from a copy.
    for offset in (0, 1, 3, 8, 16):
        source = pa.array([1, None, 3, 4, 5, 6, 7, 8, 9, 10, None, 12, 13, 14, 15, 16, 17, 18],
                          type=pa.int64())
        before = source.to_pylist()
        bitmap, _ = arrow_array_adapter(source.slice(offset))
        bitmap[:] = 0
        assert source.to_pylist() == before, offset


def test_bitmap_survives_the_source_being_dropped():
    source = pa.array([1, 2, None, 4], type=pa.int64())
    bitmap, _ = arrow_array_adapter(source)
    before = bitmap.copy()
    del source
    gc.collect()
    assert bitmap.tolist() == before.tolist()


def test_the_offset_bitmap_owns_exactly_the_bytes_it_returns():
    """Check that the offset path returns exactly ``(length + 7) // 8`` bytes, in an array that owns them.

    The offset path returns the output of ``np.packbits`` unchanged. packbits fills a partial last byte with zero bits
    and returns a new array, so the result has the right length and no base. A longer result, or a slice of the
    packbits output, fails here.
    """
    for n in (9, 16, 17, 33, 64, 65, 129):
        for offset in (1, 3, 7, 8):
            if offset >= n:
                continue
            length = n - offset
            source = pa.array([None if i % 3 == 0 else i for i in range(n)], type=pa.int64())
            bitmap = create_bitmap(source.buffers()[0], offset, length)
            assert len(bitmap) == (length + 7) // 8, (n, offset)
            assert bitmap.base is None, (n, offset)


def test_the_offset_bitmap_survives_the_source_being_dropped():
    """Check that a bitmap from the offset path keeps its bits after the source array is dropped.

    The offset-0 path copies its bitmap. The offset path builds a new one with ``np.packbits``, so no Arrow memory backs
    it and it needs no copy.
    """
    source = pa.array([1, None, 3, 4, 5, None, 7, 8, 9, 10], type=pa.int64())
    bitmap = create_bitmap(source.buffers()[0], 3, 6)
    before = bitmap.tolist()
    del source
    gc.collect()
    # These arrays can take over the memory of the dropped source, which a bitmap that viewed it would then read.
    filler = [pa.array(list(range(64)), type=pa.int64()) for _ in range(200)]
    after = bitmap.tolist()
    del filler
    assert after == before


def test_sliced_list_returns_only_its_own_rows():
    """Check that a list column built over part of a child array returns the elements of its own rows only.

    ``ListArray.values`` is the whole child array and ignores the list's offset, so for each producer below it also
    holds elements that belong to no row of the column. The adapter bounds it by the first and last of the column's
    offsets.
    """
    rows = [[{"v": 10}, {"v": 11}], [{"v": 20}, {"v": 21}], [{"v": 30}, {"v": 31}]]
    ty = pa.list_(pa.struct([("v", pa.int64())]))
    larr = pa.array(rows, type=ty)
    table = pa.table({"c": larr})
    producers = {
        "Array.slice": larr.slice(2, 1),
        "Array[a:b]": larr[2:3],
        "Table.slice().to_batches()": table.slice(2, 1).to_batches()[0].column("c"),
        "RecordBatch.slice().column()": table.to_batches()[0].slice(1, 2).column("c"),
        "to_batches(max_chunksize)": table.to_batches(max_chunksize=2)[1].column("c"),
        "ListArray.from_arrays": pa.ListArray.from_arrays(pa.array([2, 4, 6], pa.int32()), larr.values),
        # Each producer above ends at the last row, where offsets[n] equals len(values), so none of them tests the
        # upper bound. The next two end before the last row.
        "middle row only": larr.slice(1, 1),
        "leading two rows": larr.slice(0, 2),
    }
    for name, arr in producers.items():
        _, _, datas = arrow_array_adapter(arr)
        want = [element["v"] for row in arr.to_pylist() for element in row]
        assert datas["v"].tolist() == want, name


def test_sliced_list_sees_the_slice_null_count_not_the_parent_s():
    """Check that the refusal of a null row reads the null count of the slice, not of its parent.

    The slice that contains the null row is refused. The slice that excludes it adapts, at offset 2.
    """
    ty = pa.list_(pa.struct([("v", pa.int64())]))
    larr = pa.array([[{"v": 1}], None, [{"v": 3}]], type=ty)
    with pytest.raises(NotImplementedError, match="null row"):
        arrow_array_adapter(larr.slice(1, 2))
    _, _, datas = arrow_array_adapter(larr.slice(2, 1))
    assert datas["v"].tolist() == [3]


def test_a_trailing_nul_is_refused_rather_than_truncated():
    """Check that a string ending in NUL is refused with its element's index, and that other strings with NULs adapt.

    The pattern includes the element's index, because a message that points at the wrong element is no help on a wide
    batch. A match on the phrase alone would miss an index written into the message as a constant.
    """
    # numpy pads a short element of a |U array with NUL, so it cannot tell a trailing NUL in a value from that padding:
    # "ab\x00" stored in a |U array reads back as "ab".
    for values, index_ in ((["ab\x00", "abc"], 0), (["ab\x00\x00"], 0), (["ok", "x\x00"], 1)):
        pattern = rf"element {index_} of a {len(values)}-element string array"
        with pytest.raises(ValueError, match=pattern):
            arrow_array_adapter(pa.array(values, type=pa.string()))
        with pytest.raises(ValueError, match=pattern):
            arrow_array_adapter(pa.array(values, type=pa.large_string()))
    # A leading or an interior NUL is not padding, so those values adapt unchanged.
    _, data = arrow_array_adapter(pa.array(["a\x00b", "\x00ab", "abc"], type=pa.string()))
    assert data.tolist() == ["a\x00b", "\x00ab", "abc"]
    # The adapter never decodes a null slot, so a masked-out value that ends in NUL is not refused. pc.if_else builds
    # the column because it leaves the masked-out bytes in the data buffer. take() would give the null slot no bytes
    # (offsets [0, 0, 2]). The `non_empty` term alone would then keep the slot out of the check, and the case would not
    # test the `live` term.
    masked = pc.if_else(pa.array([False, True]),
                        pa.array(["ab\x00", "ok"], type=pa.string()),
                        pa.scalar(None, pa.string()))
    assert masked.to_pylist() == [None, "ok"]
    offsets = masked.buffers()[1]
    assert np.frombuffer(memoryview(offsets), dtype=np.int32)[:3].tolist() == [0, 3, 5], \
        "the null slot must carry its dead bytes, or this does not test `live`"
    _, data = arrow_array_adapter(masked)
    assert data.tolist() == ["", "ok"]


def test_plain_list_raises_a_typed_error():
    with pytest.raises(NotImplementedError) as excinfo:
        arrow_array_adapter(pa.array([[1, 2], [3]], pa.list_(pa.int64())))
    assert "int64" in str(excinfo.value)


def test_plain_list_raises_the_same_typed_error_under_dash_O():
    """Check that a list of non-struct elements raises ``NotImplementedError`` under ``python -O`` as well.

    ``-O`` strips assert statements, so the refusal must be an explicit raise.
    """
    src = (
        "import pyarrow as pa\n"
        "from numbarrow.core.adapters import arrow_array_adapter\n"
        "try:\n"
        "    arrow_array_adapter(pa.array([[1, 2], [3]], pa.list_(pa.int64())))\n"
        "except NotImplementedError as exc:\n"
        "    print('typed', 'int64' in str(exc))\n"
    )
    # -O applies to a whole interpreter, so the case runs in a child.
    out = subprocess.run([sys.executable, "-O", "-c", src], capture_output=True, text=True, env=CHILD_ENV)
    assert out.stdout.strip() == "typed True", out


def test_map_column_still_adapts():
    # pa.MapArray is a subclass of pa.ListArray whose values are a StructArray of keys and values, so the check that a
    # list holds structs lets a map through.
    m = pa.array([[("a", 1), ("b", 2)]], type=pa.map_(pa.string(), pa.int64()))
    _, _, datas = arrow_array_adapter(m)
    assert datas["value"].tolist() == [1, 2]


def test_unsupported_arrow_type_names_itself():
    # pyarrow names this type "float", and numpy names its class float32. The word boundary after "float" matches the
    # Arrow name only. The rest of the pattern requires the name of the dictionary that would need an entry for it.
    with pytest.raises(ValueError, match=r"float\b.*arrow_to_numpy_dtypes"):
        uniform_arrow_array_adapter(pa.array([1.0, 2.0], type=pa.float32()))


def test_cjk_string_width_counts_characters_not_bytes():
    for arrow_ty in (pa.string(), pa.large_string()):
        _, data = arrow_array_adapter(pa.array(["\u65e5" * 30, "z"], type=arrow_ty))
        assert data.dtype == np.dtype("<U30"), arrow_ty
        assert data[0] == "\u65e5" * 30 and data[1] == "z"


def test_string_width_is_never_too_small():
    rng = random.Random(20260828)
    alphabets = ["abc", "\u00e9\u00f1", "\u65e5\u672c\u8a9e", "\U0001f600\U0001f680"]
    for _ in range(400):
        alphabet = rng.choice(alphabets)
        values = ["".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 12)))
                  for _ in range(rng.randrange(1, 6))]
        _, data = arrow_array_adapter(pa.array(values, type=pa.string()))
        assert list(data) == values
        assert data.dtype.itemsize // 4 == max(len(v) for v in values) or not any(values)


def test_all_empty_strings():
    _, data = arrow_array_adapter(pa.array(["", "", ""], type=pa.string()))
    assert list(data) == ["", "", ""]


def test_adapter_result_keeps_the_arrow_buffer_alive():
    for arrow_ty, values in [
        (pa.int32(), [1, 2, 3]), (pa.int64(), [1, 2, 3]), (pa.float64(), [1.0, 2.0]),
        (pa.uint8(), [1, 2]),
    ]:
        _, data = arrow_array_adapter(pa.array(values, type=arrow_ty))
        assert owner_of(data) is not None, arrow_ty


def test_adapter_result_survives_an_unbound_source():
    """Check that adapted data stays correct when the source array is never bound to a name.

    pyarrow can free such a source as soon as the adapter returns. A view that does not keep its buffer alive then
    reads freed memory, which can give the right values, wrong values or a crash, depending on the size of the
    allocation.
    """
    src = (
        "import numpy as np, pyarrow as pa\n"
        "from numbarrow.core.adapters import arrow_array_adapter\n"
        "for n in (2, 64, 128, 32768, 1 << 17):\n"
        "    bitmap, data = arrow_array_adapter(pa.array(np.arange(n, dtype=np.int64)))\n"
        "    assert data.tolist() == list(range(n)), n\n"
        "print('ok')\n"
    )
    # A child interpreter runs the sizes, so that a crash fails this test rather than end the test run.
    out = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, env=CHILD_ENV)
    assert out.returncode == 0 and out.stdout.strip() == "ok", (out.returncode, out.stderr[-400:])


def test_zero_length_list_array_with_null_buffers():
    """Check that a zero-length list column whose validity and offsets buffers are null adapts to empty fields.

    pyarrow's full validation accepts such an array.
    """
    src = (
        "import pyarrow as pa\n"
        "from numbarrow.core.adapters import arrow_array_adapter\n"
        "ty = pa.list_(pa.struct([('v', pa.int64())]))\n"
        "arr = pa.Array.from_buffers(ty, 0, [None, None],\n"
        "                            children=[pa.array([], type=pa.struct([('v', pa.int64())]))])\n"
        "arr.validate(full=True)\n"
        "_, _, datas = arrow_array_adapter(arr)\n"
        "assert datas['v'].tolist() == []\n"
        "print('ok')\n"
    )
    # Reading an element of a null offsets buffer crashes the process with a segmentation fault, so the case runs in a
    # child interpreter.
    out = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, env=CHILD_ENV)
    assert out.returncode == 0 and out.stdout.strip() == "ok", (out.returncode, out.stderr[-300:])


def test_zero_length_list_array_shapes():
    ty = pa.list_(pa.struct([("v", pa.int64())]))
    for label, arr in [
        ("empty", pa.array([], type=ty)),
        ("sliced to zero", pa.array([[{"v": 1}]], type=ty).slice(0, 0)),
        ("sliced to zero at the end", pa.array([[{"v": 1}]], type=ty).slice(1, 0)),
    ]:
        _, _, datas = arrow_array_adapter(arr)
        assert datas["v"].tolist() == [], label


def test_every_adapter_path_returns_a_read_only_result():
    """Check that adapted data is read-only whatever the Arrow type.

    The bool, string, large_string and date32 adapters allocate new numpy arrays rather than view the Arrow buffer. A
    new array is writable unless the adapter clears its flag.
    """
    cases = {
        "int32": pa.array([1, 2], type=pa.int32()),
        "int64": pa.array([1, 2], type=pa.int64()),
        "float64": pa.array([1.0, 2.0], type=pa.float64()),
        "uint8": pa.array([1, 2], type=pa.uint8()),
        "bool": pa.array([True, False], type=pa.bool_()),
        "string": pa.array(["a", "b"], type=pa.string()),
        "large_string": pa.array(["a", "b"], type=pa.large_string()),
        "date32": pa.array([1, 2], type=pa.int32()).cast(pa.date32()),
        "date64": pa.array([86400000], type=pa.int64()).cast(pa.date64()),
        "timestamp": pa.array([1], type=pa.int64()).cast(pa.timestamp("ms")),
    }
    writable = [name for name, arr in cases.items()
                if arrow_array_adapter(arr)[1].flags.writeable]
    assert writable == []


def test_read_only_holds_within_one_struct_column():
    # A struct column adapts each field through the adapter of its type: the int64 field becomes a view and the string
    # field a new array. Both come back read-only.
    arr = pa.array([{"i": 1, "s": "a"}],
                   type=pa.struct([("i", pa.int64()), ("s", pa.string())]))
    _, _, datas = arrow_array_adapter(arr)
    assert {name: d.flags.writeable for name, d in datas.items()} == {"i": False, "s": False}


def test_read_only_holds_for_an_empty_array():
    """Check that adapted data is read-only for a zero-length array too, as a column and as a struct field.

    For a zero-length array, each fixed-width, boolean and string adapter returns a new array, which is writable
    unless the adapter clears its flag.
    """
    cases = {
        "int32": pa.array([], type=pa.int32()),
        "int64": pa.array([], type=pa.int64()),
        "float64": pa.array([], type=pa.float64()),
        "uint8": pa.array([], type=pa.uint8()),
        "bool": pa.array([], type=pa.bool_()),
        "string": pa.array([], type=pa.string()),
        "large_string": pa.array([], type=pa.large_string()),
        "date32": pa.array([1], type=pa.int32()).cast(pa.date32()).slice(0, 0),
    }
    writable = [name for name, arr in cases.items()
                if arrow_array_adapter(arr)[1].flags.writeable]
    assert writable == []
    # A struct column adapts each field through the same adapters.
    empty_struct = pa.StructArray.from_arrays(
        [pa.array([], type=pa.int64()), pa.array([], type=pa.string())], ["i", "s"])
    _, _, datas = arrow_array_adapter(empty_struct)
    assert {name: d.flags.writeable for name, d in datas.items()} == {"i": False, "s": False}


def test_unsupported_type_errors_name_the_type_without_the_data():
    """Check that a refusal names the array's type and leaves out its values.

    An array's repr carries every value it holds. A message that interpolated the array would grow with the data and
    copy the caller's values into any log that records the traceback.
    """
    rows = [[i, i + 1] for i in range(500)]
    with pytest.raises(NotImplementedError) as excinfo:
        arrow_array_adapter(pa.array(rows, type=pa.list_(pa.int64())))
    message = str(excinfo.value)
    assert "int64" in message
    assert "499" not in message
    assert len(message) < 200

    with pytest.raises(NotImplementedError) as excinfo:
        arrow_array_adapter(pa.array(["abcdef"] * 500, type=pa.binary()))
    message = str(excinfo.value)
    assert "binary" in message
    assert len(message) < 200

    # The binary case cannot check content: pyarrow renders binary values in hex, so "abcdef" is absent even from a
    # message that interpolates the array. A length bound misses a short array, because three interpolated values keep
    # such a message under 200 characters. decimal128 renders its values as written, so this case checks the content.
    with pytest.raises(NotImplementedError) as excinfo:
        arrow_array_adapter(
            pa.array([Decimal("1234567890.12")] * 3, type=pa.decimal128(20, 2)))
    message = str(excinfo.value)
    assert "decimal128" in message
    assert "1234567890.12" not in message
    assert len(message) < 200


def test_bytes_under_a_null_slot_are_not_decoded():
    # pc.if_else leaves the masked-out value's bytes in the data buffer. Decoding the null slot would return "SECRET",
    # a value the caller masked out.
    values = pa.array(["keepme", "SECRET"], type=pa.string())
    masked = pa.compute.if_else(pa.array([True, False]), values, pa.scalar(None, pa.string()))
    assert masked.to_pylist() == ["keepme", None]
    bitmap, data = arrow_array_adapter(masked)
    assert list(data) == ["keepme", ""]
    assert is_null(1, bitmap) and not is_null(0, bitmap)


def test_width_ignores_a_wide_value_under_a_null_slot():
    values = pa.array(["ab", "x" * 40], type=pa.string())
    masked = pa.compute.if_else(pa.array([True, False]), values, pa.scalar(None, pa.string()))
    _, data = arrow_array_adapter(masked)
    assert data.dtype == np.dtype("<U2")


def test_a_zero_length_array_with_null_buffers_adapts_like_its_empty_twin():
    """Check that a zero-length array with null buffers adapts like the same array with empty buffers.

    Each null-buffered array passes pyarrow's full validation and holds no values, as its twin does, so the two must
    adapt alike. pyarrow refuses to build the non-empty version of this shape, as
    test_pyarrow_refuses_a_non_empty_array_with_no_data_buffer checks, so a pyarrow caller can pass a null data buffer
    to an adapter only at length zero.
    """
    cases = [
        (pa.Array.from_buffers(pa.int64(), 0, [None, None]),
         pa.array([], type=pa.int64())),
        (pa.Array.from_buffers(pa.bool_(), 0, [None, None]),
         pa.array([], type=pa.bool_())),
        (pa.Array.from_buffers(pa.string(), 0, [None, None, pa.py_buffer(b"")]),
         pa.array([], type=pa.string())),
        (pa.Array.from_buffers(pa.large_string(), 0, [None, None, pa.py_buffer(b"")]),
         pa.array([], type=pa.large_string())),
    ]
    for null_buffered, empty_buffered in cases:
        null_buffered.validate(full=True)
        bitmap, data = arrow_array_adapter(null_buffered)
        want_bitmap, want_data = arrow_array_adapter(empty_buffered)
        assert bitmap is want_bitmap is None, null_buffered.type
        assert data.dtype == want_data.dtype, null_buffered.type
        assert len(data) == 0, null_buffered.type
        # The zero-length result is read-only like a non-empty one.
        assert not data.flags.writeable, null_buffered.type


def test_a_non_empty_all_null_array_carries_both_buffers_and_adapts():
    """Check that a non-empty all-null array, whole or sliced, has both buffers and adapts to a full-length result.

    pyarrow allocates a values buffer for an all-null array whatever its bitmap says, so a fully null column never
    reaches the adapters' guard for a missing data buffer. In the sliced cases, the adapter skips the values before and
    after the slice.
    """
    cases = [
        (pa.nulls(5, type=pa.int64()), np.dtype(np.int64)),
        (pa.nulls(5, type=pa.float64()), np.dtype(np.float64)),
        (pa.nulls(5, type=pa.bool_()), np.dtype(np.bool_)),
        (pa.nulls(5, type=pa.int64())[1:4], np.dtype(np.int64)),
        (pa.nulls(5, type=pa.bool_())[1:4], np.dtype(np.bool_)),
    ]
    for arr, want_dtype in cases:
        assert arr.buffers()[1] is not None, arr.type
        bitmap, data = arrow_array_adapter(arr)
        assert bitmap is not None, arr.type
        assert all(is_null(i, bitmap) for i in range(len(arr))), arr.type
        assert data.dtype == want_dtype, arr.type
        assert len(data) == len(arr), arr.type
        # The test does not check the values. The bytes under a null slot belong to no value, so their content is not
        # part of the answer.
        assert not data.flags.writeable, arr.type


def test_pyarrow_refuses_a_non_empty_array_with_no_data_buffer():
    """Check that pyarrow refuses to build a non-empty int64 or bool array without a data buffer.

    This refusal keeps the adapters' guard for a missing data buffer out of reach of a pyarrow caller. pyarrow 14.0.0
    and 24.0.0 also refuse the shape when they import an array through the C Data Interface. If pyarrow starts to
    build it, this test fails, and the guard becomes reachable from pyarrow.
    """
    for ty in (pa.int64(), pa.bool_()):
        validity = pa.nulls(5, type=ty).buffers()[0]
        with pytest.raises(pa.ArrowInvalid, match="Missing values buffer"):
            pa.Array.from_buffers(ty, 5, [validity, None])


def test_a_non_array_reaching_the_dispatcher_still_raises_not_implemented():
    """Check that the dispatcher raises ``NotImplementedError`` for an object that is not a pyarrow Array.

    The dispatcher's base implementation receives every object whose type has no registered adapter. A pa.Scalar and
    a pa.Field have an Arrow type but no length. A list, an ndarray and a str have a length but no Arrow type, and None
    and an int have neither. A message that read a missing attribute would raise ``TypeError`` or ``AttributeError``
    instead, which escapes a caller that catches ``NotImplementedError``.
    """
    for obj in (pa.scalar(5, pa.int64()), pa.field("f", pa.int64())):
        with pytest.raises(NotImplementedError, match="int64"):
            arrow_array_adapter(obj)
    for obj in ([1, 2, 3], np.array([1, 2, 3]), "abc", None, 7):
        with pytest.raises(NotImplementedError, match="not a pyarrow Array"):
            arrow_array_adapter(obj)


def test_repeated_struct_field_names_raise_a_typed_error():
    # Arrow allows a struct to repeat a field name. The adapter returns the fields in dictionaries keyed by name, which
    # cannot hold both, so it refuses such a struct.
    dup = pa.StructArray.from_arrays(
        [pa.array([1, 2], type=pa.int64()), pa.array([3, 4], type=pa.int64())], ["v", "v"])
    with pytest.raises(NotImplementedError, match="repeated field names"):
        arrow_array_adapter(dup)


def test_boolean_offset_crossing_a_byte_boundary():
    values = [bool((i * 7) % 3) for i in range(20)]
    arr = pa.array(values, type=pa.bool_())
    for offset in (1, 7, 8, 9, 15):
        _, data = arrow_array_adapter(arr.slice(offset))
        assert data.tolist() == values[offset:], offset


def test_the_boolean_view_covers_the_offset_as_well_as_the_length(monkeypatch):
    """Check that the boolean adapter's view of the packed bits covers the offset as well as the length.

    The values cannot show an undersized view. numba compiles unpack_booleans without bounds checking by default, so a
    read past the end of the view continues into the pyarrow buffer, which is still valid, and returns the right
    values. The test therefore records the size the adapter asks the uint8 viewer for, and checks it.
    """
    from numbarrow.core import adapters

    sizes = []
    real_viewer = adapters.arrays_viewers[np.uint8]

    def spy(pointer, size):
        sizes.append(size)
        return real_viewer(pointer, size)

    monkeypatch.setitem(adapters.arrays_viewers, np.uint8, spy)
    values = [bool((i * 7) % 3) for i in range(20)]
    arr = pa.array(values, type=pa.bool_())
    for offset in (1, 7, 8, 9, 15):
        sizes.clear()
        sliced = arr.slice(offset)
        _, data = arrow_array_adapter(sliced)
        assert data.tolist() == values[offset:], offset
        # The bits lie at their absolute positions in the buffer, so the view must reach the last of them:
        # ceil((offset + length) / 8) bytes.
        want = (sliced.offset + len(sliced) + 7) // 8
        assert sizes[0] == want, (offset, sizes[0], want)


def test_a_zero_length_array_whose_offset_points_past_its_buffer_adapts():
    """Check that a zero-length int64 array whose offset points past its empty buffer adapts to an empty result.

    np.frombuffer refuses an offset past the end of a buffer with a bare ``ValueError`` that names neither the array
    nor its type. The adapter therefore returns early for a zero-length array, before it reads the buffer.
    """
    arr = pa.Array.from_buffers(pa.int64(), 0, [None, pa.py_buffer(b"")], offset=2)
    arr.validate(full=True)
    bitmap, data = arrow_array_adapter(arr)
    assert bitmap is None
    assert data.dtype == np.dtype("int64")
    assert len(data) == 0
    assert not data.flags.writeable


def test_create_bitmap_owns_its_memory_on_the_offset_path():
    """Check that create_bitmap returns an array that owns its memory on the offset path.

    test_create_str_array_all_null_bitmap_is_a_copy covers the offset-0 path. On the offset path, a slice of the
    ``np.packbits`` output would have that output as its base and keep it alive.
    """
    buf = pa.py_buffer(bytes([0xB5, 0x6C, 0xF0]))
    bitmap = create_bitmap(buf, 3, 13)
    assert bitmap.base is None
    assert bitmap.flags.owndata


def test_result_is_read_only_whatever_the_buffer_provenance():
    """Check that adapted data is read-only both for a locally built array and for the same array read back from IPC.

    np.frombuffer makes a read-only view only of a buffer that pyarrow marks immutable. A buffer read from an Arrow IPC
    stream is immutable, and a locally built one is mutable. Spark sends mapInArrow its batches through Arrow IPC.
    numba types a read-only array differently from a writable one, so a UDF that compiles against a local batch could
    fail on a Spark batch if the two differed.
    """
    col = pa.array([10, 20, 30, 40], type=pa.int64())
    batch = pa.RecordBatch.from_arrays([col], names=["v"])
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    ipc_column = pa.ipc.open_stream(pa.BufferReader(sink.getvalue())).read_next_batch().column("v")
    assert col.buffers()[1].is_mutable and not ipc_column.buffers()[1].is_mutable
    for label, arr in [("local", col), ("ipc", ipc_column)]:
        _, data = arrow_array_adapter(arr)
        assert not data.flags.writeable, label
        assert data.tolist() == [10, 20, 30, 40], label


def test_the_read_only_flag_cannot_be_flipped_back():
    """Check that a caller cannot make an adapted view writable and change the source Arrow array through it.

    numpy refuses to set ``WRITEABLE`` on an array whose base is read-only. Each view rests on a pyarrow buffer that
    exports its memory read-only, so the flip fails as it does on pyarrow's own ``to_numpy(zero_copy_only=True)``.
    """
    views = {
        "int64": pa.array([1, 2, 3], type=pa.int64()),
        "float64": pa.array([1.5, 2.5], type=pa.float64()),
        "date64": pa.array([86400000], type=pa.int64()).cast(pa.date64()),
        "timestamp": pa.array([1, 2], type=pa.int64()).cast(pa.timestamp("ms")),
        "struct child": pa.array([{"f": 1}], type=pa.struct([("f", pa.int64())])),
    }
    for label, arr in views.items():
        adapted = arrow_array_adapter(arr)
        data = adapted[2]["f"] if len(adapted) == 3 else adapted[1]
        before = arr.to_pylist()
        with pytest.raises(ValueError, match="WRITEABLE"):
            data.flags.writeable = True
        assert arr.to_pylist() == before, label


def test_a_bool_array_is_refused_by_the_uniform_adapter():
    # Arrow packs booleans one bit per element, and a view of one byte per element reads wrong values. The uniform
    # adapter refuses a bool array; the dispatcher sends one to the boolean adapter, which unpacks the bits.
    with pytest.raises(ValueError, match="bool"):
        uniform_arrow_array_adapter(pa.array([True, False, True], type=pa.bool_()))


def test_multi_buffer_types_name_themselves_in_the_uniform_adapter():
    # The uniform adapter looks the type up before it unpacks the buffers, so an array with more than two buffers
    # fails with its type in the message rather than with "too many values to unpack".
    for arrow_ty, values in [(pa.string(), ["a"]), (pa.binary(), [b"a"]),
                             (pa.list_(pa.int64()), [[1]])]:
        with pytest.raises(ValueError) as excinfo:
            uniform_arrow_array_adapter(pa.array(values, type=arrow_ty))
        assert str(arrow_ty) in str(excinfo.value), arrow_ty


if __name__ == "__main__":
    test_create_str_array()
    test_create_str_array_long_with_null()
    test_create_str_array_all_null_bitmap_is_a_copy()
    test_structured_array_adapter()
    test_uniform_arrow_array_adapter_1()


def test_a_list_row_of_a_different_length_is_refused():
    """Check that a list column whose rows differ in length is refused, and that rows of equal length adapt.

    The adapter returns the flattened elements without offsets. An element maps back to its row only when every row
    has the same length. An empty row or a row of another length breaks that as a null row does: the three rows of the
    first case would flatten to two elements.
    """
    ty = pa.list_(pa.struct([("v", pa.int64())]))
    for rows in (
        [[{"v": 1}], [], [{"v": 3}]],
        [[{"v": 1}, {"v": 2}], [{"v": 3}]],
        [[{"v": 1}], [{"v": 2}, {"v": 3}, {"v": 4}]],
    ):
        with pytest.raises(NotImplementedError, match="not all the same length"):
            arrow_array_adapter(pa.array(rows, type=ty))
    # Rows of equal length still adapt when each holds more than one element: two rows of two map back exactly.
    _, _, datas = arrow_array_adapter(
        pa.array([[{"v": 1}, {"v": 2}], [{"v": 3}, {"v": 4}]], type=ty))
    assert datas["v"].tolist() == [1, 2, 3, 4]


def test_a_struct_level_null_hides_its_child_value_from_the_string_adapter():
    """Check that the adapter ignores a string hidden by a struct-level null when it checks NULs and sizes the column.

    A row that is null at the struct level leaves the child's own bitmap unchanged, so the child's data still holds
    the hidden value. No caller can see that value: its trailing NUL is no reason to refuse the batch, and its 200
    characters are no measure of the column's width.
    """
    ty = pa.struct([("s", pa.string())])
    hidden_nul = pa.StructArray.from_arrays(
        [pa.array(["ok", "bad\x00", "fine"], type=pa.string())], ["s"],
        mask=pa.array([False, True, False]))
    assert hidden_nul.to_pylist() == [{"s": "ok"}, None, {"s": "fine"}]
    _, _, datas = arrow_array_adapter(hidden_nul)
    assert datas["s"].tolist() == ["ok", "", "fine"]

    hidden_wide = pa.StructArray.from_arrays(
        [pa.array(["ab", "X" * 200, "cd"], type=pa.string())], ["s"],
        mask=pa.array([False, True, False]))
    _, _, datas = arrow_array_adapter(hidden_wide)
    assert datas["s"].dtype == np.dtype("<U2"), (
        "width must come from observable values, not from one hidden under a "
        "struct-level null")
    # The adapter still returns the struct's bitmap and the field's own bitmap separately, as is_null_struct takes them.
    struct_bitmap, bitmaps, _ = arrow_array_adapter(hidden_wide)
    assert struct_bitmap is not None
    assert bitmaps["s"] is None, "the field itself has no nulls of its own"
    assert isinstance(pa.array([{"s": "x"}], type=ty), pa.StructArray)


def test_an_empty_element_never_indexes_the_neighbour_s_dead_bytes():
    """Check that an empty string is not refused over a NUL in bytes outside it.

    Computed as its end minus one, an empty element's last byte is the byte before its start: the previous element's
    last byte, or -1 for the first element, which numpy reads as the last byte of the data. Here that byte is the NUL
    that ends the masked-out value under the null slot. The ``non_empty`` term keeps empty elements out of the
    trailing-NUL check, and this test fails with ``>=`` in place of its ``>``.
    """
    masked = pc.if_else(pa.array([True, False]),
                        pa.array(["", "ab\x00"], type=pa.string()),
                        pa.scalar(None, pa.string()))
    assert masked.to_pylist() == ["", None]
    offsets = np.frombuffer(memoryview(masked.buffers()[1]), dtype=np.int32)[:3]
    assert offsets.tolist() == [0, 0, 3], "the dead bytes must be present to test this"
    _, data = arrow_array_adapter(masked)
    assert data.tolist() == ["", ""]


def test_a_live_empty_element_is_not_judged_by_the_span_s_first_byte():
    """Check that a valid empty string is not refused when the column's data starts with a NUL byte.

    ``last_byte`` is 0 for an empty element, the first byte of the data. If the trailing-NUL check lacked its
    ``non_empty`` term, it would judge an empty value by that byte. A column whose first value starts with a NUL would
    then have every valid empty string refused.
    """
    arr = pa.array(["\x00x", ""], type=pa.string())
    bounds = np.frombuffer(memoryview(arr.buffers()[2]), dtype=np.uint8)
    assert bounds[0] == 0, "this test needs the span to start with a NUL byte"
    _, data = arrow_array_adapter(arr)
    assert data.tolist() == ["\x00x", ""]


def test_a_ragged_map_raises_as_the_readme_says():
    """Check that a map column whose rows hold different numbers of entries is refused, and a uniform one adapts.

    A MapArray is a ListArray, so the rule that every row has the same length binds a map column too. Rows of
    different lengths are the usual shape of a map column, and the flattened entries could not say which row each
    entry belongs to.
    """
    ragged = pa.array([[("a", 1), ("b", 2)], [("c", 3)]],
                      type=pa.map_(pa.string(), pa.int64()))
    with pytest.raises(NotImplementedError, match="not all the same length"):
        arrow_array_adapter(ragged)
    uniform = pa.array([[("a", 1), ("b", 2)], [("c", 3), ("d", 4)]],
                       type=pa.map_(pa.string(), pa.int64()))
    _, _, datas = arrow_array_adapter(uniform)
    assert datas["value"].tolist() == [1, 2, 3, 4]


def test_the_base_of_a_view_cannot_be_released_from_under_it():
    """Check that a view's base has no ``release()``, and that the view keeps its data once the source is dropped.

    Given a memoryview, np.frombuffer keeps another memoryview as the array's base, and its ``release()`` drops the
    hold on the source buffer. A caller who released that base and dropped the source would read freed memory. The
    adapter gives np.frombuffer a pyarrow buffer instead, which has no ``release()``.
    """
    n = 1 << 16

    def view():
        source = pa.array(np.arange(n) * 7 + 3)
        return arrow_array_adapter(source)[1]

    data = view()
    before = data.copy()
    assert not hasattr(data.base, "release")
    with pytest.raises(ValueError, match="WRITEABLE"):
        data.flags.writeable = True
    gc.collect()
    # Buffers of the source's size, filled with 0xA5, can take over its freed memory, so a view of that memory would
    # read changed values.
    junk = [pa.allocate_buffer(n * 8) for _ in range(6)]
    for buffer in junk:
        np.frombuffer(buffer, dtype=np.uint8)[:] = 0xA5
    assert np.array_equal(data, before)
