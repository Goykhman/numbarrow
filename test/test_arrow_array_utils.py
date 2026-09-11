import gc
import random
import subprocess
import sys
from decimal import Decimal

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


def owner_of(array):
    """Walk an array's base chain and return the pyarrow object backing it."""
    node = array
    for _ in range(8):
        node = node.obj if isinstance(node, memoryview) else getattr(node, "base", None)
        if node is None:
            return None
        if isinstance(node, (pa.Buffer, pa.Array)):
            return node
    return None


def test_create_str_array():
    pa_a = pa.array(["first", "second", None, "third", "fourth element", "f"], type=pa.string())
    bitmap, np_a = create_str_array(pa_a)
    ref = ["first", "second", "", "third", "fourth element", "f"]
    assert is_null(2, bitmap) and all(not is_null(i, bitmap) for i in range(len(pa_a)) if i != 2)
    # zip() stops at the shorter side, so a truncated result used to pass.
    assert len(np_a) == len(ref)
    assert list(np_a) == ref


def test_create_str_array_long_with_null():
    # Bitmap spans two bytes once the array has 9+ elements, so this pins
    # that the copy guard never calls bool() on a multi-byte bitmap.
    vals = ["s%d" % i for i in range(9)] + [None]
    bitmap, np_a = create_str_array(pa.array(vals, type=pa.string()))
    assert is_null(9, bitmap) and all(not is_null(i, bitmap) for i in range(9))
    assert np_a[0] == "s0" and np_a[8] == "s8" and np_a[9] == ""
    sliced = pa.array(vals + ["tail"], type=pa.string())[1:]
    bitmap_s, np_s = create_str_array(sliced)
    assert is_null(8, bitmap_s) and not is_null(9, bitmap_s)


def test_create_str_array_all_null_bitmap_is_a_copy():
    # An all-null bitmap byte is zero (falsy); the copy must still happen
    # so the returned bitmap owns its memory instead of viewing the Arrow
    # buffer, which the caller may let die.
    bitmap, np_a = create_str_array(pa.array([None] * 8, type=pa.string()))
    assert all(is_null(i, bitmap) for i in range(8))
    assert bitmap.base is None


def test_structured_array_adapter():
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
    # Checked by mutating rather than by walking .base: a view built over a raw
    # address also has no pyarrow object in its base chain, so a base-chain
    # assertion would pass against the very code this guards.
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
    # The offset path returns np.packbits' output directly, which is only
    # correct because the bits handed to it were padded to a whole number of
    # bytes, so it already produces exactly (length + 7) // 8 of them and owns
    # them. Changing the padding would silently return a longer bitmap.
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
    # Same guarantee as the offset-0 path, reached a different way: nothing
    # Arrow-owned backs this one, so it holds without a copy.
    source = pa.array([1, None, 3, 4, 5, None, 7, 8, 9, 10], type=pa.int64())
    bitmap = create_bitmap(source.buffers()[0], 3, 6)
    before = bitmap.tolist()
    del source
    gc.collect()
    filler = [pa.array(list(range(64)), type=pa.int64()) for _ in range(200)]
    after = bitmap.tolist()
    del filler
    assert after == before


def test_sliced_list_returns_only_its_own_rows():
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
        # Every producer above is a trailing slice, where offsets[n] happens to
        # equal len(values), so the upper bound of the slice is never tested.
        "middle row only": larr.slice(1, 1),
        "leading two rows": larr.slice(0, 2),
    }
    for name, arr in producers.items():
        _, _, datas = arrow_array_adapter(arr)
        want = [element["v"] for row in arr.to_pylist() for element in row]
        assert datas["v"].tolist() == want, name


def test_sliced_list_sees_the_slice_null_count_not_the_parent_s():
    # The guard must read the slice's own null count. A slice that still
    # contains the null row is refused; one that excludes it adapts normally,
    # which also keeps the offset handling this case was written to cover.
    ty = pa.list_(pa.struct([("v", pa.int64())]))
    larr = pa.array([[{"v": 1}], None, [{"v": 3}]], type=ty)
    with pytest.raises(NotImplementedError, match="null row"):
        arrow_array_adapter(larr.slice(1, 2))
    _, _, datas = arrow_array_adapter(larr.slice(2, 1))
    assert datas["v"].tolist() == [3]


def test_a_trailing_nul_is_refused_rather_than_truncated():
    # numpy pads a short |U element with NUL, so a trailing NUL in the value is
    # indistinguishable from that padding: "ab\x00" used to come back as "ab".
    # The element index is asserted, not just the phrase: a wide batch is
    # unusable if the message points at the wrong element, and matching only on
    # "ends in a NUL" leaves a hard-coded index undetected.
    for values, index_ in ((["ab\x00", "abc"], 0), (["ab\x00\x00"], 0), (["ok", "x\x00"], 1)):
        pattern = rf"element {index_} of a {len(values)}-element string array"
        with pytest.raises(ValueError, match=pattern):
            arrow_array_adapter(pa.array(values, type=pa.string()))
        with pytest.raises(ValueError, match=pattern):
            arrow_array_adapter(pa.array(values, type=pa.large_string()))
    # Leading and interior NULs are representable and stay supported.
    _, data = arrow_array_adapter(pa.array(["a\x00b", "\x00ab", "abc"], type=pa.string()))
    assert data.tolist() == ["a\x00b", "\x00ab", "abc"]
    # A null slot is never decoded, so dead bytes ending in NUL do not trip it.
    # Built with pc.if_else, which leaves the masked-out bytes in the buffer:
    # .take() emits a byte-EMPTY null slot (offsets [0, 0, 2]), so the guard
    # short-circuits on non_empty and the `live` term is never reached.
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
    # The old guard was a bare assert, which -O strips; the input then reached
    # a struct-field loop and raised a TypeError about pyarrow.lib.DataType.
    src = (
        "import pyarrow as pa\n"
        "from numbarrow.core.adapters import arrow_array_adapter\n"
        "try:\n"
        "    arrow_array_adapter(pa.array([[1, 2], [3]], pa.list_(pa.int64())))\n"
        "except NotImplementedError as exc:\n"
        "    print('typed', 'int64' in str(exc))\n"
    )
    out = subprocess.run([sys.executable, "-O", "-c", src], capture_output=True, text=True)
    assert out.stdout.strip() == "typed True", out


def test_map_column_still_adapts():
    # pa.MapArray is a pa.ListArray whose values are a key/value StructArray,
    # so the struct guard must let it through.
    m = pa.array([[("a", 1), ("b", 2)]], type=pa.map_(pa.string(), pa.int64()))
    _, _, datas = arrow_array_adapter(m)
    assert datas["value"].tolist() == [1, 2]


def test_unsupported_arrow_type_names_itself():
    # Matches the arrow type name and the map that would have to gain it. The
    # pre-fix message named the numpy class, so a bare "float" match passed
    # against the very code this guards.
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
    # The source is never bound to a name, so it is collectable the moment the
    # adapter returns. A view that owns nothing reads freed memory: correct at
    # tiny sizes, wrong in the middle, and a segfault once large enough.
    src = (
        "import numpy as np, pyarrow as pa\n"
        "from numbarrow.core.adapters import arrow_array_adapter\n"
        "for n in (2, 64, 128, 32768, 1 << 17):\n"
        "    bitmap, data = arrow_array_adapter(pa.array(np.arange(n, dtype=np.int64)))\n"
        "    assert data.tolist() == list(range(n)), n\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == "ok", (out.returncode, out.stderr[-400:])


def test_zero_length_list_array_with_null_buffers():
    # The Arrow spec lets a zero-length array carry null buffers and pyarrow's
    # own full validation accepts one. Indexing a null offsets buffer
    # segfaults rather than raising, which would take the whole process down,
    # so this runs out of process.
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
    out = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True)
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
    # The contract must not depend on which Arrow type the caller passed. bool,
    # string, large_string and date32 allocate fresh numpy arrays rather than
    # viewing the Arrow buffer, and all four came back writable.
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
    # The split used to land inside a single column: an int64 field read-only
    # and a string field writable.
    arr = pa.array([{"i": 1, "s": "a"}],
                   type=pa.struct([("i", pa.int64()), ("s", pa.string())]))
    _, _, datas = arrow_array_adapter(arr)
    assert {name: d.flags.writeable for name, d in datas.items()} == {"i": False, "s": False}


def test_read_only_holds_for_an_empty_array():
    # The contract must not depend on the batch's length either. Every
    # fixed-width and string path returns a fresh np.empty for a zero-length
    # array, and each of those came back writable while its non-empty
    # counterpart did not.
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
    # ...and through a struct column, which reaches the same paths per field.
    empty_struct = pa.StructArray.from_arrays(
        [pa.array([], type=pa.int64()), pa.array([], type=pa.string())], ["i", "s"])
    _, _, datas = arrow_array_adapter(empty_struct)
    assert {name: d.flags.writeable for name, d in datas.items()} == {"i": False, "s": False}


def test_unsupported_type_errors_name_the_type_without_the_data():
    # Both messages used to interpolate the array itself, whose repr carries
    # every value it holds: unbounded in size and a copy of the caller's data
    # in whatever log catches the traceback.
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

    # `"abcdef" not in message` cannot fail on a binary array: pyarrow renders
    # its elements as hex, so the interpolating message did not contain it
    # either, which left `len(message) < 200` doing all the work. A length
    # bound is not a content bound: at three elements the interpolating message
    # was 132 chars and passed every assertion above. decimal128 renders
    # human-readably, so it discriminates at any size.
    with pytest.raises(NotImplementedError) as excinfo:
        arrow_array_adapter(
            pa.array([Decimal("1234567890.12")] * 3, type=pa.decimal128(20, 2)))
    message = str(excinfo.value)
    assert "decimal128" in message
    assert "1234567890.12" not in message
    assert len(message) < 200


def test_bytes_under_a_null_slot_are_not_decoded():
    # pc.if_else leaves the masked-out value's bytes in place, so decoding the
    # slot hands back a value the caller explicitly masked out.
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
    # The Arrow spec lets a zero-length array carry null buffers and pyarrow's
    # own full validation accepts one, so it is the same logical array as the
    # one with 0-byte buffers and has to get the same answer. pyarrow refuses
    # to build the non-empty version of this shape, so length 0 is the only way
    # a null data buffer can reach an adapter at all.
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
        # Read-only on this path too, so the contract does not depend on
        # whether the batch happened to be empty.
        assert not data.flags.writeable, null_buffered.type


def test_a_non_empty_all_null_array_carries_both_buffers_and_adapts():
    # The complement of the zero-length case above. An all-null array is not a
    # buffer-less one: pyarrow allocates the values buffer whatever the validity
    # bitmap says, so a fully null column arrives with both buffers present and
    # never reaches the missing-data-buffer guard. Sliced as well, where the
    # values the adapter must skip sit before the start of the logical array.
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
        # Only the shape is pinned. The bytes under a null slot belong to no
        # logical value, so what they happen to hold is not part of the answer.
        assert not data.flags.writeable, arr.type


def test_pyarrow_refuses_a_non_empty_array_with_no_data_buffer():
    # What keeps the missing-data-buffer branch out of reach of a pyarrow
    # caller, asserted rather than assumed. The adapters keep the guard because
    # a foreign producer can still hand that shape over the C Data Interface; if
    # pyarrow ever starts building it, the branch becomes reachable from an
    # ordinary array and turns from a typed error into a wrong refusal.
    for ty in (pa.int64(), pa.bool_()):
        validity = pa.nulls(5, type=ty).buffers()[0]
        with pytest.raises(pa.ArrowInvalid, match="Missing values buffer"):
            pa.Array.from_buffers(ty, 5, [validity, None])


def test_a_non_array_reaching_the_dispatcher_still_raises_not_implemented():
    # The fallback is the base singledispatch implementation, so it receives
    # anything that is not a registered Array. A pa.Scalar and a pa.Field have
    # a type but no length; building the message with len() turned the
    # documented NotImplementedError into a TypeError that escaped any caller
    # wrapping this in `except NotImplementedError`.
    for obj in (pa.scalar(5, pa.int64()), pa.field("f", pa.int64())):
        with pytest.raises(NotImplementedError, match="int64"):
            arrow_array_adapter(obj)
    # The mirror case: an object that is not a pyarrow type at all has a length
    # but no `type`, and reading it raised AttributeError past the same caller.
    for obj in ([1, 2, 3], np.array([1, 2, 3]), "abc", None, 7):
        with pytest.raises(NotImplementedError, match="not a pyarrow Array"):
            arrow_array_adapter(obj)


def test_repeated_struct_field_names_raise_a_typed_error():
    # Arrow permits them; keying by name cannot represent them, and
    # StructArray.field(name) raised a bare KeyError naming nothing.
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
    # The values alone cannot catch an undersized view: unpack_booleans is
    # @njit with an explicit signature, so numba does not bounds-check it and
    # the read runs off the end of the view into the still-valid pyarrow buffer,
    # returning the right answer. Sizing the view from the length alone, which
    # ignores the offset, therefore leaves every value assertion green. So the
    # size the adapter asks for is recorded and checked directly.
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
        # The bits sit at their absolute positions in the buffer, so the view
        # has to reach the last one: ceil((offset + length) / 8).
        want = (sliced.offset + len(sliced) + 7) // 8
        assert sizes[0] == want, (offset, sizes[0], want)


def test_a_zero_length_array_whose_offset_points_past_its_buffer_adapts():
    # The shape the zero-length early return exists for. np.frombuffer would
    # raise a bare "buffer is smaller than requested size (0)" naming neither
    # the array nor the type; every other zero-length array in the suite has a
    # buffer big enough for the read, so nothing else reaches the branch.
    arr = pa.Array.from_buffers(pa.int64(), 0, [None, pa.py_buffer(b"")], offset=2)
    arr.validate(full=True)
    bitmap, data = arrow_array_adapter(arr)
    assert bitmap is None
    assert data.dtype == np.dtype("int64")
    assert len(data) == 0
    assert not data.flags.writeable


def test_create_bitmap_owns_its_memory_on_the_offset_path():
    # create_bitmap's docstring promises the result always owns its memory. The
    # all-null test next to it uses an offset-0 array, so it pins only that
    # path; a slice of the packbits array on the offset path keeps its parent
    # alive and went unnoticed.
    buf = pa.py_buffer(bytes([0xB5, 0x6C, 0xF0]))
    bitmap = create_bitmap(buf, 3, 13)
    assert bitmap.base is None
    assert bitmap.flags.owndata


def test_result_is_read_only_whatever_the_buffer_provenance():
    # np.frombuffer reports read-only only for buffers pyarrow marks immutable,
    # which is true after an IPC round trip and false for a locally built
    # array. The result must not differ between the two, or a UDF compiles in a
    # unit test and fails on a real Spark batch.
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
    # A flag is the caller's to flip: `data.flags.writeable = True` used to
    # succeed on a view over a locally built array, and a store through it then
    # changed the source Arrow array. pyarrow's own to_numpy(zero_copy_only=True)
    # refuses the flip, and so does a view whose base is a read-only memoryview.
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
    # Arrow packs booleans one bit per element, so there is no uniform view.
    # The old dtype lookup succeeded and returned wrong values.
    with pytest.raises(ValueError, match="bool"):
        uniform_arrow_array_adapter(pa.array([True, False, True], type=pa.bool_()))


def test_multi_buffer_types_name_themselves_in_the_uniform_adapter():
    # Resolved before the buffers are unpacked, so a three-buffer layout
    # reports its type instead of "too many values to unpack".
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
    # The result is the flattened elements with no offsets, so element i only
    # maps back to a row when every row is the same length. An empty row and a
    # ragged row break that exactly as a null row does, and used to be answered
    # with silently misaligned data: three rows came back as two elements.
    ty = pa.list_(pa.struct([("v", pa.int64())]))
    for rows in (
        [[{"v": 1}], [], [{"v": 3}]],
        [[{"v": 1}, {"v": 2}], [{"v": 3}]],
        [[{"v": 1}], [{"v": 2}, {"v": 3}, {"v": 4}]],
    ):
        with pytest.raises(NotImplementedError, match="not all the same length"):
            arrow_array_adapter(pa.array(rows, type=ty))
    # Uniform rows of width > 1 stay supported: 2 rows of 2 is well defined.
    _, _, datas = arrow_array_adapter(
        pa.array([[{"v": 1}, {"v": 2}], [{"v": 3}, {"v": 4}]], type=ty))
    assert datas["v"].tolist() == [1, 2, 3, 4]


def test_a_struct_level_null_hides_its_child_value_from_the_string_adapter():
    # A row null at the struct layer leaves its child's own bitmap untouched, so
    # the string adapter used to treat the hidden bytes as a live value: it
    # refused the batch over a trailing NUL no caller can observe, and sized the
    # column's width from a value no caller can observe.
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
    # The two validity layers stay separate: this function reports the field's
    # own bitmap and the struct's, and is_null_struct takes both.
    struct_bitmap, bitmaps, _ = arrow_array_adapter(hidden_wide)
    assert struct_bitmap is not None
    assert bitmaps["s"] is None, "the field itself has no nulls of its own"
    assert isinstance(pa.array([{"s": "x"}], type=ty), pa.StructArray)


def test_an_empty_element_never_indexes_the_neighbour_s_dead_bytes():
    # `non_empty` keeps an empty element out of the trailing-NUL scan. Without
    # it, or with `>=` instead of `>`, the empty element's last byte index walks
    # backwards into the previous element, or to -1 which numpy wraps to the end
    # of the span, and the guard fires on an empty string.
    masked = pc.if_else(pa.array([True, False]),
                        pa.array(["", "ab\x00"], type=pa.string()),
                        pa.scalar(None, pa.string()))
    assert masked.to_pylist() == ["", None]
    offsets = np.frombuffer(memoryview(masked.buffers()[1]), dtype=np.int32)[:3]
    assert offsets.tolist() == [0, 0, 3], "the dead bytes must be present to test this"
    _, data = arrow_array_adapter(masked)
    assert data.tolist() == ["", ""]


def test_a_live_empty_element_is_not_judged_by_the_span_s_first_byte():
    # `last_byte` is 0 for an empty element, so without the `non_empty` term in
    # the trailing-NUL scan an empty value is judged by the FIRST byte of the
    # span. When the column's first value begins with a NUL, that byte is 0 and
    # every live empty string in the column is refused.
    arr = pa.array(["\x00x", ""], type=pa.string())
    bounds = np.frombuffer(memoryview(arr.buffers()[2]), dtype=np.uint8)
    assert bounds[0] == 0, "this test needs the span to start with a NUL byte"
    _, data = arrow_array_adapter(arr)
    assert data.tolist() == ["\x00x", ""]


def test_a_ragged_map_raises_as_the_readme_says():
    # A MapArray is a ListArray, so the same-length contract binds it too. The
    # README documents this explicitly because a ragged map is the usual shape
    # and it used to be answered with entries no row could claim.
    ragged = pa.array([[("a", 1), ("b", 2)], [("c", 3)]],
                      type=pa.map_(pa.string(), pa.int64()))
    with pytest.raises(NotImplementedError, match="not all the same length"):
        arrow_array_adapter(ragged)
    uniform = pa.array([[("a", 1), ("b", 2)], [("c", 3), ("d", 4)]],
                       type=pa.map_(pa.string(), pa.int64()))
    _, _, datas = arrow_array_adapter(uniform)
    assert datas["value"].tolist() == [1, 2, 3, 4]
