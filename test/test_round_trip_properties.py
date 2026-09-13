"""Round-trip invariants over the whole mapInArrow path.

The adapter side was tested in isolation and the output side was not tested at
all, which is how a trailing NUL came to be refused on the way in while leading
and interior NULs were silently destroyed on the way out. A value that a
pass-through UDF returns unchanged must arrive unchanged, and the column a UDF
names must be the column that arrives. Both are properties of the path, not of
either end, so they are asserted here rather than in the adapter tests.
"""
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from numbarrow.core.mapinarrow_factory import make_mapinarrow_func


def _identity(data_dict, bitmap_dict, broadcasts):
    return {name: value for name, value in data_dict.items()}


def _round_trip(column, udf=_identity, name="c"):
    batch = pa.RecordBatch.from_arrays([column], names=[name])
    fn = make_mapinarrow_func(udf, input_columns=[name])
    return list(fn(iter([batch])))[0]


# Every shape a supported column can take, with no nulls so that value
# preservation is isolated from null handling.
COLUMNS = {
    "int64": pa.array([1, -2, 3], type=pa.int64()),
    "int32": pa.array([1, -2], type=pa.int32()),
    "float64": pa.array([1.5, -0.25], type=pa.float64()),
    "uint8": pa.array([0, 255], type=pa.uint8()),
    "bool": pa.array([True, False, True], type=pa.bool_()),
    "ascii": pa.array(["a", "bb", "ccc"], type=pa.string()),
    "empty strings": pa.array(["", "", ""], type=pa.string()),
    "cjk": pa.array(["中文", "字"], type=pa.string()),
    "astral": pa.array(["\U0001F600x", "y"], type=pa.string()),
    "interior nul": pa.array(["a\x00b", "plain"], type=pa.string()),
    "leading nul": pa.array(["\x00ab", "plain"], type=pa.string()),
    "large_string": pa.array(["a", "bb"], type=pa.large_string()),
    "date32": pa.array([1, -2], type=pa.int32()).cast(pa.date32()),
    "date64": pa.array([86400000, 172800000], type=pa.int64()).cast(pa.date64()),
    "timestamp": pa.array([1, 2], type=pa.int64()).cast(pa.timestamp("us")),
    "timestamp tz": pa.array([1, 2], type=pa.int64()).cast(pa.timestamp("us", "UTC")),
}

# Where the type that comes back differs from the type that went in, and why:
# the numpy dtype is all the output side has, so a large_string is a string,
# a date64 is the timestamp[ms] its int64 milliseconds imply, and a zoned
# timestamp is naive. output_schema restores each, which is tested below.
DRIFT = {
    "large_string": pa.string(),
    "date64": pa.timestamp("ms"),
    "timestamp tz": pa.timestamp("us"),
}


@pytest.mark.parametrize("label", sorted(COLUMNS))
def test_a_value_survives_the_round_trip(label):
    # The type is asserted too: a value comparison alone could not see a
    # column change type on the way through.
    column = COLUMNS[label]
    got = _round_trip(column).column("c")
    assert got.type == DRIFT.get(label, column.type), f"{label}: {column.type} came back {got.type}"
    expected = column.cast(got.type).to_pylist() if label in DRIFT else column.to_pylist()
    assert got.to_pylist() == expected, (
        f"{label}: went in as {column.to_pylist()!r}, came out as {got.to_pylist()!r}"
    )


@pytest.mark.parametrize("label", sorted(DRIFT))
def test_output_schema_restores_a_drifted_type(label):
    column = COLUMNS[label]
    batch = pa.RecordBatch.from_arrays([column], names=["c"])
    fn = make_mapinarrow_func(_identity, input_columns=["c"], output_schema=pa.schema([("c", column.type)]))
    got = list(fn(iter([batch])))[0].column("c")
    assert got.type == column.type and got.to_pylist() == column.to_pylist(), label


def test_a_udf_names_its_output_columns():
    # Returned columns are bound by name, not by the order the dict happens to
    # iterate in, or two same-typed columns silently swap.
    def swap(data_dict, bitmap_dict, broadcasts):
        return {"second": np.array([10, 20], dtype=np.int64),
                "first": np.array([1, 2], dtype=np.int64)}

    batch = pa.RecordBatch.from_arrays([pa.array([0, 0], type=pa.int64())], names=["c"])
    out = list(make_mapinarrow_func(swap, input_columns=["c"])(iter([batch])))[0]
    assert out.column("first").to_pylist() == [1, 2]
    assert out.column("second").to_pylist() == [10, 20]


def test_a_string_column_keeps_its_width_across_the_round_trip():
    # The |U width is data-dependent, so a column whose widest value is under a
    # null must not size the output from a value no caller can see.
    #
    # The null slot has to still hold that value's bytes for this to test
    # anything: pa.array(["ab", None, "cd"]) leaves the offsets at [0, 2, 2, 4],
    # so nothing is hidden under the null and any width passes. if_else against
    # a null scalar keeps the wide value's bytes in place and only clears the
    # validity bit.
    values = pa.array(["ab", "x" * 40, "cd"], type=pa.string())
    column = pc.if_else(pa.array([True, False, True]), values,
                        pa.scalar(None, pa.string()))
    seen = {}

    def record_width(data_dict, bitmap_dict, broadcasts):
        seen["dtype"] = data_dict["c"].dtype
        return {name: value for name, value in data_dict.items()}

    got = _round_trip(column, udf=record_width).column("c")
    assert seen["dtype"] == np.dtype("<U2"), seen["dtype"]
    assert [v for v in got.to_pylist() if v] == ["ab", "cd"]


def _pair_identity(data_dict, bitmap_dict, broadcasts):
    return {name: (value, bitmap_dict[name]) for name, value in data_dict.items()}


@pytest.mark.parametrize("label", sorted(COLUMNS))
def test_a_null_survives_the_round_trip_through_a_pair(label):
    # The second row is masked out with its bytes left in place, which is what
    # a bare pass-through republishes; the pair carries the validity out.
    column = COLUMNS[label]
    keep = pa.array([i != 1 for i in range(len(column))])
    nulled = pc.if_else(keep, column, pa.scalar(None, column.type))
    got = _round_trip(nulled, udf=_pair_identity).column("c")
    assert got.type == DRIFT.get(label, column.type), f"{label}: {column.type} came back {got.type}"
    assert got.null_count == 1 and got.to_pylist() == nulled.cast(got.type).to_pylist(), (
        f"{label}: went in as {nulled.to_pylist()!r}, came out as {got.to_pylist()!r}"
    )
