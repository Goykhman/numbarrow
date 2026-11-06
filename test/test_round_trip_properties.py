"""Tests that a column comes out of a mapInArrow function with the values and nulls it went in with.

The tests run one-column batches through ``make_mapinarrow_func``, most with a UDF that returns its input. They check
that each value comes out as it went in, and that each output column is the one the UDF named. The columns cover each
type the adapters support apart from structs and lists, among them strings with non-ASCII characters and with a
leading or an interior NUL.
The adapter refuses only a trailing NUL, so a leading or an interior NUL reaches the output side, where ``pa.array``
given the numpy unicode array would cut the value at that NUL. Both properties belong to the whole path rather than
to either end; the adapter tests check the input side alone. No Spark session runs here, and
``test_mapinarrow_spark.py`` runs the path under Spark.
"""
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from numbarrow.core.mapinarrow_factory import Nullable, make_mapinarrow_func


def _identity(data_dict, bitmap_dict, broadcasts):
    return {name: value for name, value in data_dict.items()}


def _round_trip(column, udf=_identity, name="c"):
    batch = pa.RecordBatch.from_arrays([column], names=[name])
    fn = make_mapinarrow_func(udf, input_columns=[name])
    return list(fn(iter([batch])))[0]


# One column of each type the adapters support, apart from struct and list columns. The columns hold no nulls, so that
# a changed value cannot be blamed on null handling.
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

# The columns whose type comes back changed, and the type they come back as. The output side has only the numpy dtype
# to go on: a large_string comes back a string, a date64 comes back the timestamp[ms] of its datetime64[ms] values, and
# a zoned timestamp comes back naive. output_schema restores each, as test_output_schema_restores_a_drifted_type checks.
DRIFT = {
    "large_string": pa.string(),
    "date64": pa.timestamp("ms"),
    "timestamp tz": pa.timestamp("us"),
}


@pytest.mark.parametrize("label", sorted(COLUMNS))
def test_a_value_survives_the_round_trip(label):
    # The type is checked as well, because a comparison of values cannot see a column change type on the way.
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
    # The output batch takes each column's name from its key in the returned dict, not from the dict's order. Bound by
    # order, two columns of one type could swap places with no error to show it.
    def swap(data_dict, bitmap_dict, broadcasts):
        return {"second": np.array([10, 20], dtype=np.int64),
                "first": np.array([1, 2], dtype=np.int64)}

    batch = pa.RecordBatch.from_arrays([pa.array([0, 0], type=pa.int64())], names=["c"])
    out = list(make_mapinarrow_func(swap, input_columns=["c"])(iter([batch])))[0]
    assert out.column("first").to_pylist() == [1, 2]
    assert out.column("second").to_pylist() == [10, 20]


def test_a_string_column_keeps_its_width_across_the_round_trip():
    # The adapter sets the width of a string column's numpy dtype from its widest non-null value. The widest value
    # here, 40 characters, lies under a null, so the UDF must receive a <U2 array, and the two non-null values must
    # come back.
    #
    # The null slot must still hold the wide value's bytes for the test to mean anything. pa.array(["ab", None, "cd"])
    # gives offsets [0, 2, 2, 4], with nothing under the null, and any width would pass. pc.if_else against a null
    # scalar keeps the wide value's bytes and clears only the validity bit.
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


def _nullable_identity(data_dict, bitmap_dict, broadcasts):
    return {name: Nullable(value, bitmap_dict[name]) for name, value in data_dict.items()}


@pytest.mark.parametrize("label", sorted(COLUMNS))
def test_a_null_survives_the_round_trip_through_nullable(label):
    # pc.if_else nulls the second row. A UDF that returned the bare array would bring the row back as a value, because
    # a bare array carries no nulls out other than a NaT. Nullable carries the bitmap out with the data.
    column = COLUMNS[label]
    keep = pa.array([i != 1 for i in range(len(column))])
    nulled = pc.if_else(keep, column, pa.scalar(None, column.type))
    got = _round_trip(nulled, udf=_nullable_identity).column("c")
    assert got.type == DRIFT.get(label, column.type), f"{label}: {column.type} came back {got.type}"
    assert got.null_count == 1 and got.to_pylist() == nulled.cast(got.type).to_pylist(), (
        f"{label}: went in as {nulled.to_pylist()!r}, came out as {got.to_pylist()!r}"
    )
