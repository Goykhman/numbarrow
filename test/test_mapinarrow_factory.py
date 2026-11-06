"""Tests of ``make_mapinarrow_func`` on pyarrow record batches built in Python, without Spark.

The input tests check what the UDF receives. ``input_columns`` selects and orders the columns, and a struct or
list-of-struct column arrives as a dict of its fields. The factory folds a struct's validity into each field's bitmap:
the bitmap marks a row null when the field or the whole struct row is null. The output tests check how the dict that
the UDF returns becomes a record batch. They cover binding to ``output_schema`` by name and the conversion of strings,
bytes, ``datetime64`` units, record arrays and maps. They also check that the factory refuses a dict key or a field
name that no declared struct field has. The test names call this check the key check for rows and the field guard for
ready-built arrays. The :class:`Nullable` tests check that a returned bitmap becomes the column's validity, and that a
bitmap from ``bitmap_dict`` is accepted only on a column of the row count it covers. Tests that need pandas skip
without it, and the cases that need ``pa.opaque`` or ``pa.list_view`` are left out on a pyarrow without them.
``test_adapters.py`` tests the adapter alone, ``test_mapinarrow_spark.py`` runs the factory inside Spark,
``test_messages.py`` checks the wording of its errors, and ``test_round_trip_properties.py`` checks that values survive
the whole path.
"""
import collections
import datetime
import gc
import weakref

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from numbarrow.core.is_null import is_null
from numbarrow.core.mapinarrow_factory import Nullable, make_mapinarrow_func
from test.conftest import Wrapped


def run_batch(batch, input_columns=None):
    """Run ``batch`` through the factory and return what its UDF received.

    The result maps ``"data"`` to the UDF's ``data_dict`` and ``"bitmap"`` to its ``bitmap_dict``.
    """
    seen = {}

    def main(data_dict, bitmap_dict, broadcasts):
        seen["data"] = data_dict
        seen["bitmap"] = bitmap_dict
        return {"out": np.zeros(1, dtype=np.int64)}

    list(make_mapinarrow_func(main, input_columns=input_columns)(iter([batch])))
    return seen


def nulls(bitmap, n):
    return [is_null(i, bitmap) for i in range(n)]


def test_struct_null_row_reaches_the_udf():
    """A null row of a struct column is null in each field's bitmap.

    ``pa.array``, ``from_pylist`` and ``StructArray.from_arrays(mask=...)`` all build a null struct row over a child
    with no validity buffer, so the child alone cannot show the null. The factory folds the struct's validity into each
    field's bitmap, so one ``is_null`` call per field sees the null row.
    """
    col = pa.array([{"v": 10}, None, {"v": 30}], type=pa.struct([pa.field("v", pa.int64())]))
    seen = run_batch(pa.RecordBatch.from_arrays([col], names=["s"]))
    assert sorted(seen["bitmap"]) == ["s"] and sorted(seen["bitmap"]["s"]) == ["v"]
    assert nulls(seen["bitmap"]["s"]["v"], 3) == [False, True, False]


def test_no_bitmap_when_nothing_is_null_at_either_layer():
    """A struct column with no null row and no null field gives each field a ``None`` bitmap under its own key."""
    col = pa.array([{"v": 10}, {"v": 30}], type=pa.struct([pa.field("v", pa.int64())]))
    seen = run_batch(pa.RecordBatch.from_arrays([col], names=["s"]))
    assert "v" in seen["bitmap"]["s"] and seen["bitmap"]["s"]["v"] is None
    assert seen["data"]["s"]["v"].tolist() == [10, 30]


def test_both_null_layers_are_folded_together():
    """A field's bitmap is null where the field is null and where the whole struct row is null."""
    inner = pa.array([1, None, 3], type=pa.int64())
    col = pa.StructArray.from_arrays([inner], ["v"], mask=pa.array([False, False, True]))
    seen = run_batch(pa.RecordBatch.from_arrays([col], names=["s"]))
    #                      valid, null field, null row
    assert nulls(seen["bitmap"]["s"]["v"], 3) == [False, True, True]


def test_a_field_named_after_its_own_column_is_reachable():
    col = pa.array([{"s": 1}], type=pa.struct([("s", pa.int64())]))
    seen = run_batch(pa.RecordBatch.from_arrays([col], names=["s"]))
    assert seen["data"]["s"]["s"].tolist() == [1]


def test_struct_column_named_after_another_columns_field():
    """A struct column named like a field of another struct column keeps its own data, in either column order.

    ``data_dict`` holds each struct's fields under the name of its column, so the column ``region`` and the field
    ``store.region`` never share a key.
    """
    region = pa.array([{"code": 1}, {"code": 2}], type=pa.struct([("code", pa.int64())]))
    store = pa.array([{"region": 10, "sqft": 1.0}, {"region": 20, "sqft": 2.0}],
                     type=pa.struct([("region", pa.int64()), ("sqft", pa.float64())]))
    batch = pa.RecordBatch.from_arrays([region, store], names=["region", "store"])
    for order in (["region", "store"], ["store", "region"]):
        seen = run_batch(batch, input_columns=order)
        assert list(seen["data"]) == order
        assert seen["data"]["region"]["code"].tolist() == [1, 2]
        assert seen["data"]["store"]["region"].tolist() == [10, 20]
        assert sorted(seen["data"]["store"]) == ["region", "sqft"]


def test_duplicate_input_columns_is_harmless():
    """A column that ``input_columns`` names twice arrives once, without an error.

    Both names select the same arrays, so the factory drops the repeat rather than refusing it.
    """
    batch = pa.RecordBatch.from_arrays([pa.array([1, 2], type=pa.int64())], names=["a"])
    seen = run_batch(batch, input_columns=["a", "a"])
    assert seen["data"]["a"].tolist() == [1, 2]
    assert sorted(seen["bitmap"]) == ["a"]


def test_input_columns_selects_only_the_named_columns():
    """``data_dict`` and ``bitmap_dict`` hold only the columns that ``input_columns`` names, in its order."""
    batch = pa.RecordBatch.from_pydict({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    seen = run_batch(batch, input_columns=["c", "a"])
    assert list(seen["data"]) == ["c", "a"] and list(seen["bitmap"]) == ["c", "a"]
    assert seen["data"]["c"].tolist() == [5, 6]


def test_input_columns_is_read_once_so_a_generator_serves_every_batch():
    """A generator passed as ``input_columns`` selects the same columns in every batch, not only in the first.

    The factory reads ``input_columns`` once, when it makes the function. Read inside the batch loop, a generator,
    ``map()`` or ``filter()`` would be used up by the first batch, and every later batch would reach the UDF with no
    columns.
    """
    batches = [pa.RecordBatch.from_pydict({"x": [1, 2], "y": [0, 0]}),
               pa.RecordBatch.from_pydict({"x": [3], "y": [0]})]
    seen = []

    def main(data_dict, bitmap_dict, broadcasts):
        seen.append(list(data_dict))
        return {"out": data_dict["x"] * 2}

    got = list(make_mapinarrow_func(main, input_columns=(name for name in ["x"]))(iter(batches)))
    assert seen == [["x"], ["x"]]
    assert [batch.column("out").to_pylist() for batch in got] == [[2, 4], [6]]


def test_a_struct_field_sharing_a_column_name_reaches_the_udf():
    """A top-level column and a struct field of the same name both reach the UDF, in either column order.

    A Spark schema may hold such a pair, as ``id`` and ``order.id`` here. ``data_dict`` holds the field under its
    struct column, as ``data_dict["order"]["id"]``, so the field neither replaces the column ``id`` nor collides with
    it.
    """
    ids = pa.array([1, 2], type=pa.int64())
    orders = pa.array([{"id": 10, "total": 1.5}, {"id": 20, "total": 2.5}],
                      type=pa.struct([("id", pa.int64()), ("total", pa.float64())]))
    batch = pa.RecordBatch.from_arrays([ids, orders], names=["id", "order"])
    for order in (["id", "order"], ["order", "id"]):
        seen = run_batch(batch, input_columns=order)
        assert seen["data"]["id"].tolist() == [1, 2]
        assert seen["data"]["order"]["id"].tolist() == [10, 20]
        assert seen["bitmap"]["id"] is None and seen["bitmap"]["order"]["id"] is None


def test_null_free_column_arrives_as_none():
    """``bitmap_dict`` holds a key for a column whether or not the batch has a null in it.

    A UDF that indexes ``bitmap_dict`` by column name therefore works on every batch of a schema. The test checks the
    key only; ``test_null_free_column_of_every_supported_top_level_type`` checks that the value is ``None``.
    """
    with_null = pa.array([1.0, None, 3.0], type=pa.float64())
    without_null = pa.array([1.0, 2.0, 3.0], type=pa.float64())
    keys = []
    for col in (with_null, without_null):
        seen = run_batch(pa.RecordBatch.from_arrays([col], names=["magnitude"]))
        keys.append(sorted(seen["bitmap"]))
        assert "magnitude" in seen["bitmap"]
    assert keys[0] == keys[1] == ["magnitude"]


def test_null_free_column_of_every_supported_top_level_type():
    """A null-free column arrives with a ``None`` bitmap under its own key, for nine flat types.

    The types are int32, int64, float64, bool, string, large_string, date32, date64 and timestamp. The adapter also
    supports uint8, which the test leaves out.
    """
    columns = {
        "i32": pa.array([1, 2], type=pa.int32()),
        "i64": pa.array([1, 2], type=pa.int64()),
        "f64": pa.array([1.0, 2.0], type=pa.float64()),
        "b": pa.array([True, False], type=pa.bool_()),
        "s": pa.array(["a", "b"], type=pa.string()),
        "ls": pa.array(["a", "b"], type=pa.large_string()),
        "d32": pa.array([1, 2], type=pa.int32()).cast(pa.date32()),
        "d64": pa.array([86400000, 172800000], type=pa.int64()).cast(pa.date64()),
        "ts": pa.array([1, 2], type=pa.int64()).cast(pa.timestamp("ms")),
    }
    batch = pa.RecordBatch.from_arrays(list(columns.values()), names=list(columns))
    seen = run_batch(batch)
    assert sorted(seen["bitmap"]) == sorted(columns)
    assert all(seen["bitmap"][name] is None for name in columns)


def test_list_of_struct_column_nests_its_fields():
    """A list-of-struct column arrives as a dict of its fields, each holding the elements of every row, flattened."""
    col = pa.array([[{"v": 1}, {"v": 2}], [{"v": 3}, {"v": 4}]],
                   type=pa.list_(pa.struct([("v", pa.int64())])))
    seen = run_batch(pa.RecordBatch.from_arrays([col], names=["rows"]))
    assert sorted(seen["bitmap"]["rows"]) == ["v"]
    assert sorted(seen["data"]["rows"]) == ["v"]
    assert seen["data"]["rows"]["v"].tolist() == [1, 2, 3, 4]


def test_empty_batch():
    """A batch of no rows reaches the UDF as an empty array, with the column's key in ``bitmap_dict``."""
    batch = pa.RecordBatch.from_arrays([pa.array([], type=pa.int64())], names=["a"])
    seen = run_batch(batch)
    assert seen["data"]["a"].tolist() == []
    assert "a" in seen["bitmap"]


def test_zero_field_struct_column_arrives_as_an_empty_dict():
    """A struct column with no fields arrives as an empty dict in both ``data_dict`` and ``bitmap_dict``.

    The column's key is present like any other, so a UDF that indexes a column it selected finds the key.
    """
    z = pa.array([{}, {}], type=pa.struct([]))
    n = pa.array([1, 2], type=pa.int64())
    batch = pa.RecordBatch.from_arrays([z, n], names=["z", "n"])
    seen = run_batch(batch)
    assert seen["data"]["z"] == {} and seen["bitmap"]["z"] == {}
    assert seen["data"]["n"].tolist() == [1, 2]


def test_two_fields_of_one_struct_do_not_share_a_bitmap():
    """Two fields of one struct get equal bitmaps in separate arrays.

    When the fields have no nulls of their own, each field's bitmap is a copy of the struct's validity. Without the
    copy, the two fields would share one array, and a write through one would change the other.
    """
    inner_a = pa.array([1, 2], type=pa.int64())
    inner_b = pa.array([3, 4], type=pa.int64())
    col = pa.StructArray.from_arrays([inner_a, inner_b], ["a", "b"],
                                     mask=pa.array([False, True]))
    seen = run_batch(pa.RecordBatch.from_arrays([col], names=["s"]))
    bitmap_a, bitmap_b = seen["bitmap"]["s"]["a"], seen["bitmap"]["s"]["b"]
    assert bitmap_a is not bitmap_b
    assert bitmap_a.tolist() == bitmap_b.tolist()
    bitmap_a[0] = 0
    assert bitmap_b[0] != 0


def test_a_list_column_with_a_null_row_is_refused():
    """A list-of-struct column with a null outer row raises ``NotImplementedError``.

    The adapter returns the flattened struct elements with no offsets, and a null outer row holds no element. No bitmap
    can mark that row, and the elements no longer line up with the rows: here the field holds ``[1, 3]`` for three
    rows, and its second element belongs to the third row.
    """
    ty = pa.list_(pa.struct([("v", pa.int64())]))
    col = pa.array([[{"v": 1}], None, [{"v": 3}]], type=ty)
    assert col.null_count == 1
    with pytest.raises(NotImplementedError, match="null row"):
        run_batch(pa.RecordBatch.from_arrays([col], names=["rows"]))


def test_null_struct_element_inside_a_list_row_is_visible():
    """A null struct element inside a list row is null in the field's bitmap.

    The struct's validity covers the flattened elements, and the factory folds it into each field's bitmap. A null
    outer row is another case: ``test_a_list_column_with_a_null_row_is_refused`` checks that it raises.
    """
    inner = pa.array([{"v": 1}, None, {"v": 3}, {"v": 4}],
                     type=pa.struct([("v", pa.int64())]))
    col = pa.ListArray.from_arrays(pa.array([0, 2, 4], type=pa.int32()), inner)
    assert col.null_count == 0
    seen = run_batch(pa.RecordBatch.from_arrays([col], names=["rows"]))
    bitmap = seen["bitmap"]["rows"]["v"]
    assert bitmap is not None
    assert [is_null(i, bitmap) for i in range(4)] == [False, True, False, False]


OUT_SCHEMA = pa.schema([("a", pa.int64()), ("b", pa.int64())])


def run_outputs(outputs, output_schema=None):
    """Return the record batch the factory builds from ``outputs``, the dict its UDF returns, with ``output_schema``."""
    batch = pa.RecordBatch.from_arrays([pa.array([0, 0], type=pa.int64())], names=["c"])

    def main(data_dict, bitmap_dict, broadcasts):
        return outputs

    fn = make_mapinarrow_func(main, input_columns=["c"], output_schema=output_schema)
    return list(fn(iter([batch])))[0]


def test_output_schema_binds_the_udf_dict_by_name():
    """With ``output_schema``, the factory binds the returned dict by name rather than by insertion order.

    Spark binds the columns of a returned batch by position, so two columns of one type built in the other order would
    swap without an error.
    """
    outputs = {"b": np.array([10, 20], dtype=np.int64),
               "a": np.array([1, 2], dtype=np.int64)}
    got = run_outputs(outputs, OUT_SCHEMA)
    assert got.schema.names == ["a", "b"]
    assert got.column("a").to_pylist() == [1, 2]
    assert got.column("b").to_pylist() == [10, 20]


def test_without_an_output_schema_insertion_order_still_decides():
    """Without ``output_schema``, the columns of the built batch follow the insertion order of the returned dict.

    A caller who passes no schema relies on that order to match the schema given to ``mapInArrow``.
    """
    outputs = {"b": np.array([10, 20], dtype=np.int64),
               "a": np.array([1, 2], dtype=np.int64)}
    got = run_outputs(outputs)
    assert got.schema.names == ["b", "a"]
    assert got.column("b").to_pylist() == [10, 20]


def test_output_schema_refuses_a_column_the_udf_did_not_return():
    outputs = {"a": np.array([1, 2], dtype=np.int64)}
    with pytest.raises(KeyError, match="b"):
        run_outputs(outputs, OUT_SCHEMA)


def test_output_schema_refuses_a_value_that_cannot_convert():
    """Floats with a fraction under an int64 field raise ``ArrowInvalid`` naming the column, instead of truncating."""
    outputs = {"a": np.array([1.5, 2.5], dtype=np.float64),
               "b": np.array([3, 4], dtype=np.int64)}
    with pytest.raises(pa.ArrowInvalid, match="'a'"):
        run_outputs(outputs, OUT_SCHEMA)


def test_an_empty_string_column_keeps_its_type():
    """An empty unicode array comes back ``string``, as a non-empty one does.

    ``pa.array([])`` infers ``null``. A UDF that filters a whole batch away would otherwise yield a schema that its
    other batches do not share, and Spark's writer refuses a second schema.
    """
    outputs = {"s": np.empty(0, dtype="<U5"), "n": np.empty(0, dtype=np.int64)}
    got = run_outputs(outputs)
    assert got.schema.types == [pa.string(), pa.int64()]
    assert got.num_rows == 0


def test_a_bytes_column_keeps_its_nuls():
    """A bytes (``|S``) array keeps its leading and interior NULs and comes back ``binary``.

    ``pa.array`` reads a fixed-width bytes or unicode array with C string semantics and cuts each value at its first
    NUL. The factory converts both through ``tolist()``, which keeps every NUL but a trailing one.
    """
    values = [b"a\x00b", b"\x00lead", b"plain", b"x\x00\x00y"]
    got = run_outputs({"b": np.array(values, dtype="S5")})
    assert got.column("b").type == pa.binary()
    assert got.column("b").to_pylist() == values


def test_output_schema_refuses_a_key_it_does_not_name():
    """A returned key that ``output_schema`` does not name raises ``ValueError`` naming it, instead of being dropped.

    A dropped key would lose a column, a misspelled one for instance, without an error.
    """
    outputs = {"a": np.array([1, 2], dtype=np.int64),
               "b": np.array([3, 4], dtype=np.int64),
               "c": np.array([5, 6], dtype=np.int64)}
    with pytest.raises(ValueError, match="'c'"):
        run_outputs(outputs, OUT_SCHEMA)


def test_an_output_key_that_is_not_a_str_is_refused_by_name():
    """A returned key that is not a str raises ``TypeError`` naming the key and the rule.

    ``pa.RecordBatch.from_arrays`` would raise "expected bytes, int found", which names neither.
    """
    with pytest.raises(TypeError, match=r"5 is a int, not a str"):
        run_outputs({5: np.arange(2)})


def test_a_dict_under_one_output_key_is_refused():
    """A dict returned as a column's value raises ``TypeError`` naming the column, with or without ``output_schema``.

    ``pa.array`` iterates a mapping over its keys, so the dict would become a string column of its two keys, with no
    error.
    """
    arr = np.array([1, 2], dtype=np.int64)
    with pytest.raises(TypeError, match="'s'"):
        run_outputs({"s": {"a": arr, "b": arr}})
    with pytest.raises(TypeError, match="'s'"):
        run_outputs({"s": {"a": arr, "b": arr}}, pa.schema([("s", pa.string())]))


def test_output_schema_builds_a_string_column_as_declared():
    """A unicode array under a declared ``large_string`` is built as that type, and keeps an interior NUL.

    An empty array keeps the declared type too.
    """
    schema = pa.schema([("s", pa.large_string())])
    got = run_outputs({"s": np.array(["x", "y\x00z"])}, schema)
    assert got.column("s").type == pa.large_string()
    assert got.column("s").to_pylist() == ["x", "y\x00z"]
    empty = run_outputs({"s": np.empty(0, dtype="<U1")}, schema)
    assert empty.column("s").type == pa.large_string()


def test_a_day_unit_datetime64_declared_as_a_timestamp_keeps_its_dates():
    """A ``datetime64[D]`` array under a declared timestamp comes back as the midnights of its dates.

    Given a timestamp type, ``pa.array`` reads the array's 8-byte values as the 4-byte days of a ``date32``, so three
    dates would come back as the first date, the epoch and the second date. The factory converts the array to
    ``date32`` first and casts that to the declared type.
    """
    days = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[D]")
    midnights = [datetime.datetime(2020, 1, day) for day in (1, 2, 3)]
    for unit in ("s", "ms", "us", "ns"):
        got = run_outputs({"t": days}, pa.schema([("t", pa.timestamp(unit))])).column("t")
        assert got.type == pa.timestamp(unit), unit
        assert got.to_pylist() == midnights, unit
    zoned = run_outputs({"t": days}, pa.schema([("t", pa.timestamp("s", "UTC"))])).column("t")
    assert zoned.to_pylist() == [stamp.replace(tzinfo=datetime.timezone.utc) for stamp in midnights]


def test_a_day_unit_datetime64_under_another_declared_type_is_its_date32_cast():
    """A ``datetime64[D]`` array under a declared int32, string or int64 gives what its ``date32`` gives when cast.

    The results are the day numbers, the ISO dates, and pyarrow's ``ArrowNotImplementedError`` with the column named.
    Without the ``date32`` step, ``pa.array`` would read four dates under int32 as two day numbers and two zeros.
    """
    days = np.array(["2020-03-05", "1999-12-31", "2024-01-02", "1970-01-05"], dtype="datetime64[D]")
    numbers = run_outputs({"t": days}, pa.schema([("t", pa.int32())])).column("t")
    assert numbers.to_pylist() == [18326, 10956, 19724, 4]
    strings = run_outputs({"t": days}, pa.schema([("t", pa.string())])).column("t")
    assert strings.to_pylist() == ["2020-03-05", "1999-12-31", "2024-01-02", "1970-01-05"]
    with pytest.raises(pa.ArrowNotImplementedError, match=r"'t'.*date32"):
        run_outputs({"t": days}, pa.schema([("t", pa.int64())]))


def test_a_declared_type_keeps_the_other_datetime64_conversions():
    """Converting a day unit through ``date32`` leaves the other ``datetime64`` and ``timedelta64`` conversions alone.

    A unit change that drops digits still raises, and a date type still floors a timestamp to its day without an
    error. A day-unit ``timedelta64`` comes back in seconds, and a month-unit one raises, because a month has no fixed
    length.
    """
    days = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")
    sub_second = datetime.datetime(2020, 1, 1, 12, 34, 56, 789012)
    seconds = np.array([sub_second], dtype="datetime64[s]")
    for unit in ("ms", "us"):
        got = run_outputs({"t": seconds}, pa.schema([("t", pa.timestamp(unit))])).column("t")
        assert got.to_pylist() == [sub_second.replace(microsecond=0)], unit
    with pytest.raises(pa.ArrowInvalid, match=r"'t'.*would lose data"):
        run_outputs({"t": np.array([sub_second], dtype="datetime64[ms]")},
                    pa.schema([("t", pa.timestamp("s"))]))
    floored = run_outputs({"t": seconds}, pa.schema([("t", pa.date32())])).column("t")
    assert floored.to_pylist() == [datetime.date(2020, 1, 1)]
    dated = run_outputs({"t": days}, pa.schema([("t", pa.date64())])).column("t")
    assert dated.to_pylist() == [datetime.date(2020, 1, 1), datetime.date(2020, 1, 2)]
    # Arrow has no day unit for a duration, so the factory takes a day-unit timedelta64 to seconds.
    spans = np.array([1, 2], dtype="timedelta64[D]")
    for schema in (None, pa.schema([("t", pa.duration("s"))])):
        got = run_outputs({"t": spans}, schema).column("t")
        assert got.type == pa.duration("s") and got.to_pylist() == [datetime.timedelta(days=d) for d in (1, 2)]
    with pytest.raises(TypeError, match=r"'t'.*no fixed length"):
        run_outputs({"t": np.array([1, 2], dtype="timedelta64[M]")})


def test_a_coarse_time_unit_under_a_declared_non_temporal_type_is_refused_rather_than_rescaled():
    """A day- or hour-unit array under a declared int64 or string raises ``TypeError`` naming the column.

    Taking the unit to seconds would change the counts, so ``[1, 2]`` days would arrive as ``[86400, 172800]``.
    ``pa.array`` itself refuses these units.
    """
    for dtype in ("timedelta64[D]", "timedelta64[h]", "datetime64[h]"):
        counts = np.array([1, 2], dtype=dtype)
        for declared in (pa.int64(), pa.string()):
            with pytest.raises(TypeError, match=r"'n'.*would change them"):
                run_outputs({"n": counts}, pa.schema([("n", declared)]))
    # The factory still applies a unit's multiplier under a non-temporal type, because pa.array would read each count
    # as one of the base unit: counts of one and two five-second bins arrive as 5 and 10. A temporal declared type
    # still takes a coarse unit.
    bins = np.array([1, 2], dtype="timedelta64[5s]")
    assert run_outputs({"n": bins}, pa.schema([("n", pa.int64())])).column("n").to_pylist() == [5, 10]
    hours = np.array([1, 2], dtype="datetime64[h]")
    dated = run_outputs({"n": hours}, pa.schema([("n", pa.date32())])).column("n")
    assert dated.to_pylist() == [datetime.date(1970, 1, 1), datetime.date(1970, 1, 1)]


def test_output_schema_refuses_a_struct_key_no_field_has():
    """Under a declared struct, a dict key that no field has raises ``ValueError``, and a missing key gives a null.

    Arrow matches struct fields by exact name and fills a field that no key names with null. Dicts keyed ``Amount``
    and ``Label`` under the fields ``amount`` and ``label`` would build a column of null fields, with no error.
    """
    schema = pa.schema([("s", pa.struct([("amount", pa.int64()), ("label", pa.string())]))])
    got = run_outputs({"s": [{"amount": 1, "label": "x"}, {"amount": 2}]}, schema)
    assert got.column("s").to_pylist() == [{"amount": 1, "label": "x"}, {"amount": 2, "label": None}]
    with pytest.raises(ValueError, match="Amount"):
        run_outputs({"s": [{"Amount": 1, "Label": "x"}]}, schema)


def test_output_schema_refuses_a_struct_array_whose_fields_differ():
    """A struct array whose field names differ from the declared ones raises ``ValueError`` naming the field.

    The factory casts a ready-built array to the declared type. From pyarrow 19 on, the cast matches struct fields by
    name, so fields ``x`` and ``y`` cast to ``p`` and ``q`` would come back all null. An array with the declared names
    is cast as usual.
    """
    declared = pa.schema([("s", pa.struct([("p", pa.int64()), ("q", pa.int64())]))])
    built = pa.array([{"x": 1, "y": 2}], type=pa.struct([("x", pa.int64()), ("y", pa.int64())]))
    with pytest.raises(ValueError, match="'x'"):
        run_outputs({"s": built}, declared)
    same_names = pa.array([{"p": 1, "q": 2}], type=pa.struct([("p", pa.int32()), ("q", pa.int32())]))
    got = run_outputs({"s": same_names}, declared)
    assert got.column("s").type == declared.field("s").type
    assert got.column("s").to_pylist() == [{"p": 1, "q": 2}]
    nested = pa.array([[{"x": 1}]], type=pa.list_(pa.struct([("x", pa.int64())])))
    with pytest.raises(ValueError, match="'x'"):
        run_outputs({"s": nested}, pa.schema([("s", pa.list_(pa.struct([("p", pa.int64())])))]))
    # The next two cases nest a struct inside a struct and inside a map, each with an inner field that the declared
    # type does not name. Each case takes its own branch of the check.
    inner_built = pa.array([{"a": {"x": 1, "Y": 2}}],
                           type=pa.struct([("a", pa.struct([("x", pa.int64()), ("Y", pa.int64())]))]))
    inner_declared = pa.struct([("a", pa.struct([("x", pa.int64()), ("y", pa.int64())]))])
    with pytest.raises(ValueError, match="'Y'"):
        run_outputs({"s": inner_built}, pa.schema([("s", inner_declared)]))
    mapped_built = pa.array([[("k", {"Y": 1})]], type=pa.map_(pa.string(), pa.struct([("Y", pa.int64())])))
    with pytest.raises(ValueError, match="'Y'"):
        run_outputs({"s": mapped_built}, pa.schema([("s", pa.map_(pa.string(), pa.struct([("y", pa.int64())])))]))
    # A dictionary-encoded struct column takes the same check. The cast converts the dictionary's value structs, so
    # from pyarrow 19 on a renamed field would come back null in every row.
    encoded = pa.DictionaryArray.from_arrays(pa.array([0, 1, 0], type=pa.int32()),
                                             pa.array([{"x": 1}, {"x": 2}], type=pa.struct([("x", pa.int64())])))
    with pytest.raises(ValueError, match="'x'"):
        run_outputs({"s": encoded}, pa.schema([("s", pa.dictionary(pa.int32(), pa.struct([("y", pa.int64())])))]))


def test_a_struct_key_no_field_has_is_refused_at_any_depth():
    """A dict key that no declared field has raises ``ValueError`` wherever the struct sits.

    The struct sits in a list, in another struct, behind an object array or a generator, in a map's values or keys, and
    under an extension type. ``pa.array`` fills a misspelled field with null in every one of these cases, with no
    error.
    """
    inner = pa.struct([("amount", pa.int64())])
    cases = {
        "list of structs": (pa.list_(inner), [[{"Amount": 1}], [{"amount": 2}]]),
        "struct in struct": (pa.struct([("id", pa.int64()), ("inner", inner)]),
                             [{"id": 1, "inner": {"Amount": 9}}]),
        "object array of dicts": (inner, np.array([{"Amount": 1}, {"amount": 2}], dtype=object)),
        "generator of dicts": (inner, ({"Amount": i} for i in range(2))),
        "map of structs": (pa.map_(pa.string(), inner), [{"k": {"Amount": 1}}]),
        "map of structs from pairs": (pa.map_(pa.string(), inner), [[("k", {"Amount": 1})]]),
        "struct-keyed map": (pa.map_(inner, pa.int64()), [[({"Amount": 1}, 5)]]),
        # An extension type over a struct carries the struct's fields. The key check unwraps it both to decide whether
        # the type holds dicts and to compare the keys.
        "extension over struct": (Wrapped(inner), [{"Amount": 1}]),
    }
    for label, (declared_type, value) in cases.items():
        with pytest.raises(ValueError, match="Amount"):
            run_outputs({"s": value}, pa.schema([("s", declared_type)]))
    good = run_outputs({"s": [[{"amount": 1}], [{"amount": 2}]]}, pa.schema([("s", pa.list_(inner))]))
    assert good.column("s").to_pylist() == [[{"amount": 1}], [{"amount": 2}]]
    pairs = run_outputs({"s": [[("k", {"amount": 1})]]}, pa.schema([("s", pa.map_(pa.string(), inner))]))
    assert pairs.column("s").to_pylist() == [[("k", {"amount": 1})]]


def test_a_generator_output_keeps_its_rows_past_the_key_check():
    """A generator of correctly keyed dicts under a declared struct keeps every row.

    The key check reads the rows before ``pa.array`` does, and a generator yields its rows only once. The factory
    therefore reads a generator into a list first; without the list, the column would come back with no rows. The
    generator in ``test_a_struct_key_no_field_has_is_refused_at_any_depth`` raises before ``pa.array`` reads, so that
    test cannot detect lost rows.
    """
    schema = pa.schema([("s", pa.struct([("amount", pa.int64())]))])
    got = run_outputs({"s": ({"amount": i} for i in range(2))}, schema)
    assert got.column("s").to_pylist() == [{"amount": 0}, {"amount": 1}]


def test_a_missing_row_of_a_list_or_map_column_is_a_null_not_a_crash():
    """A list or map column given as a pandas Series with a ``NaN`` or ``pd.NA`` row converts with that row null.

    pandas may mark a missing row with ``NaN`` or ``pd.NA`` rather than ``None``, and ``pa.array`` converts both to
    null. The key check passes over a row that it cannot iterate, so such a row does not raise "'float' object is not
    iterable".
    """
    pd = pytest.importorskip("pandas")
    inner = pa.struct([("amount", pa.int64())])
    listed, mapped = pa.list_(inner), pa.map_(pa.string(), inner)
    cases = {
        "list rows, NaN": (listed, pd.Series([[{"amount": 1}], np.nan])),
        "list rows, pd.NA": (listed, pd.Series([[{"amount": 1}], pd.NA])),
        "map rows, NaN": (mapped, pd.Series([{"k": {"amount": 1}}, np.nan])),
        "map rows, pd.NA": (mapped, pd.Series([{"k": {"amount": 1}}, pd.NA])),
        "pair rows, NaN": (mapped, pd.Series([[("k", {"amount": 1})], np.nan])),
    }
    for label, (declared, value) in cases.items():
        got = run_outputs({"s": value}, pa.schema([("s", declared)])).column("s")
        assert got.to_pylist() == pa.array(value, type=declared).to_pylist(), label
    # The key check still finds a misspelled key in a Series with a missing row.
    for declared, value in ((listed, pd.Series([[{"Amount": 1}], np.nan])),
                            (mapped, pd.Series([{"k": {"Amount": 1}}, np.nan]))):
        with pytest.raises(ValueError, match="Amount"):
            run_outputs({"s": value}, pa.schema([("s", declared)]))


def test_a_scalar_row_of_a_list_or_map_column_is_not_spread_by_the_key_check():
    """The key check does not iterate a str, bytes or numeric ndarray row of a list or map column.

    Such a row iterates over scalars that carry no keys, and ``pa.array`` refuses it at its first element, naming the
    column. Spreading a long row into a list before that refusal takes seconds and hundreds of megabytes.
    """
    # Each of these row types fails the test if anything iterates it.
    class SpreadStr(str):
        def __iter__(self):
            raise AssertionError("the key check spread a str row")

    class SpreadBytes(bytes):
        def __iter__(self):
            raise AssertionError("the key check spread a bytes row")

    class SpreadArray(np.ndarray):
        def __iter__(self):
            raise AssertionError("the key check spread an ndarray row")

    inner = pa.struct([("amount", pa.int64())])
    listed, mapped = pa.list_(inner), pa.map_(pa.string(), inner)
    rows = [SpreadStr("abc"), SpreadBytes(b"abc"), np.arange(3).view(SpreadArray)]
    for declared, wrap in ((listed, lambda row: [row]), (mapped, lambda row: [row]),
                           (pa.list_(listed), lambda row: [[row]])):
        for row in rows:
            with pytest.raises((ValueError, TypeError), match="'s'"):
                run_outputs({"s": wrap(row)}, pa.schema([("s", declared)]))
    # The key check still finds a misspelled key in a row beside such a row.
    with pytest.raises(ValueError, match="Amount"):
        run_outputs({"s": [[{"Amount": 1}], SpreadStr("abc")]}, pa.schema([("s", listed)]))


def test_a_malformed_map_pair_is_refused_naming_the_column():
    """A map row whose pair has one element, or three, raises from ``pa.array`` naming the column.

    The factory adds the column's name to the error from ``pa.array``, and the key check raises no ``IndexError`` of
    its own.
    """
    # A map of plain values carries no dicts, so the key check does not run for it.
    schema = pa.schema([("m", pa.map_(pa.string(), pa.int64()))])
    with pytest.raises((ValueError, TypeError), match="'m'"):
        run_outputs({"m": [[("k",)]]}, schema)
    # The key check walks the pairs of a map of structs itself, and it passes over a pair of the wrong length.
    of_structs = pa.schema([("m", pa.map_(pa.string(), pa.struct([("amount", pa.int64())])))])
    for rows in ([[("k",)]], [[("k", {"amount": 1}, 2)]]):
        with pytest.raises((ValueError, TypeError), match="'m'"):
            run_outputs({"m": rows}, of_structs)


def test_output_columns_of_different_lengths_are_named():
    """Columns of different lengths raise ``ValueError`` listing each column's length, with or without a schema.

    pyarrow's own error says only "2 vs 3" and names neither column.
    """
    with pytest.raises(ValueError, match=r"'a': 3.*'b': 2"):
        run_outputs({"a": [1, 2, 3], "b": [1, 2]})
    with pytest.raises(ValueError, match=r"'a': 3.*'b': 2"):
        run_outputs({"a": [1, 2, 3], "b": [1, 2]}, OUT_SCHEMA)


def test_an_overflowing_output_value_names_its_column():
    with pytest.raises(OverflowError, match="'i'"):
        run_outputs({"i": [10 ** 400, 2]}, pa.schema([("i", pa.int64())]))


def test_output_schema_builds_a_map_column():
    """Under a declared map type, a list of dicts and a list of key-value pairs each build the map column.

    pyarrow cannot infer a map: a list of dicts infers a struct, which does not cast to a map, and a list of pairs
    fails inference. The factory builds the column as the declared type from the start.
    """
    schema = pa.schema([("m", pa.map_(pa.string(), pa.int64()))])
    for value in ([{"k": 1, "j": 2}], [[("k", 1), ("j", 2)]]):
        got = run_outputs({"m": value}, schema)
        assert got.column("m").type == schema.field("m").type
        assert got.column("m").to_pylist() == [[("k", 1), ("j", 2)]]


def test_a_record_array_becomes_a_struct_column():
    """A numpy record array becomes a struct column with one child per field, with or without ``output_schema``.

    A record array is what an ``@njit`` function returns for a numba record type, and ``pa.array`` refuses it with
    "Unsupported numpy type".
    """
    records = np.array([(1, 2.5, "ab"), (3, 4.5, "c\x00d")],
                       dtype=[("i", "i8"), ("f", "f8"), ("s", "U3")])
    got = run_outputs({"r": records})
    assert got.column("r").type == pa.struct([("i", pa.int64()), ("f", pa.float64()), ("s", pa.string())])
    assert got.column("r").to_pylist() == [{"i": 1, "f": 2.5, "s": "ab"}, {"i": 3, "f": 4.5, "s": "c\x00d"}]
    # A declared field that the record array lacks comes back null, and a record field that the declared struct lacks
    # is refused.
    declared = pa.schema([("r", pa.struct([("i", pa.int32()), ("s", pa.large_string()), ("extra", pa.int64())]))])
    got = run_outputs({"r": records[["i", "s"]]}, declared)
    assert got.column("r").type == declared.field("r").type
    assert got.column("r").to_pylist() == [{"i": 1, "s": "ab", "extra": None}, {"i": 3, "s": "c\x00d", "extra": None}]
    with pytest.raises(ValueError, match="'f'"):
        run_outputs({"r": records}, declared)
    with pytest.raises(TypeError, match=r"'r'.*cannot become int64"):
        run_outputs({"r": records}, pa.schema([("r", pa.int64())]))
    # A field that fails to convert is named in the error, beside the column.
    wide = np.array([(2 ** 40,), (1,)], dtype=[("i", "i8")])
    with pytest.raises(pa.ArrowInvalid, match=r"'r'.*field 'i'"):
        run_outputs({"r": wide}, pa.schema([("r", pa.struct([("i", pa.int32())]))]))


def test_a_record_array_field_that_fails_is_named_without_a_declared_type():
    """A record array field that fails to convert is named in the error, with or without a declared struct.

    Arrow has no complex type, so the complex128 field fails both when inferred and under a declared float64.
    """
    records = np.zeros(2, dtype=[("ok", "i8"), ("bad", "c16")])
    with pytest.raises(pa.ArrowException, match=r"'r'.*field 'bad'"):
        run_outputs({"r": records})
    declared = pa.schema([("r", pa.struct([("ok", pa.int64()), ("bad", pa.float64())]))])
    with pytest.raises(pa.ArrowException, match=r"'r'.*field 'bad'"):
        run_outputs({"r": records}, declared)


def test_a_record_array_with_no_fields_keeps_its_rows():
    """A record array with no fields becomes a column of empty structs, one per row, with or without a declared type.

    ``pa.StructArray.from_arrays([], names=[])`` has no child to take a length from and builds no rows. As the only
    output column, such a column would empty the batch without an error.
    """
    records = np.array([()] * 2, dtype=[])
    for schema in (None, pa.schema([("r", pa.struct([]))])):
        got = run_outputs({"r": records}, schema)
        assert got.column("r").type == pa.struct([])
        assert got.column("r").to_pylist() == [{}, {}]


def test_a_chunked_output_is_combined_before_it_is_read():
    """The factory combines a ``ChunkedArray`` into one array before it converts the column.

    A chunked struct column with other field names is then refused like any ready-built struct array, with the array's
    type in the message. A chunked int32 column comes back as one int32 column.
    """
    chunked = pa.chunked_array([pa.array([{"x": 1}], type=pa.struct([("x", pa.int64())]))])
    with pytest.raises(ValueError, match=r"the array is struct<x: int64>"):
        run_outputs({"s": chunked}, pa.schema([("s", pa.struct([("y", pa.int64())]))]))
    got = run_outputs({"s": pa.chunked_array([[1, 2], [3]], type=pa.int32())}).column("s")
    assert got.type == pa.int32() and got.to_pylist() == [1, 2, 3]


def test_an_output_side_failure_names_its_column():
    with pytest.raises(pa.ArrowInvalid, match="'bad'"):
        run_outputs({"bad": np.zeros((2, 2))})
    with pytest.raises(pa.ArrowInvalid, match="'a'"):
        run_outputs({"a": np.array([1.5, 2.5])}, pa.schema([("a", pa.int64())]))
    with pytest.raises(TypeError, match="'a'"):
        run_outputs({"a": 3})


def test_main_func_must_return_a_dict():
    # The pattern matches the refusal's own words, because the error that code after the guard would raise, "'NoneType'
    # object is not iterable", also names NoneType.
    with pytest.raises(TypeError, match="main_func must return a dict.*NoneType"):
        run_outputs(None)


def test_an_all_none_object_column_keeps_a_declared_type():
    """An object array holding only ``None`` comes back as its declared type, all null.

    Without ``output_schema`` the same column infers ``null``, as an empty list does. Declaring the type is how a UDF
    keeps such a column's type.
    """
    schema = pa.schema([("s", pa.string())])
    got = run_outputs({"s": np.array([None, None], dtype=object)}, schema)
    assert got.column("s").type == pa.string() and got.column("s").null_count == 2
    assert run_outputs({"s": np.array([None, None], dtype=object)}).column("s").type == pa.null()


def test_a_nullable_carries_the_input_nulls_out():
    """A ``Nullable`` with the input column's bitmap carries the input's nulls out, with or without a declared type.

    A bare array has no validity, so each null goes out as the value stored under it: ``[1, None, 3]`` doubled comes
    back ``[2, 0, 6]``.
    """
    column = pa.array([1, None, 3], type=pa.int64())
    batch = pa.RecordBatch.from_arrays([column], names=["c"])

    def double(data_dict, bitmap_dict, broadcasts):
        return {"bare": data_dict["c"] * 2, "kept": Nullable(data_dict["c"] * 2, bitmap_dict["c"])}

    got = list(make_mapinarrow_func(double, input_columns=["c"])(iter([batch])))[0]
    assert got.column("bare").to_pylist() == [2, 0, 6]
    assert got.column("kept").to_pylist() == [2, None, 6]

    def through(data_dict, bitmap_dict, broadcasts):
        return {"kept": Nullable(data_dict["c"], bitmap_dict["c"])}

    schema = pa.schema([("kept", pa.float64())])
    typed = list(make_mapinarrow_func(through, input_columns=["c"], output_schema=schema)(iter([batch])))[0]
    assert typed.column("kept").type == pa.float64()
    assert typed.column("kept").to_pylist() == [1.0, None, 3.0]


def test_a_nullable_masks_a_value_the_caller_masked_out():
    """A ``Nullable`` with the input's bitmap keeps a null that has a value stored under it.

    ``pc.if_else`` sets the second row null but leaves 42 in the data buffer, so a bare pass-through returns 42.
    """
    values = pa.array([1, 42, 3], type=pa.int64())
    column = pc.if_else(pa.array([True, False, True]), values, pa.scalar(None, pa.int64()))
    batch = pa.RecordBatch.from_arrays([column], names=["c"])

    def both(data_dict, bitmap_dict, broadcasts):
        return {"bare": data_dict["c"], "kept": Nullable(data_dict["c"], bitmap_dict["c"])}

    got = list(make_mapinarrow_func(both, input_columns=["c"])(iter([batch])))[0]
    assert got.column("bare").to_pylist() == [1, 42, 3]
    assert got.column("kept").to_pylist() == [1, None, 3]


def test_a_nullable_takes_the_bitmap_and_the_data_zero_copy():
    """A ``Nullable`` of an int64 ndarray and a packed bitmap builds a column over both arrays, copying neither.

    A fixed-width array with no nulls yet takes the bitmap as its validity buffer and keeps the ndarray as its data
    buffer.
    """
    data = np.array([1, 2, 3], dtype=np.int64)
    bitmap = np.array([0b101], dtype=np.uint8)
    got = run_outputs({"a": Nullable(data, bitmap)}).column("a")
    assert got.to_pylist() == [1, None, 3]
    assert got.buffers()[0].address == bitmap.ctypes.data
    assert got.buffers()[1].address == data.ctypes.data


def test_a_nullable_keeps_the_nulls_the_data_already_has():
    """A ``Nullable`` masks data that cannot take the bitmap as a buffer, and a null already in the data stays null.

    Data with nulls, an offset or child arrays is masked: here a list holding ``None``, a sliced Arrow array and a
    record array.
    """
    bitmap = np.array([0b011], dtype=np.uint8)
    assert run_outputs({"a": Nullable([None, 2, 3], bitmap)}).column("a").to_pylist() == [None, 2, None]
    sliced = pa.array([9, 1, 2, 3], type=pa.int64())[1:]
    assert run_outputs({"a": Nullable(sliced, bitmap)}).column("a").to_pylist() == [1, 2, None]
    records = np.array([(1, 2.5), (3, 4.5)], dtype=[("i", "i8"), ("f", "f8")])
    got = run_outputs({"r": Nullable(records, np.array([0b10], dtype=np.uint8))}).column("r")
    assert got.to_pylist() == [None, {"i": 3, "f": 4.5}]


def test_a_nullable_dictionary_column_is_masked_rather_than_rebuilt():
    """A ``Nullable`` of a dictionary column comes back dictionary-encoded, with the bitmap's nulls.

    ``pa.Array.from_buffers`` cannot rebuild a dictionary array: Arrow's C++ aborts the process with "Check failed:
    (data->dictionary) != (nullptr)" instead of raising. The factory therefore masks a dictionary column and never
    rebuilds one from buffers.
    """
    values = pa.array(["a", "b", "c"]).dictionary_encode()
    bitmap = np.array([0b101], dtype=np.uint8)
    declared = pa.schema([("d", pa.dictionary(pa.int32(), pa.string()))])
    for schema in (None, declared):
        got = run_outputs({"d": Nullable(values, bitmap)}, schema).column("d")
        assert got.type == values.type, schema
        assert got.to_pylist() == ["a", None, "c"], schema
        assert got.null_count == 1, schema


def test_a_nullable_with_no_bitmap_is_the_bare_array():
    """A ``Nullable`` with a ``None`` bitmap gives the same column as its bare data.

    ``bitmap_dict`` holds ``None`` for a column with no validity buffer, so a UDF can pass that ``None`` through
    unchanged. An empty column with an empty bitmap gives an empty column of the data's type.
    """
    got = run_outputs({"a": Nullable(np.array([1, 2], dtype=np.int64), None)}).column("a")
    assert got.to_pylist() == [1, 2] and got.null_count == 0
    empty = run_outputs({"a": Nullable(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.uint8))})
    assert empty.num_rows == 0 and empty.column("a").type == pa.int64()


def test_a_bitmap_of_the_wrong_length_is_refused():
    """A bitmap whose size is not ``(rows + 7) // 8`` bytes raises ``ValueError`` naming the column.

    pyarrow accepts a validity buffer that is too long, so a bitmap built for a longer column would be read without an
    error. A buffer that is too short makes pyarrow raise an error that names no column.
    """
    data = np.array([1, 2, 3], dtype=np.int64)
    with pytest.raises(ValueError, match=r"'a'.*3 bytes.*24 rows.*3 rows"):
        run_outputs({"a": Nullable(data, np.zeros(3, dtype=np.uint8))})
    with pytest.raises(ValueError, match=r"'a'.*0 bytes"):
        run_outputs({"a": Nullable(data, np.zeros(0, dtype=np.uint8))})


def test_a_bitmap_that_is_not_packed_uint8_is_refused():
    """A bitmap that is not a one-dimensional uint8 array raises ``TypeError`` naming the column.

    A boolean mask is the likely mistake, and Arrow would read its bytes as packed bits.
    """
    data = np.array([1, 2, 3], dtype=np.int64)
    with pytest.raises(TypeError, match=r"'a'.*bool"):
        run_outputs({"a": Nullable(data, np.array([True, False, True]))})
    with pytest.raises(TypeError, match=r"'a'.*2-dimensional"):
        run_outputs({"a": Nullable(data, np.zeros((1, 1), dtype=np.uint8))})


def test_a_bitmap_that_is_not_an_ndarray_is_refused():
    """A bitmap that is not an ndarray raises ``TypeError`` naming the column and the packed uint8 array it must be.

    The factory checks the type before it reads ``bitmap.dtype``. The factory adds the column's name only to certain
    exception classes, and ``AttributeError`` is not one of them, so the error would read "'list' object has no
    attribute 'dtype'".
    """
    data = np.array([1, 2, 3], dtype=np.int64)
    shapes = {
        "list": [0b101],
        "bytes": b"\x05",
        "UInt8Array": pa.array([5], type=pa.uint8()),
        "int": 5,
    }
    for name, bitmap in shapes.items():
        with pytest.raises(TypeError, match=r"'a'.*packed uint8 array.*not a " + name):
            run_outputs({"a": Nullable(data, bitmap)})


def test_a_handed_out_bitmap_is_refused_on_a_resized_column():
    """A bitmap from ``bitmap_dict`` raises on a column of another row count; a bitmap the UDF builds does not.

    A packed bitmap cannot tell 2 rows from 3, because both take one byte. A UDF that drops a row and passes the
    batch's bitmap through would give each row after the dropped one the bit of the row before it, without an error.
    The factory checks a bitmap that the UDF builds by its byte count alone.
    """
    column = pa.array([None, 2, 3], type=pa.int64())
    batch = pa.RecordBatch.from_arrays([column], names=["c"])

    def drop_one(data_dict, bitmap_dict, broadcasts):
        return {"c": Nullable(data_dict["c"][1:], bitmap_dict["c"])}

    with pytest.raises(ValueError, match=r"'c'.*handed out.*3 rows.*2 rows"):
        list(make_mapinarrow_func(drop_one, input_columns=["c"])(iter([batch])))

    def own_bitmap(data_dict, bitmap_dict, broadcasts):
        return {"c": Nullable(data_dict["c"][1:], np.packbits([1, 0], bitorder="little"))}

    got = list(make_mapinarrow_func(own_bitmap, input_columns=["c"])(iter([batch])))[0]
    assert got.column("c").to_pylist() == [2, None]


def test_a_handed_out_field_bitmap_covers_the_flattened_elements():
    """A field bitmap of a list-of-struct column is accepted on a column with one row per flattened element.

    The bitmaps of a list-of-struct column cover its flattened struct elements, six here for two outer rows. The
    factory checks a bitmap from ``bitmap_dict`` against the count it covers, so it accepts the bitmap on six rows and
    refuses it on three.
    """
    ty = pa.list_(pa.struct([("a", pa.int64())]))
    column = pa.array([[{"a": 1}, {"a": None}, {"a": 3}], [{"a": 4}, {"a": 5}, {"a": 6}]], type=ty)
    batch = pa.RecordBatch.from_arrays([column], names=["s"])

    def passed_through(data_dict, bitmap_dict, broadcasts):
        return {"out": Nullable(data_dict["s"]["a"] * 2, bitmap_dict["s"]["a"])}

    got = list(make_mapinarrow_func(passed_through, input_columns=["s"])(iter([batch])))[0]
    assert got.column("out").to_pylist() == [2, None, 6, 8, 10, 12]

    def copied(data_dict, bitmap_dict, broadcasts):
        return {"out": Nullable(data_dict["s"]["a"] * 2, bitmap_dict["s"]["a"].copy())}

    # A copy is a different object from the bitmap in bitmap_dict, so the factory checks it by its byte count alone.
    same = list(make_mapinarrow_func(copied, input_columns=["s"])(iter([batch])))[0]
    assert same.column("out").to_pylist() == [2, None, 6, 8, 10, 12]

    def half(data_dict, bitmap_dict, broadcasts):
        return {"out": Nullable(data_dict["s"]["a"][:3], bitmap_dict["s"]["a"])}

    with pytest.raises(ValueError, match=r"'out'.*handed out.*6 rows.*3 rows"):
        list(make_mapinarrow_func(half, input_columns=["s"])(iter([batch])))


def test_a_handed_out_bitmap_stays_alive_while_the_udf_runs():
    """A bitmap from ``bitmap_dict`` stays alive while the UDF runs, even after the UDF deletes it from the dict.

    The factory recognizes these bitmaps by ``id``, and Python may reuse an ``id`` once its object is freed. If the
    deleted bitmap were freed, a one-byte bitmap that the UDF builds could take its ``id``. The factory would then
    refuse a correctly resized column as if it carried the batch's bitmap.
    """
    column = pa.array([1.0, None, 3.0, 4.0, None])
    # The factory's batch loop still references the adapter's result for the last column it adapted. The column w comes
    # after v, so that reference does not keep v's bitmap alive, and only the factory's record of its bitmaps can.
    batch = pa.RecordBatch.from_arrays([column, pa.array(range(5))], names=["v", "w"])

    def drops_and_rebuilds(data_dict, bitmap_dict, broadcasts):
        handed = weakref.ref(bitmap_dict["v"])
        del bitmap_dict["v"]
        assert handed() is not None, "the handed-out bitmap was freed"
        return {"out": Nullable(data_dict["v"][:3] * 2, np.packbits([1, 0, 1], bitorder="little"))}

    got = list(make_mapinarrow_func(drops_and_rebuilds)(iter([batch])))[0]
    assert got.column("out").to_pylist() == [2.0, None, 6.0]


def test_a_bare_tuple_is_a_sequence_not_a_pair():
    """A bare tuple is a column of its elements, even a 2-tuple ending in ``None`` or holding two arrays.

    Only a ``Nullable`` is read as data and bitmap. A rule that read some bare 2-tuples as pairs would misread one of
    these columns.
    """
    assert run_outputs({"a": (1, 2, 3)}).column("a").to_pylist() == [1, 2, 3]
    assert run_outputs({"a": ("x", "y")}).column("a").to_pylist() == ["x", "y"]
    assert run_outputs({"a": (5, None)}).column("a").to_pylist() == [5, None]
    listed = run_outputs({"a": ([1, 2], None)}).column("a")
    assert listed.type == pa.list_(pa.int64()) and listed.to_pylist() == [[1, 2], None]
    arrays = run_outputs({"a": (np.array([1, 2]), np.array([3, 4]))}).column("a")
    assert arrays.type == pa.list_(pa.int64()) and arrays.to_pylist() == [[1, 2], [3, 4]]


def test_a_time_unit_multiplier_is_folded_in_before_arrow_reads_it():
    """A ``datetime64`` or ``timedelta64`` array with a unit multiplier, such as ``[5s]`` or ``[2D]``, keeps its values.

    ``pa.array`` reads numpy's base unit and ignores the multiplier, so five-second bins would arrive at one-second
    steps. A ``datetime64[2D]`` array would also miss the conversion through ``date32`` and be misread like a bare day
    unit.
    """
    base = np.datetime64("2020-03-01T10:00:00")
    stamps = (base + np.arange(3) * np.timedelta64(5, "s")).astype("datetime64[5s]")
    expected = [datetime.datetime(2020, 3, 1, 10, 0, s) for s in (0, 5, 10)]
    got = run_outputs({"t": stamps}).column("t")
    assert got.type == pa.timestamp("s") and got.to_pylist() == expected
    declared = pa.schema([("t", pa.timestamp("s"))])
    assert run_outputs({"t": stamps}, declared).column("t").to_pylist() == expected
    days = np.array(["2020-03-01", "2020-03-03", "2020-03-05"], dtype="datetime64[2D]")
    assert run_outputs({"d": days}).column("d").to_pylist() == [datetime.date(2020, 3, d) for d in (1, 3, 5)]
    stamped = run_outputs({"d": days}, pa.schema([("d", pa.timestamp("s"))])).column("d")
    assert stamped.to_pylist() == [datetime.datetime(2020, 3, d) for d in (1, 3, 5)]
    deltas = np.array([15, 60, 45], dtype="timedelta64[s]").astype("timedelta64[15s]")
    assert run_outputs({"e": deltas}).column("e").to_pylist() == [datetime.timedelta(seconds=s) for s in (15, 60, 45)]


def test_a_coarse_time_unit_becomes_seconds_or_days_and_a_finer_one_is_refused():
    """Without a declared type, a coarse time unit becomes seconds or days, and a picosecond unit raises ``TypeError``.

    An hour-unit ``datetime64`` becomes ``timestamp[s]``, a month-unit one ``date32``, and a week-unit ``timedelta64`` a
    duration of the same length. Arrow's timestamps and durations have units from seconds to nanoseconds, and
    ``pa.array`` raises ``ArrowNotImplementedError`` for an hour, minute, week, month or year unit. A month-unit
    ``timedelta64`` raises ``TypeError``, because a month has no fixed length.
    """
    hours = np.array(["2020-03-01T10", "2020-03-01T11"], dtype="datetime64[h]")
    got = run_outputs({"t": hours}).column("t")
    assert got.type == pa.timestamp("s")
    assert got.to_pylist() == [datetime.datetime(2020, 3, 1, h) for h in (10, 11)]
    months = np.array(["2020-03", "2020-04"], dtype="datetime64[M]")
    got = run_outputs({"d": months}).column("d")
    assert got.type == pa.date32() and got.to_pylist() == [datetime.date(2020, 3, 1), datetime.date(2020, 4, 1)]
    weeks = np.array([1, 2], dtype="timedelta64[W]")
    assert run_outputs({"e": weeks}).column("e").to_pylist() == [datetime.timedelta(weeks=w) for w in (1, 2)]
    with pytest.raises(TypeError, match=r"'t'.*nanoseconds"):
        run_outputs({"t": np.array([1, 2], dtype="datetime64[ps]")})
    with pytest.raises(TypeError, match=r"'e'.*no fixed length"):
        run_outputs({"e": np.array([1, 2], dtype="timedelta64[M]")})


def test_a_nullable_inside_a_nullable_is_refused():
    """A ``Nullable`` whose data is another ``Nullable`` raises ``TypeError`` naming the column.

    A ``Nullable`` is a 2-tuple, so ``pa.array`` would read the inner one as a column of two rows: its data and its
    bitmap.
    """
    inner = Nullable(np.arange(2, dtype=np.int64), None)
    with pytest.raises(TypeError, match=r"'out'.*Nullable inside a Nullable"):
        run_outputs({"out": Nullable(inner, None)})


def test_the_key_check_reaches_rows_that_are_not_dicts():
    """The key check finds a misspelled nested key behind rows that are not dicts.

    The rows are tuples, namedtuples, rows like pyspark's ``Row``, sequences with ``__len__`` and ``__getitem__`` but no
    ``__iter__``, and map entries given as ``{"key": ..., "value": ...}`` dicts. ``pa.array`` matches a nested dict's
    keys by name behind each of them, so a misspelled key would come back null.
    """
    Outer = collections.namedtuple("Outer", ["id", "inner"])

    # A tuple that keeps its field names in __fields__, as pyspark's Row does.
    class Row(tuple):
        __fields__ = ["id", "inner"]

    # A sequence with only __len__ and __getitem__, which iterates all the same.
    class Seq:
        def __init__(self, items):
            self._items = items

        def __len__(self):
            return len(self._items)

        def __getitem__(self, index):
            return self._items[index]

    nested = pa.schema([("s", pa.struct([("id", pa.int64()), ("inner", pa.struct([("amount", pa.int64())]))]))])
    for rows in ([(1, {"Amount": 5}), (2, {"Amount": 6})],
                 [Outer(1, {"Amount": 5}), Outer(2, {"Amount": 6})],
                 [Row((1, {"Amount": 5})), Row((2, {"Amount": 6}))]):
        with pytest.raises(ValueError, match=r"'s'.*'Amount'"):
            run_outputs({"s": rows}, nested)
    listed = pa.schema([("s", pa.list_(pa.struct([("amount", pa.int64())])))])
    with pytest.raises(ValueError, match=r"'s'.*'Amount'"):
        run_outputs({"s": [Seq([{"amount": 1}]), Seq([{"Amount": 2}])]}, listed)
    mapped = pa.schema([("s", pa.map_(pa.string(), pa.struct([("amount", pa.int64())])))])
    entries = [[{"key": "k", "value": {"Amount": 1}}], [{"key": "j", "value": {"amount": 2}}]]
    with pytest.raises(ValueError, match=r"'s'.*'Amount'"):
        run_outputs({"s": entries}, mapped)
    good = run_outputs({"s": [Outer(1, {"amount": 5}), (2, {"amount": 6})]}, nested).column("s")
    assert good.to_pylist() == [{"id": 1, "inner": {"amount": 5}}, {"id": 2, "inner": {"amount": 6}}]


def test_a_namedtuple_row_naming_the_fields_in_another_order_is_refused():
    """A namedtuple row that names the declared fields in another order raises ``ValueError``.

    ``pa.array`` binds a namedtuple by position, so ``YX(y=100, x=0)`` under ``struct<x, y>`` would put 100 in ``x``
    and 0 in ``y``, with no error.
    """
    YX = collections.namedtuple("YX", ["y", "x"])
    schema = pa.schema([("s", pa.struct([("x", pa.int64()), ("y", pa.int64())]))])
    with pytest.raises(ValueError, match=r"'s'.*\['y', 'x'\].*position"):
        run_outputs({"s": [YX(100, 0), YX(200, 1)]}, schema)
    XY = collections.namedtuple("XY", ["x", "y"])
    got = run_outputs({"s": [XY(0, 100), XY(1, 200)]}, schema).column("s")
    assert got.to_pylist() == [{"x": 0, "y": 100}, {"x": 1, "y": 200}]


def test_a_namedtuple_row_naming_other_fields_is_refused_rather_than_bound_by_position():
    """A namedtuple row whose field names are not the declared ones raises ``ValueError``.

    The factory compares the row's names with the declared names as lists, so it refuses a misnamed or an extra field
    as well as a reordered one. ``pa.array`` would bind ``Point(lon=10, latitude=50)`` under ``struct<lat, lon>`` by
    position, putting 10 in ``lat`` and 50 in ``lon``.
    """
    Point = collections.namedtuple("Point", ["lon", "latitude"])
    schema = pa.schema([("p", pa.struct([("lat", pa.float64()), ("lon", pa.float64())]))])
    with pytest.raises(ValueError, match=r"'p'.*\['lon', 'latitude'\].*not the declared fields.*position"):
        run_outputs({"p": [Point(lon=10.0, latitude=50.0)]}, schema)
    Extra = collections.namedtuple("Extra", ["lat", "lon", "alt"])
    with pytest.raises(ValueError, match=r"'p'.*\['lat', 'lon', 'alt'\].*not the declared fields"):
        run_outputs({"p": [Extra(50.0, 10.0, 0.0)]}, schema)


def test_pyarrow_scalar_rows_are_left_to_pa_array():
    """Rows that are pyarrow ``MapScalar`` values pass the key check untouched and convert through ``pa.array``.

    A ``MapScalar`` is a Mapping from pyarrow 21 on, and its ``values`` is an array rather than a method, so the key
    check would fail if it called ``values()``. ``pa.array`` checks a scalar row against the declared type itself.
    """
    mapped = pa.map_(pa.string(), pa.struct([("amount", pa.int64())]))
    rows = list(pa.array([[("k", {"amount": 1})], [("j", {"amount": 2}), ("i", {"amount": 3})]], type=mapped))
    got = run_outputs({"s": rows}, pa.schema([("s", mapped)])).column("s")
    assert got.to_pylist() == [[("k", {"amount": 1})], [("j", {"amount": 2}), ("i", {"amount": 3})]]


def test_a_pandas_series_row_is_refused_rather_than_read_by_label():
    """A pandas Series as a row raises ``TypeError`` that names the column and suggests ``list(row)``.

    ``pa.array`` reads a Series row by its index labels. A sorted Series would come back in label order, and one whose
    labels are not ``0..n-1``, as a groupby can leave them, would raise a bare ``KeyError``.
    """
    pd = pytest.importorskip("pandas")
    rows = [pd.Series([3, 1, 2]).sort_values(), pd.Series([6, 5, 4]).sort_values()]
    schema = pa.schema([("s", pa.list_(pa.int64()))])
    with pytest.raises(TypeError, match=r"'s'.*Series.*list\(row\)"):
        run_outputs({"s": rows}, schema)
    got = run_outputs({"s": [list(row) for row in rows]}, schema).column("s")
    assert got.to_pylist() == [[1, 2, 3], [4, 5, 6]]
    # pa.array reads a DataFrame by its column labels, so this one-column frame would raise a bare KeyError: 0.
    frame = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(TypeError, match=r"'out'.*DataFrame"):
        run_outputs({"out": frame[["x"]]})
    # pa.array returns a ChunkedArray for a Series over a multi-chunk pyarrow array, and the factory combines it.
    chunked = pd.concat([pd.Series(["a"], dtype="string[pyarrow]"), pd.Series(["b"], dtype="string[pyarrow]")])
    assert run_outputs({"w": chunked}).column("w").to_pylist() == ["a", "b"]


def test_a_pandas_series_row_inside_an_object_array_is_refused_too():
    """A pandas Series inside an object array is refused like one in a list, with or without a declared type.

    ``pa.array`` reads it by label there too: a Series sorted to ``[1, 2, 3]`` with labels ``[1, 2, 0]`` would come
    back ``[3, 1, 2]``.
    """
    pd = pytest.importorskip("pandas")
    rows = np.empty(1, dtype=object)
    rows[0] = pd.Series([3, 1, 2]).sort_values()
    with pytest.raises(TypeError, match=r"'s'.*Series.*list\(row\)"):
        run_outputs({"s": rows}, pa.schema([("s", pa.list_(pa.int64()))]))
    with pytest.raises(TypeError, match=r"'s'.*Series"):
        run_outputs({"s": rows})


def test_a_pandas_series_row_under_a_struct_names_to_dict_as_the_remedy():
    """Under a declared struct, the error for a Series row suggests ``row.to_dict()``, not ``row.to_numpy()``.

    ``pa.array`` refuses an ndarray and a list as a struct row, and converts the dict that ``row.to_dict()`` returns.
    """
    pd = pytest.importorskip("pandas")
    schema = pa.schema([("p", pa.struct([("lat", pa.float64()), ("lon", pa.float64())]))])
    rows = [pd.Series({"lat": 50.0, "lon": 10.0}), pd.Series({"lat": 1.0, "lon": 2.0})]
    with pytest.raises(TypeError, match=r"'p'.*Series.*row\.to_dict\(\)") as excinfo:
        run_outputs({"p": rows}, schema)
    assert "to_numpy" not in str(excinfo.value)
    got = run_outputs({"p": [row.to_dict() for row in rows]}, schema).column("p")
    assert got.to_pylist() == [{"lat": 50.0, "lon": 10.0}, {"lat": 1.0, "lon": 2.0}]


def test_a_key_error_from_pa_array_names_the_column_and_the_field():
    """A ``KeyError`` from ``pa.array`` names the column, and the field when a record array's field raised it.

    ``pa.array`` reads a ``UserDict`` row of a struct by index and raises ``KeyError(0)``. The factory adds the column
    to that error, whose message would otherwise be "0".
    """
    rows = [collections.UserDict({"amount": 1}), collections.UserDict({"amount": 2})]
    schema = pa.schema([("s", pa.struct([("amount", pa.int64())]))])
    with pytest.raises(KeyError, match=r"output column 's': 0"):
        run_outputs({"s": rows}, schema)
    records = np.array([(1, rows[0])], dtype=[("i", "i8"), ("o", "O")])
    declared = pa.schema([("r", pa.struct([("i", pa.int64()), ("o", pa.struct([("amount", pa.int64())]))]))])
    with pytest.raises(KeyError, match=r"'r'.*field 'o': 0"):
        run_outputs({"r": records}, declared)


def test_the_field_guard_sees_through_extension_map_and_view_layouts():
    """The field-name checks look through an extension type, a map declared as a list of entries, and a list view.

    The extension array and the map array are ready-built arrays, whose field names the factory compares with the
    declared ones before it casts. The list view is a declared type over rows of dicts, which the key check reads.
    Compared as unlike kinds, none of the three would be refused here. The list view would come back with nulls on any
    pyarrow that has list views, and the other two from pyarrow 19 on.
    """
    point = pa.struct([("x", pa.float64()), ("y", pa.float64())])
    if hasattr(pa, "opaque"):
        ext_type = pa.opaque(point, "point", "vendor")
        ext = pa.ExtensionArray.from_storage(ext_type, pa.array([{"x": 1.0, "y": 2.0}], type=point))
        declared = pa.schema([("p", pa.struct([("lon", pa.float64()), ("lat", pa.float64())]))])
        with pytest.raises(ValueError, match=r"'p'.*\['x', 'y'\]"):
            run_outputs({"p": ext}, declared)
    mapped = pa.array([[("k", {"Amount": 1})]], type=pa.map_(pa.string(), pa.struct([("Amount", pa.int64())])))
    entries = pa.struct([("key", pa.string()), ("value", pa.struct([("amount", pa.int64())]))])
    with pytest.raises(ValueError, match=r"'m'.*\['Amount'\]"):
        run_outputs({"m": mapped}, pa.schema([("m", pa.list_(entries))]))
    if hasattr(pa, "list_view"):
        viewed = pa.schema([("v", pa.list_view(pa.struct([("amount", pa.int64())])))])
        with pytest.raises(ValueError, match=r"'v'.*'Amount'"):
            run_outputs({"v": [[{"Amount": 5}]]}, viewed)


def test_a_batch_whose_inferred_type_differs_from_the_first_is_refused_by_name():
    """Without ``output_schema``, a batch whose inferred schema differs from the first batch's raises ``ValueError``.

    The error names the column and both types, or both lists of columns, and ``output_schema`` lets the same values
    through. Spark's writer refuses a batch whose schema differs from the first, and its error names no column.
    """
    batches = [pa.RecordBatch.from_pydict({"x": [1, 2]}), pa.RecordBatch.from_pydict({"x": [3]})]
    values = iter([[None, None], ["a"]])
    fn = make_mapinarrow_func(lambda d, b, br: {"s": next(values)})
    with pytest.raises(ValueError, match=r"'s'.*null.*string.*output_schema"):
        list(fn(iter(batches)))
    values = iter([[1, 2], [1.5]])
    fn = make_mapinarrow_func(lambda d, b, br: {"n": next(values)})
    with pytest.raises(ValueError, match=r"'n'.*int64.*double"):
        list(fn(iter(batches)))
    values = iter([{"a": [1, 2]}, {"b": [3]}])
    fn = make_mapinarrow_func(lambda d, b, br: next(values))
    with pytest.raises(ValueError, match=r"\['b'\].*\['a'\]"):
        list(fn(iter(batches)))
    values = iter([[None, None], ["a"]])
    declared = pa.schema([("s", pa.string())])
    fn = make_mapinarrow_func(lambda d, b, br: {"s": next(values)}, output_schema=declared)
    assert [batch.column("s").to_pylist() for batch in fn(iter(batches))] == [[None, None], ["a"]]


def test_a_nullable_extension_column_is_masked_through_its_storage():
    """A ``Nullable`` of an extension column over a dictionary is masked through its storage and keeps its type.

    An extension type reports no fields and no dictionary, whatever its storage. Judged by that type alone, a
    dictionary storage would go to ``from_buffers`` and abort the interpreter. A storage with a null would go to
    ``pc.if_else``, which has no kernel for extension types.
    """
    if not hasattr(pa, "opaque"):
        pytest.skip("pa.opaque arrived in pyarrow 17")
    labels = pa.opaque(pa.dictionary(pa.int32(), pa.string()), "label", "vendor")
    bitmap = np.array([0b101], dtype=np.uint8)
    for storage in (pa.array(["a", "b", "c"]).dictionary_encode(), pa.array(["a", None, "c"]).dictionary_encode()):
        column = pa.ExtensionArray.from_storage(labels, storage)
        got = run_outputs({"out": Nullable(column, bitmap)}).column("out")
        assert got.type == labels and got.storage.to_pylist() == ["a", None, "c"]


def test_nothing_of_a_batch_is_held_while_the_next_one_is_read():
    """No array adapted from a batch is alive when the factory asks for the next batch.

    If the factory's generator kept a batch's adapted arrays across its ``yield``, two batches' ``|U`` copies of a
    string column would be alive at once while the next batch is adapted.
    """
    seen = []

    def main(data_dict, bitmap_dict, broadcasts):
        seen.append(weakref.ref(data_dict["s"]))
        return {"n": np.zeros(len(data_dict["s"]), dtype=np.int64)}

    alive_when_the_next_is_read = []

    # Each time the factory asks for a batch, the iterator records whether the previous batch's adapted array is alive.
    class Batches:
        def __init__(self):
            self.batches = [pa.RecordBatch.from_pydict({"s": ["a", "b"]}), pa.RecordBatch.from_pydict({"s": ["c"]})]

        def __iter__(self):
            return self

        def __next__(self):
            if seen:
                gc.collect()
                alive_when_the_next_is_read.append(seen[-1]() is not None)
            if not self.batches:
                raise StopIteration
            return self.batches.pop(0)

    for _ in make_mapinarrow_func(main)(Batches()):
        pass
    assert alive_when_the_next_is_read == [False, False]


def test_a_fixed_width_binary_column_keeps_a_trailing_nul_under_its_declared_type():
    """A ``|S16`` array under a declared ``binary(16)`` keeps every byte, a trailing NUL included.

    ``tolist()`` drops a trailing NUL, so a digest ending in 0x00, one in 256, would come back a byte short and fail
    the fixed width. Under a fixed-size binary type, the factory passes the array to ``pa.array`` directly.
    """
    digests = np.array([b"0123456789abcde\x00", b"\x00fedcba987654321", b"0123456789abcdef"], dtype="|S16")
    declared = pa.schema([("d", pa.binary(16))])
    got = run_outputs({"d": digests}, declared).column("d")
    assert got.type == pa.binary(16)
    assert got.to_pylist() == [b"0123456789abcde\x00", b"\x00fedcba987654321", b"0123456789abcdef"]


def test_the_key_check_covers_the_list_layouts_a_missing_field_and_a_none_row():
    """The key check finds a misspelled key in ``large_list`` and ``fixed_size_list`` columns.

    It also lets two rows through to ``pa.array``: a row that lacks a nested field, whose field comes back null, and a
    ``None`` row of a struct column, which comes back null. Each case exercises a separate condition of the check.
    """
    inner = pa.struct([("amount", pa.int64())])
    for layout in (pa.large_list(inner), pa.list_(inner, 1)):
        with pytest.raises(ValueError, match=r"'s'.*'Amount'"):
            run_outputs({"s": [[{"Amount": 1}], [{"amount": 2}]]}, pa.schema([("s", layout)]))
    nested = pa.schema([("s", pa.struct([("id", pa.int64()), ("inner", inner)]))])
    got = run_outputs({"s": [{"id": 1, "inner": {"amount": 5}}, {"id": 2}]}, nested).column("s")
    assert got.to_pylist() == [{"id": 1, "inner": {"amount": 5}}, {"id": 2, "inner": None}]
    got = run_outputs({"s": [{"amount": 1}, None]}, pa.schema([("s", inner)])).column("s")
    assert got.to_pylist() == [{"amount": 1}, None]


def test_an_empty_bytes_column_keeps_its_type():
    """An empty bytes (``|S``) array comes back ``binary``, as an empty unicode array comes back ``string``.

    ``pa.array`` infers ``null`` from an empty list, so the factory gives a bytes array the type ``binary`` when none is
    declared.
    """
    got = run_outputs({"b": np.empty(0, dtype="S5"), "n": np.empty(0, dtype=np.int64)})
    assert [field.type for field in got.schema] == [pa.binary(), pa.int64()]


def test_every_refusal_class_of_a_record_field_names_the_field():
    """A record field that fails with ``ValueError``, ``TypeError`` or ``OverflowError`` is named in the error.

    The three cases are a dict key that no field has, a nested record array declared as a list, and an int beyond
    int64. ``test_a_record_array_becomes_a_struct_column`` covers a pyarrow error.
    """
    inner = pa.struct([("amount", pa.int64())])
    with_dict = np.array([({"Amount": 1},)], dtype=[("meta", "O")])
    with pytest.raises(ValueError, match=r"'r': field 'meta': declared"):
        run_outputs({"r": with_dict}, pa.schema([("r", pa.struct([("meta", inner)]))]))
    nested = np.array([((1, 2),)], dtype=[("pair", [("c", "i8"), ("d", "i8")])])
    with pytest.raises(TypeError, match=r"'r': field 'pair': a record array"):
        run_outputs({"r": nested}, pa.schema([("r", pa.struct([("pair", pa.list_(pa.int64()))]))]))
    big = np.array([(2 ** 70,)], dtype=[("big", "O")])
    with pytest.raises(OverflowError, match=r"'r': field 'big'"):
        run_outputs({"r": big}, pa.schema([("r", pa.struct([("big", pa.int64())]))]))
