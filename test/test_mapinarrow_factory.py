import datetime

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from numbarrow.core.is_null import is_null
from numbarrow.core.mapinarrow_factory import Nullable, make_mapinarrow_func


def run_batch(batch, input_columns=None):
    """Run one RecordBatch through the factory and return what the UDF saw."""
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
    # Plain pa.array, from_pylist and StructArray.from_arrays(mask=...) all emit
    # a struct-null row with no child validity buffer, so the field's own bitmap
    # cannot see it. The struct-level bits are folded in, so one is_null call
    # per field does.
    col = pa.array([{"v": 10}, None, {"v": 30}], type=pa.struct([pa.field("v", pa.int64())]))
    seen = run_batch(pa.RecordBatch.from_arrays([col], names=["s"]))
    assert sorted(seen["bitmap"]) == ["s"] and sorted(seen["bitmap"]["s"]) == ["v"]
    assert nulls(seen["bitmap"]["s"]["v"], 3) == [False, True, False]


def test_no_bitmap_when_nothing_is_null_at_either_layer():
    col = pa.array([{"v": 10}, {"v": 30}], type=pa.struct([pa.field("v", pa.int64())]))
    seen = run_batch(pa.RecordBatch.from_arrays([col], names=["s"]))
    assert "v" in seen["bitmap"]["s"] and seen["bitmap"]["s"]["v"] is None
    assert seen["data"]["s"]["v"].tolist() == [10, 30]


def test_both_null_layers_are_folded_together():
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
    # A struct column called 'region' next to store: struct<region, sqft>. Each
    # field sits under its own column, so the two names never meet.
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
    # Naming a column twice produces the same arrays twice, so it must not
    # become an error.
    batch = pa.RecordBatch.from_arrays([pa.array([1, 2], type=pa.int64())], names=["a"])
    seen = run_batch(batch, input_columns=["a", "a"])
    assert seen["data"]["a"].tolist() == [1, 2]
    assert sorted(seen["bitmap"]) == ["a"]


def test_input_columns_selects_only_the_named_columns():
    # input_columns had no end-to-end test: every column reaching the UDF
    # regardless kept the suite green.
    batch = pa.RecordBatch.from_pydict({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    seen = run_batch(batch, input_columns=["c", "a"])
    assert list(seen["data"]) == ["c", "a"] and list(seen["bitmap"]) == ["c", "a"]
    assert seen["data"]["c"].tolist() == [5, 6]


def test_a_struct_field_sharing_a_column_name_reaches_the_udf():
    # Four ordinary Spark StructTypes convert to this shape: a top-level column
    # and a struct field sharing a name. Nested under its column, the field
    # neither replaces the column nor collides with it, whichever comes first.
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
    # The same schema over two batches, one with a null and one without. A UDF
    # indexing bitmap_dict must find the key in both.
    with_null = pa.array([1.0, None, 3.0], type=pa.float64())
    without_null = pa.array([1.0, 2.0, 3.0], type=pa.float64())
    keys = []
    for col in (with_null, without_null):
        seen = run_batch(pa.RecordBatch.from_arrays([col], names=["magnitude"]))
        keys.append(sorted(seen["bitmap"]))
        assert "magnitude" in seen["bitmap"]
    assert keys[0] == keys[1] == ["magnitude"]


def test_null_free_column_of_every_supported_top_level_type():
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
    col = pa.array([[{"v": 1}, {"v": 2}], [{"v": 3}, {"v": 4}]],
                   type=pa.list_(pa.struct([("v", pa.int64())])))
    seen = run_batch(pa.RecordBatch.from_arrays([col], names=["rows"]))
    assert sorted(seen["bitmap"]["rows"]) == ["v"]
    assert sorted(seen["data"]["rows"]) == ["v"]
    assert seen["data"]["rows"]["v"].tolist() == [1, 2, 3, 4]


def test_empty_batch():
    batch = pa.RecordBatch.from_arrays([pa.array([], type=pa.int64())], names=["a"])
    seen = run_batch(batch)
    assert seen["data"]["a"].tolist() == []
    assert "a" in seen["bitmap"]


def test_zero_field_struct_column_arrives_as_an_empty_dict():
    # A struct with no fields used to contribute no key at all, so a requested
    # column vanished from both dicts without a word. Nested under its own
    # name it is present and empty.
    z = pa.array([{}, {}], type=pa.struct([]))
    n = pa.array([1, 2], type=pa.int64())
    batch = pa.RecordBatch.from_arrays([z, n], names=["z", "n"])
    seen = run_batch(batch)
    assert seen["data"]["z"] == {} and seen["bitmap"]["z"] == {}
    assert seen["data"]["n"].tolist() == [1, 2]


def test_two_fields_of_one_struct_do_not_share_a_bitmap():
    # With the struct-level bits folded in and no field-level nulls, both
    # fields get the struct bitmap; without a per-field copy they would be the
    # same array, and a caller writing through one would change the other.
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
    # The fold covers the flattened struct elements, never the outer list rows,
    # so a null outer row is invisible AND shifts the element-to-row mapping:
    # this used to answer [1, 3] for three rows, with element 1 belonging to
    # row 3. Nothing in the returned shape can express that, so it is refused.
    ty = pa.list_(pa.struct([("v", pa.int64())]))
    col = pa.array([[{"v": 1}], None, [{"v": 3}]], type=ty)
    assert col.null_count == 1
    with pytest.raises(NotImplementedError, match="null row"):
        run_batch(pa.RecordBatch.from_arrays([col], names=["rows"]))


def test_null_struct_element_inside_a_list_row_is_visible():
    # Distinct from a null OUTER row, which is not reported at all: a null
    # struct ELEMENT inside a list row does have struct-level validity, and the
    # fold must surface it. No test covered this layer.
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
    """Run one batch through the factory with a UDF returning `outputs`."""
    batch = pa.RecordBatch.from_arrays([pa.array([0, 0], type=pa.int64())], names=["c"])

    def main(data_dict, bitmap_dict, broadcasts):
        return outputs

    fn = make_mapinarrow_func(main, input_columns=["c"], output_schema=output_schema)
    return list(fn(iter([batch])))[0]


def test_output_schema_binds_the_udf_dict_by_name():
    # Two same-typed columns built in the other order are what silently swap
    # when position decides. Naming the schema takes insertion order out of it.
    outputs = {"b": np.array([10, 20], dtype=np.int64),
               "a": np.array([1, 2], dtype=np.int64)}
    got = run_outputs(outputs, OUT_SCHEMA)
    assert got.schema.names == ["a", "b"]
    assert got.column("a").to_pylist() == [1, 2]
    assert got.column("b").to_pylist() == [10, 20]


def test_without_an_output_schema_insertion_order_still_decides():
    # The documented default, pinned so that adding the option did not quietly
    # change what a caller who passes nothing gets.
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
    # A float column against an int64 field: the conversion loses the fraction,
    # so it raises rather than truncating into the declared type.
    outputs = {"a": np.array([1.5, 2.5], dtype=np.float64),
               "b": np.array([3, 4], dtype=np.int64)}
    with pytest.raises(pa.ArrowInvalid, match="'a'"):
        run_outputs(outputs, OUT_SCHEMA)


def test_an_empty_string_column_keeps_its_type():
    # pa.array([]) infers null where the same column with rows infers string,
    # so a UDF that filtered a whole batch away yielded a schema its other
    # batches did not share, and Spark's writer refused the second one.
    outputs = {"s": np.empty(0, dtype="<U5"), "n": np.empty(0, dtype=np.int64)}
    got = run_outputs(outputs)
    assert got.schema.types == [pa.string(), pa.int64()]
    assert got.num_rows == 0


def test_a_bytes_column_keeps_its_nuls():
    # A |S array handed straight to pa.array is read with C string semantics,
    # the same cut a |U array used to get.
    values = [b"a\x00b", b"\x00lead", b"plain", b"x\x00\x00y"]
    got = run_outputs({"b": np.array(values, dtype="S5")})
    assert got.column("b").type == pa.binary()
    assert got.column("b").to_pylist() == values


def test_output_schema_refuses_a_key_it_does_not_name():
    # An unnamed key used to be dropped silently, the one mismatch the schema
    # did not catch.
    outputs = {"a": np.array([1, 2], dtype=np.int64),
               "b": np.array([3, 4], dtype=np.int64),
               "c": np.array([5, 6], dtype=np.int64)}
    with pytest.raises(ValueError, match="'c'"):
        run_outputs(outputs, OUT_SCHEMA)


def test_an_output_key_that_is_not_a_str_is_refused_by_name():
    # pa.RecordBatch.from_arrays died on "expected bytes, int found", naming
    # neither the key nor the rule.
    with pytest.raises(TypeError, match=r"5 is a int, not a str"):
        run_outputs({5: np.arange(2)})


def test_a_dict_under_one_output_key_is_refused():
    # pa.array iterates a mapping, so this used to come back as a string
    # column of the keys, two rows long, with nothing raised.
    arr = np.array([1, 2], dtype=np.int64)
    with pytest.raises(TypeError, match="'s'"):
        run_outputs({"s": {"a": arr, "b": arr}})
    with pytest.raises(TypeError, match="'s'"):
        run_outputs({"s": {"a": arr, "b": arr}}, pa.schema([("s", pa.string())]))


def test_output_schema_builds_a_string_column_as_declared():
    schema = pa.schema([("s", pa.large_string())])
    got = run_outputs({"s": np.array(["x", "y\x00z"])}, schema)
    assert got.column("s").type == pa.large_string()
    assert got.column("s").to_pylist() == ["x", "y\x00z"]
    empty = run_outputs({"s": np.empty(0, dtype="<U1")}, schema)
    assert empty.column("s").type == pa.large_string()


def test_a_day_unit_datetime64_declared_as_a_timestamp_keeps_its_dates():
    # pa.array reads a day-unit array's 8-byte values as the 4-byte days of a
    # date32, so three dates declared timestamp[s] came back as the first, the
    # epoch and the second: every other row wrong, with nothing raised.
    days = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[D]")
    midnights = [datetime.datetime(2020, 1, day) for day in (1, 2, 3)]
    for unit in ("s", "ms", "us", "ns"):
        got = run_outputs({"t": days}, pa.schema([("t", pa.timestamp(unit))])).column("t")
        assert got.type == pa.timestamp(unit), unit
        assert got.to_pylist() == midnights, unit
    zoned = run_outputs({"t": days}, pa.schema([("t", pa.timestamp("s", "UTC"))])).column("t")
    assert zoned.to_pylist() == [stamp.replace(tzinfo=datetime.timezone.utc) for stamp in midnights]


def test_a_day_unit_datetime64_under_another_declared_type_is_its_date32_cast():
    # The same misread under int32 gave two day numbers and two zeros for
    # four dates. Inferred first the array is a date32, and the cast to the
    # declared type gives the day numbers, the ISO dates, or pyarrow's own
    # refusal naming the column.
    days = np.array(["2020-03-05", "1999-12-31", "2024-01-02", "1970-01-05"], dtype="datetime64[D]")
    numbers = run_outputs({"t": days}, pa.schema([("t", pa.int32())])).column("t")
    assert numbers.to_pylist() == [18326, 10956, 19724, 4]
    strings = run_outputs({"t": days}, pa.schema([("t", pa.string())])).column("t")
    assert strings.to_pylist() == ["2020-03-05", "1999-12-31", "2024-01-02", "1970-01-05"]
    with pytest.raises(pa.ArrowNotImplementedError, match=r"'t'.*date32"):
        run_outputs({"t": days}, pa.schema([("t", pa.int64())]))


def test_a_declared_type_keeps_the_other_datetime64_conversions():
    # What widening the day unit must leave alone: a unit change that drops
    # digits still raises, a date type still floors to the day without a word,
    # and a timedelta is still a dtype pa.array has no converter for.
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
    spans = np.array([1, 2], dtype="timedelta64[D]")
    with pytest.raises(pa.ArrowNotImplementedError, match=r"'t'.*timedelta64"):
        run_outputs({"t": spans}, pa.schema([("t", pa.duration("s"))]))
    with pytest.raises(pa.ArrowNotImplementedError, match=r"'t'.*timedelta64"):
        run_outputs({"t": spans})


def test_output_schema_refuses_a_struct_key_no_field_has():
    # Arrow matches struct fields by exact name and nulls a missing one, so a
    # list of dicts keyed Amount/Label against amount/label used to come back
    # as two columns of nulls under an identical schema.
    schema = pa.schema([("s", pa.struct([("amount", pa.int64()), ("label", pa.string())]))])
    got = run_outputs({"s": [{"amount": 1, "label": "x"}, {"amount": 2}]}, schema)
    assert got.column("s").to_pylist() == [{"amount": 1, "label": "x"}, {"amount": 2, "label": None}]
    with pytest.raises(ValueError, match="Amount"):
        run_outputs({"s": [{"Amount": 1, "Label": "x"}]}, schema)


def test_output_schema_refuses_a_struct_array_whose_fields_differ():
    # A ready-built StructArray was cast to the declared type, and a cast
    # matches struct fields by name too: x/y against p/q came back all null.
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
    # One case per branch of the depth check: a struct inside a struct and a
    # struct inside a map, each with an inner field the declared type does
    # not name.
    inner_built = pa.array([{"a": {"x": 1, "Y": 2}}],
                           type=pa.struct([("a", pa.struct([("x", pa.int64()), ("Y", pa.int64())]))]))
    inner_declared = pa.struct([("a", pa.struct([("x", pa.int64()), ("y", pa.int64())]))])
    with pytest.raises(ValueError, match="'Y'"):
        run_outputs({"s": inner_built}, pa.schema([("s", inner_declared)]))
    mapped_built = pa.array([[("k", {"Y": 1})]], type=pa.map_(pa.string(), pa.struct([("Y", pa.int64())])))
    with pytest.raises(ValueError, match="'Y'"):
        run_outputs({"s": mapped_built}, pa.schema([("s", pa.map_(pa.string(), pa.struct([("y", pa.int64())])))]))
    # A dictionary is a layout: the cast decodes it and matches the value
    # structs by name, which filled a whole column with nulls.
    encoded = pa.DictionaryArray.from_arrays(pa.array([0, 1, 0], type=pa.int32()),
                                             pa.array([{"x": 1}, {"x": 2}], type=pa.struct([("x", pa.int64())])))
    with pytest.raises(ValueError, match="'x'"):
        run_outputs({"s": encoded}, pa.schema([("s", pa.dictionary(pa.int32(), pa.struct([("y", pa.int64())])))]))


def test_a_struct_key_no_field_has_is_refused_at_any_depth():
    # The check used to cover a plain list of dicts against a struct field and
    # nothing else: a list-of-struct column, a struct inside a struct, an
    # object array and a generator all came back null on the same typo.
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
    }
    for label, (declared_type, value) in cases.items():
        with pytest.raises(ValueError, match="Amount"):
            run_outputs({"s": value}, pa.schema([("s", declared_type)]))
    good = run_outputs({"s": [[{"amount": 1}], [{"amount": 2}]]}, pa.schema([("s", pa.list_(inner))]))
    assert good.column("s").to_pylist() == [[{"amount": 1}], [{"amount": 2}]]
    pairs = run_outputs({"s": [[("k", {"amount": 1})]]}, pa.schema([("s", pa.map_(pa.string(), inner))]))
    assert pairs.column("s").to_pylist() == [[("k", {"amount": 1})]]


def test_a_generator_output_keeps_its_rows_past_the_key_check():
    # The key check reads the rows before pa.array does, and a generator read
    # once has nothing left for the second reader: without the list it is
    # read into first, a generator of well-keyed dicts came back as a column
    # of no rows. Only the typo path, which raises before pa.array reads,
    # had a test.
    schema = pa.schema([("s", pa.struct([("amount", pa.int64())]))])
    got = run_outputs({"s": ({"amount": i} for i in range(2))}, schema)
    assert got.column("s").to_pylist() == [{"amount": 0}, {"amount": 1}]


def test_a_missing_row_of_a_list_or_map_column_is_a_null_not_a_crash():
    # pandas marks a missing row with NaN or pd.NA, never with None, and
    # pa.array turns both into nulls. The key pre-pass walked into every row
    # that was not exactly None and died on "'float' object is not iterable",
    # refusing a whole batch whose column pa.array converts.
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
    # The typo the pre-pass exists for is still caught in the same shape.
    for declared, value in ((listed, pd.Series([[{"Amount": 1}], np.nan])),
                            (mapped, pd.Series([{"k": {"Amount": 1}}, np.nan]))):
        with pytest.raises(ValueError, match="Amount"):
            run_outputs({"s": value}, pa.schema([("s", declared)]))


def test_a_scalar_row_of_a_list_or_map_column_is_not_spread_by_the_key_check():
    # A str, a bytes and a numeric ndarray iterate, over scalars that carry no
    # keys, and pa.array refuses such a row at its first element. The key
    # pre-pass used to spread the whole row into a list before that refusal,
    # seconds and hundreds of megabytes for a long one.
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
    # The rows beside it are still looked inside.
    with pytest.raises(ValueError, match="Amount"):
        run_outputs({"s": [[{"Amount": 1}], SpreadStr("abc")]}, pa.schema([("s", listed)]))


def test_a_malformed_map_pair_is_refused_naming_the_column():
    # A one-element "pair" reaches pa.array's own refusal rather than an
    # IndexError from the key check, so the column is named.
    schema = pa.schema([("m", pa.map_(pa.string(), pa.int64()))])
    with pytest.raises((ValueError, TypeError), match="'m'"):
        run_outputs({"m": [[("k",)]]}, schema)


def test_output_columns_of_different_lengths_are_named():
    # pyarrow's own refusal says "2 vs 3" and names neither column.
    with pytest.raises(ValueError, match=r"'a': 3.*'b': 2"):
        run_outputs({"a": [1, 2, 3], "b": [1, 2]})
    with pytest.raises(ValueError, match=r"'a': 3.*'b': 2"):
        run_outputs({"a": [1, 2, 3], "b": [1, 2]}, OUT_SCHEMA)


def test_an_overflowing_output_value_names_its_column():
    with pytest.raises(OverflowError, match="'i'"):
        run_outputs({"i": [10 ** 400, 2]}, pa.schema([("i", pa.int64())]))


def test_output_schema_builds_a_map_column():
    # A map has no inferred type to be cast from: a list of dicts infers a
    # struct, which does not cast to map, and a list of pairs fails inference.
    schema = pa.schema([("m", pa.map_(pa.string(), pa.int64()))])
    for value in ([{"k": 1, "j": 2}], [[("k", 1), ("j", 2)]]):
        got = run_outputs({"m": value}, schema)
        assert got.column("m").type == schema.field("m").type
        assert got.column("m").to_pylist() == [[("k", 1), ("j", 2)]]


def test_a_record_array_becomes_a_struct_column():
    # The one ndarray shape that means struct, and what an @njit function
    # returns for a numba record type; pa.array refuses it outright.
    records = np.array([(1, 2.5, "ab"), (3, 4.5, "c\x00d")],
                       dtype=[("i", "i8"), ("f", "f8"), ("s", "U3")])
    got = run_outputs({"r": records})
    assert got.column("r").type == pa.struct([("i", pa.int64()), ("f", pa.float64()), ("s", pa.string())])
    assert got.column("r").to_pylist() == [{"i": 1, "f": 2.5, "s": "ab"}, {"i": 3, "f": 4.5, "s": "c\x00d"}]
    declared = pa.schema([("r", pa.struct([("i", pa.int32()), ("s", pa.large_string()), ("extra", pa.int64())]))])
    got = run_outputs({"r": records[["i", "s"]]}, declared)
    assert got.column("r").type == declared.field("r").type
    assert got.column("r").to_pylist() == [{"i": 1, "s": "ab", "extra": None}, {"i": 3, "s": "c\x00d", "extra": None}]
    with pytest.raises(ValueError, match="'f'"):
        run_outputs({"r": records}, declared)


def test_a_record_array_with_no_fields_keeps_its_rows():
    # pa.StructArray.from_arrays([], names=[]) has no child to take a length
    # from, so the column came back with no rows and, as the only output
    # column, dropped the batch without a word.
    records = np.array([()] * 2, dtype=[])
    for schema in (None, pa.schema([("r", pa.struct([]))])):
        got = run_outputs({"r": records}, schema)
        assert got.column("r").type == pa.struct([])
        assert got.column("r").to_pylist() == [{}, {}]


def test_an_output_side_failure_names_its_column():
    with pytest.raises(pa.ArrowInvalid, match="'bad'"):
        run_outputs({"bad": np.zeros((2, 2))})
    with pytest.raises(pa.ArrowInvalid, match="'a'"):
        run_outputs({"a": np.array([1.5, 2.5])}, pa.schema([("a", pa.int64())]))
    with pytest.raises(TypeError, match="'a'"):
        run_outputs({"a": 3})


def test_main_func_must_return_a_dict():
    # Matched on the refusal's own words: "'NoneType' object is not iterable"
    # from the code after the guard also names NoneType.
    with pytest.raises(TypeError, match="main_func must return a dict.*NoneType"):
        run_outputs(None)


def test_an_all_none_object_column_keeps_a_declared_type():
    # Without a schema an object column of Nones infers null, the same trap an
    # empty string column used to fall into.
    schema = pa.schema([("s", pa.string())])
    got = run_outputs({"s": np.array([None, None], dtype=object)}, schema)
    assert got.column("s").type == pa.string() and got.column("s").null_count == 2
    assert run_outputs({"s": np.array([None, None], dtype=object)}).column("s").type == pa.null()


def test_a_nullable_carries_the_input_nulls_out():
    # A bare array republishes every null as the value under it, so [1, None, 3]
    # came back [1, 0, 3] with nothing said. Nullable folds the bitmap the UDF
    # was handed back in, with or without a declared type.
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
    # pc.if_else clears the validity bit and leaves the value's bytes in the
    # buffer, so a bare pass-through hands the masked-out value straight back.
    values = pa.array([1, 42, 3], type=pa.int64())
    column = pc.if_else(pa.array([True, False, True]), values, pa.scalar(None, pa.int64()))
    batch = pa.RecordBatch.from_arrays([column], names=["c"])

    def both(data_dict, bitmap_dict, broadcasts):
        return {"bare": data_dict["c"], "kept": Nullable(data_dict["c"], bitmap_dict["c"])}

    got = list(make_mapinarrow_func(both, input_columns=["c"])(iter([batch])))[0]
    assert got.column("bare").to_pylist() == [1, 42, 3]
    assert got.column("kept").to_pylist() == [1, None, 3]


def test_a_nullable_takes_the_bitmap_and_the_data_zero_copy():
    # A fixed-width column with no nulls of its own takes the bitmap as its
    # validity buffer and keeps the ndarray as its data buffer.
    data = np.array([1, 2, 3], dtype=np.int64)
    bitmap = np.array([0b101], dtype=np.uint8)
    got = run_outputs({"a": Nullable(data, bitmap)}).column("a")
    assert got.to_pylist() == [1, None, 3]
    assert got.buffers()[0].address == bitmap.ctypes.data
    assert got.buffers()[1].address == data.ctypes.data


def test_a_nullable_keeps_the_nulls_the_data_already_has():
    # A list holding None, a sliced Arrow array and a struct column cannot take
    # the bitmap as a buffer; they are masked instead, and keep their own nulls.
    bitmap = np.array([0b011], dtype=np.uint8)
    assert run_outputs({"a": Nullable([None, 2, 3], bitmap)}).column("a").to_pylist() == [None, 2, None]
    sliced = pa.array([9, 1, 2, 3], type=pa.int64())[1:]
    assert run_outputs({"a": Nullable(sliced, bitmap)}).column("a").to_pylist() == [1, 2, None]
    records = np.array([(1, 2.5), (3, 4.5)], dtype=[("i", "i8"), ("f", "f8")])
    got = run_outputs({"r": Nullable(records, np.array([0b10], dtype=np.uint8))}).column("r")
    assert got.to_pylist() == [None, {"i": 3, "f": 4.5}]


def test_a_nullable_dictionary_column_is_masked_rather_than_rebuilt():
    # pa.Array.from_buffers cannot rebuild a dictionary column: Arrow's C++
    # aborts the process on "Check failed: (data->dictionary) != (nullptr)",
    # taking the whole run with it rather than raising. The flat path is kept
    # off a dictionary type for that reason, and masking gives the values back
    # with the bitmap's nulls.
    values = pa.array(["a", "b", "c"]).dictionary_encode()
    bitmap = np.array([0b101], dtype=np.uint8)
    declared = pa.schema([("d", pa.dictionary(pa.int32(), pa.string()))])
    for schema in (None, declared):
        got = run_outputs({"d": Nullable(values, bitmap)}, schema).column("d")
        assert got.type == values.type, schema
        assert got.to_pylist() == ["a", None, "c"], schema
        assert got.null_count == 1, schema


def test_a_nullable_with_no_bitmap_is_the_bare_array():
    # bitmap_dict hands out None where the batch carries no validity buffer,
    # so passing it through must cost nothing and change nothing.
    got = run_outputs({"a": Nullable(np.array([1, 2], dtype=np.int64), None)}).column("a")
    assert got.to_pylist() == [1, 2] and got.null_count == 0
    empty = run_outputs({"a": Nullable(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.uint8))})
    assert empty.num_rows == 0 and empty.column("a").type == pa.int64()


def test_a_bitmap_of_the_wrong_length_is_refused():
    # The bitmap covers 8 rows per byte; one of any other size is a bitmap for
    # some other column, and pyarrow would read it without a word.
    data = np.array([1, 2, 3], dtype=np.int64)
    with pytest.raises(ValueError, match=r"'a'.*3 bytes.*24 rows.*3 rows"):
        run_outputs({"a": Nullable(data, np.zeros(3, dtype=np.uint8))})
    with pytest.raises(ValueError, match=r"'a'.*0 bytes"):
        run_outputs({"a": Nullable(data, np.zeros(0, dtype=np.uint8))})


def test_a_bitmap_that_is_not_packed_uint8_is_refused():
    # A boolean mask is the natural mistake, and its bytes would read as bits.
    data = np.array([1, 2, 3], dtype=np.int64)
    with pytest.raises(TypeError, match=r"'a'.*bool"):
        run_outputs({"a": Nullable(data, np.array([True, False, True]))})
    with pytest.raises(TypeError, match=r"'a'.*2-dimensional"):
        run_outputs({"a": Nullable(data, np.zeros((1, 1), dtype=np.uint8))})


def test_a_bitmap_that_is_not_an_ndarray_is_refused():
    # bitmap.dtype was read before anything was validated, and AttributeError
    # is not one of the classes the wrapper catches, so each of these escaped
    # as "'list' object has no attribute 'dtype'", naming neither the column
    # nor what a bitmap must be.
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
    # A packed bitmap cannot tell 2 rows from 3: both are one byte. A UDF that
    # drops a row and passes the batch's bitmap through would get the dropped
    # row's bit applied to the wrong row, silently, so that bitmap is only
    # accepted on a column of the batch's own row count. A bitmap the UDF
    # builds for its own rows goes through the byte check as before.
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
    # A list-of-struct column's bitmaps cover the flattened struct elements,
    # six of them here for two outer rows, so the documented pass-through
    # returns a column three times the batch's length and was refused as a
    # resize, with a message claiming the bitmap covered the batch's 2 rows. A
    # handed-out bitmap is checked against the count it was handed out for.
    ty = pa.list_(pa.struct([("a", pa.int64())]))
    column = pa.array([[{"a": 1}, {"a": None}, {"a": 3}], [{"a": 4}, {"a": 5}, {"a": 6}]], type=ty)
    batch = pa.RecordBatch.from_arrays([column], names=["s"])

    def passed_through(data_dict, bitmap_dict, broadcasts):
        return {"out": Nullable(data_dict["s"]["a"] * 2, bitmap_dict["s"]["a"])}

    got = list(make_mapinarrow_func(passed_through, input_columns=["s"])(iter([batch])))[0]
    assert got.column("out").to_pylist() == [2, None, 6, 8, 10, 12]

    def copied(data_dict, bitmap_dict, broadcasts):
        return {"out": Nullable(data_dict["s"]["a"] * 2, bitmap_dict["s"]["a"].copy())}

    # A copy is not the array the batch handed out, so it takes the byte check
    # instead; it answered this already while the pass-through was refused.
    same = list(make_mapinarrow_func(copied, input_columns=["s"])(iter([batch])))[0]
    assert same.column("out").to_pylist() == [2, None, 6, 8, 10, 12]

    def half(data_dict, bitmap_dict, broadcasts):
        return {"out": Nullable(data_dict["s"]["a"][:3], bitmap_dict["s"]["a"])}

    with pytest.raises(ValueError, match=r"'out'.*handed out.*6 rows.*3 rows"):
        list(make_mapinarrow_func(half, input_columns=["s"])(iter([batch])))


def test_a_bare_tuple_is_a_sequence_not_a_pair():
    # Only a Nullable is the pair. A tuple is the column it always was, a
    # 2-tuple ending in None or an array included, which a rule on bare tuples
    # would have read as data and bitmap.
    assert run_outputs({"a": (1, 2, 3)}).column("a").to_pylist() == [1, 2, 3]
    assert run_outputs({"a": ("x", "y")}).column("a").to_pylist() == ["x", "y"]
    assert run_outputs({"a": (5, None)}).column("a").to_pylist() == [5, None]
    listed = run_outputs({"a": ([1, 2], None)}).column("a")
    assert listed.type == pa.list_(pa.int64()) and listed.to_pylist() == [[1, 2], None]
    arrays = run_outputs({"a": (np.array([1, 2]), np.array([3, 4]))}).column("a")
    assert arrays.type == pa.list_(pa.int64()) and arrays.to_pylist() == [[1, 2], [3, 4]]
