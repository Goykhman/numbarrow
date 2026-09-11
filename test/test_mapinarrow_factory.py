import numpy as np
import pyarrow as pa
import pytest

from numbarrow.core.is_null import is_null
from numbarrow.core.mapinarrow_factory import make_mapinarrow_func


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


def test_output_schema_refuses_a_struct_key_no_field_has():
    # Arrow matches struct fields by exact name and nulls a missing one, so a
    # list of dicts keyed Amount/Label against amount/label used to come back
    # as two columns of nulls under an identical schema.
    schema = pa.schema([("s", pa.struct([("amount", pa.int64()), ("label", pa.string())]))])
    got = run_outputs({"s": [{"amount": 1, "label": "x"}, {"amount": 2}]}, schema)
    assert got.column("s").to_pylist() == [{"amount": 1, "label": "x"}, {"amount": 2, "label": None}]
    with pytest.raises(ValueError, match="Amount"):
        run_outputs({"s": [{"Amount": 1, "Label": "x"}]}, schema)


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


def test_an_output_side_failure_names_its_column():
    with pytest.raises(pa.ArrowInvalid, match="'bad'"):
        run_outputs({"bad": np.zeros((2, 2))})
    with pytest.raises(pa.ArrowInvalid, match="'a'"):
        run_outputs({"a": np.array([1.5, 2.5])}, pa.schema([("a", pa.int64())]))
    with pytest.raises(TypeError, match="'a'"):
        run_outputs({"a": 3})


def test_main_func_must_return_a_dict():
    with pytest.raises(TypeError, match="NoneType"):
        run_outputs(None)


def test_an_all_none_object_column_keeps_a_declared_type():
    # Without a schema an object column of Nones infers null, the same trap an
    # empty string column used to fall into.
    schema = pa.schema([("s", pa.string())])
    got = run_outputs({"s": np.array([None, None], dtype=object)}, schema)
    assert got.column("s").type == pa.string() and got.column("s").null_count == 2
    assert run_outputs({"s": np.array([None, None], dtype=object)}).column("s").type == pa.null()
