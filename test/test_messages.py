"""What an error says, and about what.

Every refusal here was the right refusal with the wrong message: a Table
column blamed a supported type and never said chunked, a struct child's
failure named the type but not the field, invalid UTF-8 escaped as a bare
UnicodeDecodeError whose position was a byte offset inside the element,
input_columns="value" was iterated character by character, a column name
Spark's case-insensitive projection had rewritten died on a bare KeyError,
and a thousand-field struct put thirteen thousand characters into every log
line that caught the traceback.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from numbarrow.core.adapters import arrow_array_adapter
from numbarrow.core.mapinarrow_factory import make_mapinarrow_func
from numbarrow.utils.arrow_array_utils import TYPE_REPR_WIDTH, renamed, type_repr

REPO = Path(__file__).resolve().parent.parent


def _wide_struct(n, child=pa.int64()):
    return pa.struct([(f"field_{i}", child) for i in range(n)])


def test_a_wide_type_is_cut_in_the_message():
    narrow = pa.struct([("a", pa.int64())])
    assert type_repr(narrow) == str(narrow)
    wide = _wide_struct(1000)
    text = type_repr(wide)
    assert len(text) < TYPE_REPR_WIDTH + 60
    assert text.startswith(str(wide)[:TYPE_REPR_WIDTH]) and "1000 fields" in text
    assert f"{len(str(wide)) - TYPE_REPR_WIDTH} more characters" in text
    wide_list = pa.list_(wide)
    listed = type_repr(wide_list)
    assert len(listed) < TYPE_REPR_WIDTH + 60
    assert f"{len(str(wide_list)) - TYPE_REPR_WIDTH} more characters" in listed


def test_the_dispatcher_and_the_adapters_cut_a_wide_type():
    names = ["v", "v"] + [f"f{i}" for i in range(1000)]
    dup = pa.StructArray.from_arrays([pa.array([1])] * len(names), names=names)
    with pytest.raises(NotImplementedError) as excinfo:
        arrow_array_adapter(dup)
    assert "repeated field names" in str(excinfo.value) and len(str(excinfo.value)) < 400
    row = {f"field_{i}": 1 for i in range(1000)}
    ragged = pa.array([[row], []], type=pa.list_(_wide_struct(1000)))
    with pytest.raises(NotImplementedError) as excinfo:
        arrow_array_adapter(ragged)
    assert "not all the same length" in str(excinfo.value) and len(str(excinfo.value)) < 500
    with pytest.raises(NotImplementedError) as excinfo:
        arrow_array_adapter(pa.array([[1]], type=pa.large_list(pa.int64())))
    assert "large_list" in str(excinfo.value)


def test_a_chunked_array_is_named_as_such():
    chunked = pa.chunked_array([[1, 2], [3]], type=pa.int64())
    with pytest.raises(NotImplementedError, match="ChunkedArray.*combine_chunks"):
        arrow_array_adapter(chunked)
    with pytest.raises(NotImplementedError, match="ChunkedArray"):
        arrow_array_adapter(pa.table({"v": [1, 2]}).column("v"))


def test_a_struct_child_failure_names_the_field():
    arr = pa.array([{"f": 1.0}], type=pa.struct([("f", pa.float32())]))
    with pytest.raises(NotImplementedError, match="struct field 'f'.*float"):
        arrow_array_adapter(arr)


def test_invalid_utf8_names_the_element():
    offsets = pa.py_buffer(np.array([0, 2, 4], dtype=np.int32).tobytes())
    data = pa.py_buffer(b"ok\xff\xfe")
    arr = pa.Array.from_buffers(pa.string(), 2, [None, offsets, data])
    with pytest.raises(ValueError, match="element 1 of a 2-element string array is not valid UTF-8"):
        arrow_array_adapter(arr)


def test_the_type_guards_survive_dash_O():
    # Bare asserts vanish under -O, so the struct and list adapters accepted
    # any array and died later on an error about something else.
    src = (
        "import pyarrow as pa\n"
        "from numbarrow.utils.arrow_array_utils import structured_array_adapter, structured_list_array_adapter\n"
        "for adapter in (structured_array_adapter, structured_list_array_adapter):\n"
        "    try:\n"
        "        adapter(pa.array([1, 2]))\n"
        "    except TypeError as exc:\n"
        "        print('typed', 'Int64Array' in str(exc))\n"
    )
    env = dict(os.environ, PYTHONPATH=str(REPO))
    out = subprocess.run([sys.executable, "-O", "-c", src], capture_output=True, text=True, env=env)
    assert out.stdout.split() == ["typed", "True", "typed", "True"], out


def _batch(**columns):
    return pa.RecordBatch.from_pydict(columns)


def test_input_columns_as_a_string_is_refused():
    # A str is iterable, so "value" was read as the columns v, a, l, u, e.
    with pytest.raises(TypeError, match="list of column names"):
        make_mapinarrow_func(lambda d, b, br: {}, input_columns="value")


def test_an_output_schema_that_is_not_a_pyarrow_schema_is_refused():
    # A PySpark StructType is what the README's own example calls
    # output_schema, and it carries .names too, so it got as far as the first
    # batch and died there on a bare "'StructField' object has no attribute
    # 'type'", naming neither the parameter nor the type it needs.
    class Field:
        def __init__(self, name):
            self.name, self.dataType = name, object()

    class Schema:
        def __init__(self, fields):
            self.fields = fields
            self.names = [field.name for field in fields]

        def __iter__(self):
            return iter(self.fields)

    with pytest.raises(TypeError, match="output_schema must be a pyarrow.Schema, not a Schema"):
        make_mapinarrow_func(lambda d, b, br: {}, output_schema=Schema([Field("out")]))


def test_a_missing_input_column_names_the_batch_columns():
    # Spark's projection is case-insensitive and rewrites the name it was
    # given, so the README's own example died on KeyError: 'value'.
    fn = make_mapinarrow_func(lambda d, b, br: {}, input_columns=["value"])
    with pytest.raises(KeyError, match=r"'value'.*\['Value', 'id'\]"):
        list(fn(iter([_batch(Value=[1.0], id=[1])])))


def test_a_key_error_message_reads_as_written():
    # KeyError.__str__ reprs its argument, which wrapped both messages in a
    # second pair of quotes, and mangles a message renamed through it.
    fn = make_mapinarrow_func(lambda d, b, br: {}, input_columns=["value"])
    with pytest.raises(KeyError) as excinfo:
        list(fn(iter([_batch(Value=[1.0])])))
    assert str(excinfo.value).startswith("column 'value' is not in this batch")
    schema = pa.schema([("b", pa.int64())])
    fn = make_mapinarrow_func(lambda d, b, br: {}, input_columns=["a"], output_schema=schema)
    with pytest.raises(KeyError) as excinfo:
        list(fn(iter([_batch(a=[1])])))
    assert str(excinfo.value).startswith("output_schema names column 'b'")
    wrapped = renamed(KeyError('Field "nope" does not exist in schema'), "prefix")
    assert isinstance(wrapped, KeyError)
    assert str(wrapped) == 'prefix: Field "nope" does not exist in schema'


def test_a_column_the_batch_carries_twice_is_refused_with_a_remedy():
    # An unaliased join produces this shape, and batch.column died on
    # pyarrow's own KeyError outside the code that names the column.
    batch = pa.RecordBatch.from_arrays([pa.array([1, 2]), pa.array([10, 20])], names=["id", "id"])
    fn = make_mapinarrow_func(lambda d, b, br: {})
    with pytest.raises(ValueError, match="'id' appears 2 times.*alias"):
        list(fn(iter([batch])))


def test_an_adapter_failure_names_the_column():
    fn = make_mapinarrow_func(lambda d, b, br: {})
    with pytest.raises(NotImplementedError, match="column 'x'.*float"):
        list(fn(iter([_batch(x=pa.array([1.0], type=pa.float32()))])))
    nested = pa.array([{"f": 1.0}], type=pa.struct([("f", pa.float32())]))
    with pytest.raises(NotImplementedError, match="column 's'.*struct field 'f'"):
        list(fn(iter([_batch(s=nested)])))


def test_an_exception_that_cannot_be_rebuilt_from_a_message_is_renamed_as_a_value_error():
    # A UnicodeDecodeError takes five constructor arguments, so rebuilding it
    # from the prefixed message would raise a second error and lose the prefix.
    wrapped = renamed(UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad"), "column 'x'")
    assert type(wrapped) is ValueError
    assert str(wrapped).startswith("column 'x': ")


def test_a_scalar_string_output_is_refused_rather_than_spread():
    # {"country": "US"} over a two-row batch came back as the rows "U" and "S",
    # and a 0-d unicode array's tolist() is that scalar, which defeated
    # pa.array's own refusal of a 0-d array.
    batch = _batch(v=[1, 2])
    for value in ("US", b"US", np.str_("US"), np.array("US"), np.array([["a", "b"]])):
        fn = make_mapinarrow_func(lambda d, b, br, value=value: {"country": value})
        with pytest.raises(TypeError, match="'country'"):
            list(fn(iter([batch])))


def test_a_union_field_under_a_struct_is_refused_before_flatten():
    # flatten() hands the struct's validity to each child, and a union carries
    # none, so Arrow's C++ layer aborted the process under a struct with a null
    # row where the typed refusal was due.
    types = pa.array([0, 1, 0], type=pa.int8())
    union = pa.UnionArray.from_sparse(types, [pa.array([1, 2, 3]), pa.array(["a", "b", "c"])])
    for mask in (None, pa.array([False, True, False])):
        struct = pa.StructArray.from_arrays([pa.array([1, 2, 3]), union], names=["ok", "u"], mask=mask)
        with pytest.raises(NotImplementedError, match=r"struct field 'u'.*union"):
            arrow_array_adapter(struct)


def test_the_unexpected_keys_listing_is_cut_with_a_count():
    # A UDF keying a dict by a row value put every key of the batch into the
    # exception, 1.5 MB for 100,000 rows, and twice into the executor logs.
    rows = [{f"user_{i:06d}": 1} for i in range(1000)]
    fn = make_mapinarrow_func(lambda d, b, br: {"counts": rows},
                              output_schema=pa.schema([("counts", pa.struct([("total", pa.int64())]))]))
    with pytest.raises(ValueError) as excinfo:
        list(fn(iter([_batch(v=list(range(1000)))])))
    message = str(excinfo.value)
    assert "'user_000000'" in message and "and 990 more" in message and len(message) < 600


def test_the_dispatcher_describes_a_scalar_and_leaks_no_type_column():
    # A null list scalar died on len(), a struct scalar of a supported column
    # was described as an unsupported array of that type, and a frame with a
    # column called type put that column's values into the message.
    for scalar in (pa.array([None], type=pa.list_(pa.int64()))[0], pa.array([{"a": 1}])[0]):
        with pytest.raises(NotImplementedError, match=r"Scalar of type .*: pass the Array"):
            arrow_array_adapter(scalar)
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"type": [f"secret-{i}" for i in range(50)], "v": range(50)})
    with pytest.raises(NotImplementedError, match="DataFrame, which is not a pyarrow Array") as excinfo:
        arrow_array_adapter(frame)
    assert "secret" not in str(excinfo.value)


def test_an_output_schema_naming_a_field_twice_is_refused_at_factory_time():
    # One dict entry filled every copy, and Spark died in the JVM with "not
    # all nodes and buffers were consumed", naming neither the column nor the
    # repeated name.
    with pytest.raises(ValueError, match=r"\['price'\] more than once"):
        make_mapinarrow_func(lambda d, b, br: {}, output_schema=pa.schema([("price", pa.float64()), ("price", pa.int32())]))
    nested = pa.schema([("s", pa.list_(pa.struct([("x", pa.int64()), ("x", pa.float64())])))])
    with pytest.raises(ValueError, match=r"\['x'\] more than once"):
        make_mapinarrow_func(lambda d, b, br: {}, output_schema=nested)


def test_the_function_names_the_shape_it_takes_when_handed_a_batch_or_a_table():
    # udf(batch) and udf(table) walked the columns and died on an attribute of
    # the first one, and mapInPandas's frames died the same way.
    fn = make_mapinarrow_func(lambda d, b, br: {"out": d["a"]})
    batch = pa.record_batch({"a": pa.array([1, 2, 3])})
    for handed in (batch, pa.Table.from_batches([batch]), [batch.to_pandas()]):
        with pytest.raises(TypeError, match="iterator of pyarrow.RecordBatch"):
            list(fn(handed))
    assert list(fn([batch]))[0].column("out").to_pylist() == [1, 2, 3]


def test_a_nested_key_refusal_names_the_path_to_the_field():
    # Two same-typed sibling fields, a map's key and its value, and different
    # depths all raised a byte-identical message naming the column alone.
    inner = pa.struct([("amount", pa.int64())])
    schema = pa.schema([("s", pa.struct([("a", inner), ("b", inner), ("m", pa.map_(pa.string(), inner))]))])
    rows = [{"a": {"amount": 1}, "b": {"Amount": 2}, "m": [("k", {"amount": 3})]}]
    fn = make_mapinarrow_func(lambda d, b, br: {"s": rows}, output_schema=schema)
    with pytest.raises(ValueError, match=r"output column 's': field 'b': declared"):
        list(fn(iter([_batch(v=[1])])))
    rows = [{"a": {"amount": 1}, "b": {"amount": 2}, "m": [("k", {"Amount": 3})]}]
    fn = make_mapinarrow_func(lambda d, b, br: {"s": rows}, output_schema=schema)
    with pytest.raises(ValueError, match=r"output column 's': field 'm': map value: declared"):
        list(fn(iter([_batch(v=[1])])))
