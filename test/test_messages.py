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
