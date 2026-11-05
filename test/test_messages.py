"""Tests that numbarrow's error messages name what is at fault and stay short.

A refusal names the column, the struct field, the element, the parameter or the path to a nested field at fault. Where
a fix exists, the message names it too, such as ``combine_chunks()`` for a chunked array or an alias for a column that
appears twice. A message that shows a pyarrow type cuts it at ``TYPE_REPR_WIDTH`` characters, because the type of a
thousand-field struct runs to over 10,000 characters and appears in every log line that records the traceback. A list
of unexpected keys is cut to ten keys, with a count of the rest. Most tests call ``arrow_array_adapter``, the
dispatcher that picks an adapter by array type, or the function that ``make_mapinarrow_func`` returns, in this process.
One test runs a child under ``python -O``, which removes ``assert`` statements, and two tests leave out their pandas
cases where pandas is not installed. ``test_adapters.py``, ``test_arrow_array_utils.py`` and
``test_mapinarrow_factory.py`` test what is converted and what is refused; this file tests what the refusals say.
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
from test.conftest import Wrapped

REPO = Path(__file__).resolve().parent.parent


def _wide_struct(n, child=pa.int64()):
    return pa.struct([(f"field_{i}", child) for i in range(n)])


def test_a_wide_type_is_cut_in_the_message():
    """Check that ``type_repr`` leaves a short type whole and cuts a long one at ``TYPE_REPR_WIDTH`` characters, noting
    how many characters it cut and, for a struct, how many fields the struct has."""
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
    """Check that the messages stay short when the struct adapter refuses a thousand-field struct whose field names
    repeat, and when the list adapter refuses a list of thousand-field structs whose rows differ in length. The test
    also checks that the dispatcher names an unsupported ``large_list`` type."""
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
    """Check that a ``ChunkedArray``, which is what a Table column is, is refused as one, with ``combine_chunks`` named
    as the fix. A message about an array of its type would blame a type that numbarrow supports."""
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
    """Check that invalid UTF-8 raises ``ValueError`` naming the element and the array's length, instead of a bare
    ``UnicodeDecodeError`` whose position is a byte offset inside the element."""
    offsets = pa.py_buffer(np.array([0, 2, 4], dtype=np.int32).tobytes())
    data = pa.py_buffer(b"ok\xff\xfe")
    arr = pa.Array.from_buffers(pa.string(), 2, [None, offsets, data])
    with pytest.raises(ValueError, match="element 1 of a 2-element string array is not valid UTF-8"):
        arrow_array_adapter(arr)


def test_the_type_guards_survive_dash_O():
    """Check that the struct and list adapters refuse an array of another type with ``TypeError`` under ``python -O``.
    Python removes ``assert`` statements under ``-O``, so a type check written as one would let any array through, to
    fail later on an unrelated error."""
    # -O applies to a whole interpreter, so the check runs in a child process.
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
    # A str is iterable, so "value" would otherwise be read as the five columns v, a, l, u and e.
    with pytest.raises(TypeError, match="list of column names"):
        make_mapinarrow_func(lambda d, b, br: {}, input_columns="value")


def test_an_empty_input_columns_is_refused():
    """Check that an ``input_columns`` that names no column, such as a generator already used up or a filter that
    matched nothing, is refused when the function is made. Otherwise ``main_func`` would get no columns: one that reads
    ``data_dict["x"]`` would fail on a bare ``KeyError``, and one that transforms whatever arrives would return a batch
    of no rows and no columns without an error."""
    used_up = (name for name in ["x"])
    list(used_up)
    for empty in ([], used_up, filter(lambda name: name.startswith("feat_"), ["x", "y"])):
        with pytest.raises(ValueError, match="input_columns names no column"):
            make_mapinarrow_func(lambda d, b, br: {}, input_columns=empty)


def test_an_output_schema_that_is_not_a_pyarrow_schema_is_refused():
    """Check that an ``output_schema`` that is not a ``pyarrow.Schema`` is refused when the function is made, naming
    the type it needs.

    ``mapInArrow`` itself takes a PySpark ``StructType`` as its schema, so a caller may pass one here. Without the
    check, the factory would fail on ``AttributeError: 'StructField' object has no attribute 'type'``, which names
    neither the parameter nor the type it needs.
    """
    # Field and Schema stand in for pyspark's StructField and StructType, so the test runs without pyspark.
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
    """Check that a column missing from the batch raises ``KeyError`` listing the batch's columns. Spark matches column
    names without regard to case and passes on the name as the DataFrame spells it, so a UDF that asks for "value" can
    get a batch whose column is "Value"."""
    fn = make_mapinarrow_func(lambda d, b, br: {}, input_columns=["value"])
    with pytest.raises(KeyError, match=r"'value'.*\['Value', 'id'\]"):
        list(fn(iter([_batch(Value=[1.0], id=[1])])))


def test_a_key_error_message_reads_as_written():
    """Check that numbarrow's ``KeyError`` messages read as written. ``str()`` of a plain ``KeyError`` is the repr of
    its argument, so it wraps a message in a second pair of quotes, and a message rebuilt from that ``str()`` arrives
    quoted twice, with its inner quotes escaped."""
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
    # A batch built outside Spark can hold two columns of one name. batch.column would raise pyarrow's own KeyError,
    # which suggests no fix, so the function refuses such a batch first and suggests an alias.
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
    # UnicodeDecodeError takes five constructor arguments, so rebuilding it from the prefixed message would raise a
    # second error, and the prefix would be lost.
    wrapped = renamed(UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad"), "column 'x'")
    assert type(wrapped) is ValueError
    assert str(wrapped).startswith("column 'x': ")


def test_a_scalar_string_output_is_refused_rather_than_spread():
    # pa.array spreads a str one character per row, so {"country": "US"} over a two-row batch would become the rows "U"
    # and "S"; it spreads a bytes object into one integer per byte. The factory converts a numpy unicode array through
    # tolist(). pa.array refuses a 0-d or 2-d array, but not the str or the nested lists that tolist() returns, so the
    # factory checks the dimension itself.
    batch = _batch(v=[1, 2])
    for value in ("US", b"US", np.str_("US"), np.array("US"), np.array([["a", "b"]])):
        fn = make_mapinarrow_func(lambda d, b, br, value=value: {"country": value})
        with pytest.raises(TypeError, match="'country'"):
            list(fn(iter([batch])))


def test_a_union_field_under_a_struct_is_refused_before_flatten():
    # StructArray.flatten() gives each child the struct's validity, and a union has no validity bitmap to take it.
    # Flattening a struct that has a null row and a union field therefore aborts the process in Arrow's C++ code, so
    # the adapter refuses a union field before it flattens anything, with or without a null row.
    types = pa.array([0, 1, 0], type=pa.int8())
    union = pa.UnionArray.from_sparse(types, [pa.array([1, 2, 3]), pa.array(["a", "b", "c"])])
    for mask in (None, pa.array([False, True, False])):
        struct = pa.StructArray.from_arrays([pa.array([1, 2, 3]), union], names=["ok", "u"], mask=mask)
        with pytest.raises(NotImplementedError, match=r"struct field 'u'.*union"):
            arrow_array_adapter(struct)
    # An extension type over a union is not a union type itself, so the check looks through the extension type to its
    # storage.
    wrapped = pa.ExtensionArray.from_storage(Wrapped(union.type), union)
    struct = pa.StructArray.from_arrays([pa.array([1, 2, 3]), wrapped], names=["ok", "u"])
    with pytest.raises(NotImplementedError, match=r"struct field 'u'.*union"):
        arrow_array_adapter(struct)


def test_the_unexpected_keys_listing_is_cut_with_a_count():
    # A UDF that keys a dict by a row value gives one unexpected key per row. A full list of 100,000 keys like these
    # would put 1.5 MB into the message, so the refusal shows ten keys and counts the rest.
    rows = [{f"user_{i:06d}": 1} for i in range(1000)]
    fn = make_mapinarrow_func(lambda d, b, br: {"counts": rows},
                              output_schema=pa.schema([("counts", pa.struct([("total", pa.int64())]))]))
    with pytest.raises(ValueError) as excinfo:
        list(fn(iter([_batch(v=list(range(1000)))])))
    message = str(excinfo.value)
    assert "'user_000000'" in message and "and 990 more" in message and len(message) < 600


def test_the_dispatcher_describes_a_scalar_and_leaks_no_type_column():
    """Check that the dispatcher refuses a pyarrow scalar by asking for the Array, and describes a pandas DataFrame by
    its class alone.

    A null list scalar has no length to read, and a struct scalar would otherwise be described as an array of a type
    that numbarrow supports. A DataFrame answers ``.type`` with its column called type, whose values would then appear
    in the message.
    """
    for scalar in (pa.array([None], type=pa.list_(pa.int64()))[0], pa.array([{"a": 1}])[0]):
        with pytest.raises(NotImplementedError, match=r"Scalar of type .*: pass the Array"):
            arrow_array_adapter(scalar)
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"type": [f"secret-{i}" for i in range(50)], "v": range(50)})
    with pytest.raises(NotImplementedError, match="DataFrame, which is not a pyarrow Array") as excinfo:
        arrow_array_adapter(frame)
    assert "secret" not in str(excinfo.value)


def test_an_output_schema_naming_a_field_twice_is_refused_at_factory_time():
    # The dict that main_func returns holds one value per name, so one entry would fill every copy of the name. Spark
    # would then fail in the JVM with an error that names neither the column nor the repeated name.
    with pytest.raises(ValueError, match=r"\['price'\] more than once"):
        make_mapinarrow_func(lambda d, b, br: {}, output_schema=pa.schema([("price", pa.float64()), ("price", pa.int32())]))
    nested = pa.schema([("s", pa.list_(pa.struct([("x", pa.int64()), ("x", pa.float64())])))])
    with pytest.raises(ValueError, match=r"\['x'\] more than once"):
        make_mapinarrow_func(lambda d, b, br: {}, output_schema=nested)
    # The check also looks inside a map's entries, and through an extension type to its storage.
    twice = pa.struct([("a", pa.int64()), ("a", pa.int64())])
    for shape in (pa.map_(pa.string(), twice), Wrapped(twice)):
        with pytest.raises(ValueError, match=r"\['a'\] more than once"):
            make_mapinarrow_func(lambda d, b, br: {}, output_schema=pa.schema([("s", shape)]))


def test_the_function_names_the_shape_it_takes_when_handed_a_batch_or_a_table():
    """Check that the function refuses a RecordBatch, a Table or a list of pandas DataFrames with ``TypeError`` saying
    it takes an iterator of RecordBatch, and accepts ``[batch]``. Iterating a batch or a table yields its columns, so
    without the check the function would fail on a missing attribute of the first column. A list of the DataFrames
    that ``mapInPandas`` passes would fail the same way."""
    fn = make_mapinarrow_func(lambda d, b, br: {"out": d["a"]})
    batch = pa.RecordBatch.from_pydict({"a": [1, 2, 3]})
    handed = [batch, pa.Table.from_batches([batch])]
    try:
        import pandas  # noqa: F401
        handed.append([batch.to_pandas()])
    except ImportError:
        # The CI job that tests the ends of the pyarrow range installs no pandas.
        pass
    for shape in handed:
        with pytest.raises(TypeError, match="iterator of pyarrow.RecordBatch"):
            list(fn(shape))
    assert list(fn([batch]))[0].column("out").to_pylist() == [1, 2, 3]


def test_a_nested_key_refusal_names_the_path_to_the_field():
    """Check that the refusal of a misspelled nested key names the path to the field, down to a map's value. Without
    the path, two sibling fields of one type, a map's key and its value, and fields at different depths would all get
    the same message, naming the column alone."""
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


def test_the_dispatcher_cuts_a_wide_type_for_a_chunked_array():
    """Check that the message stays short when the dispatcher refuses a ChunkedArray of a thousand-field struct. The
    other wide-type tests check the adapters' messages, and the dispatcher would put every field into its own message
    if it spelled out the type in full."""
    column = pa.chunked_array([pa.array([{f"field_{i}": 1 for i in range(1000)}], type=_wide_struct(1000))])
    with pytest.raises(NotImplementedError) as excinfo:
        arrow_array_adapter(column)
    assert "ChunkedArray" in str(excinfo.value) and len(str(excinfo.value)) < 400
