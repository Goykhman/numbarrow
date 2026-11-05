"""Tests that hold the README and the docstring of ``make_mapinarrow_func`` to what the code does.

The README's Supported Types table is parsed row by row. Its type names must match the array types that the
dispatcher, ``arrow_array_adapter``, registers, and its Copy? column must say whether each adapter returns fresh memory
or a view of the Arrow buffer. Other tests find a phrase of the factory docstring or of the README verbatim, and then
run the conversion that the phrase describes. The rest compare the README's pandas row with the floors in
``pyproject.toml``, and check in a child process when numba reads ``NUMBA_CACHE_DIR``. A test fails when the text and
the behavior disagree. A rewording that drops a pinned phrase fails as well, and the phrase in the test then changes
with it. ``test_mapinarrow_spark.py`` pins two more phrases of the factory docstring and checks them against Spark.
"""
import datetime
import decimal
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from numbarrow.core.adapters import arrow_array_adapter
from numbarrow.core.mapinarrow_factory import Nullable, make_mapinarrow_func

README = Path(__file__).resolve().parent.parent / "README.md"
PYPROJECT = README.parent / "pyproject.toml"

# The docstring with its white space collapsed, so that a phrase that wraps across lines matches as one string.
FACTORY_DOC = " ".join(make_mapinarrow_func.__doc__.split())

# The README with its white space collapsed the same way.
README_TEXT = " ".join(README.read_text().split())

SUB_SECOND = datetime.datetime(2020, 1, 1, 12, 34, 56, 789012)

# One sample array for each type that the README table names. A row added to the table without a sample here fails
# test_every_documented_type_has_a_sample.
SAMPLES = {
    "Int32Array": pa.array([1, 2], type=pa.int32()),
    "Int64Array": pa.array([1, 2], type=pa.int64()),
    "DoubleArray": pa.array([1.0, 2.0], type=pa.float64()),
    "BooleanArray": pa.array([True, False], type=pa.bool_()),
    "Date32Array": pa.array([1, 2], type=pa.int32()).cast(pa.date32()),
    "Date64Array": pa.array([86400000], type=pa.int64()).cast(pa.date64()),
    "TimestampArray": pa.array([1], type=pa.int64()).cast(pa.timestamp("ms")),
    "UInt8Array": pa.array([1, 2], type=pa.uint8()),
    "StringArray": pa.array(["a", "b"], type=pa.string()),
    "LargeStringArray": pa.array(["a", "b"], type=pa.large_string()),
    "StructArray": pa.array([{"a": 1}], type=pa.struct([("a", pa.int64())])),
    "ListArray": pa.array([[{"a": 1}]], type=pa.list_(pa.struct([("a", pa.int64())]))),
}


def _table_rows():
    """Return the README's Supported Types table as a list of (type names, Copy? cell) pairs, one per row."""
    text = README.read_text()
    section = text.split("## Supported Types", 1)[1].split("\n## ", 1)[0]
    rows = []
    for line in section.splitlines():
        if not line.startswith("|") or line.startswith("|---") or "PyArrow Type" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        names = re.findall(r"`(\w+Array)`", cells[0])
        if names:
            rows.append((names, cells[2]))
    return rows


def _documented_types():
    return {n for names, _ in _table_rows() for n in names}


def _registered_types():
    """Return the names of the array types that the dispatcher registers, leaving out its fallback for ``object``."""
    return {k.__name__ for k in arrow_array_adapter.registry if k is not object}


def test_every_registered_type_is_documented():
    missing = _registered_types() - _documented_types()
    assert not missing, f"dispatcher handles types the README table omits: {sorted(missing)}"


def test_every_documented_type_is_registered():
    extra = _documented_types() - _registered_types()
    assert not extra, f"README table claims types the dispatcher does not handle: {sorted(extra)}"


def test_every_documented_type_has_a_sample():
    rows = _documented_types()
    assert not rows - set(SAMPLES), f"table rows with no sample here: {sorted(rows - set(SAMPLES))}"


def _is_view(array):
    """Return True when ``array`` views an Arrow buffer, and False when it owns fresh memory.

    A view has a memoryview or a ``pa.Buffer`` within eight steps along its chain of ``.base`` objects.
    """
    node = array
    for _ in range(8):
        if isinstance(node, (memoryview, pa.Buffer)):
            return True
        node = getattr(node, "base", None)
        if node is None:
            return False
    return False


@pytest.mark.parametrize("names,copy_answer", _table_rows())
def test_documented_copy_column_matches_reality(names, copy_answer):
    for name in names:
        result = arrow_array_adapter(SAMPLES[name])
        if len(result) == 3:
            # The adapter returns three items for a struct or list column, whose fields it adapts one by one. The
            # scalar rows already cover the fields' types, so these rows are held only to their per-field label.
            assert copy_answer.lower().startswith("per-field"), (
                f"{name}: adapts per field, README says {copy_answer!r}"
            )
            continue
        # A cell that starts with neither yes nor no fails the test instead of being skipped, so a reworded cell cannot
        # switch its row's check off.
        assert copy_answer.lower().startswith(("yes", "no")), (
            f"{name}: README says {copy_answer!r}, which is neither yes nor no"
        )
        documented_copy = copy_answer.lower().startswith("yes")
        data = result[1]
        assert isinstance(data, np.ndarray), name
        measured_copy = not _is_view(data)
        assert measured_copy == documented_copy, (
            f"{name}: README says {'copy' if documented_copy else 'view'}, "
            f"measured {'copy' if measured_copy else 'view'}"
        )


def _one_output_column(value, arrow_type):
    """Return the column that a UDF returning ``value`` produces under a declared ``arrow_type``."""
    batch = pa.RecordBatch.from_pydict({"v": [1.0]})
    fn = make_mapinarrow_func(lambda d, b, br: {"out": value},
                              output_schema=pa.schema([("out", arrow_type)]))
    return list(fn(iter([batch])))[0].column("out")


def test_the_refusals_hold_for_the_shapes_the_docstring_names_them_for():
    """Check that a fraction into ``int64`` and a sub-second timestamp into a seconds unit raise ``ArrowInvalid`` from
    an ndarray, a shape for which the factory docstring says they raise. ``pa.array`` has two converters: the typed
    converter that an ndarray goes through raises on these two conversions, and the sequence converter that a list goes
    through truncates both without an error."""
    assert "an ndarray of a numeric or datetime dtype goes through ``pa.array``'s typed converter" in FACTORY_DOC
    assert ("Both raise :class:`pyarrow.ArrowInvalid` for an integer out of the declared type's range, a float with "
            "a fraction into an integer type, and a timestamp unit change that drops digits") in FACTORY_DOC
    with pytest.raises(pa.ArrowInvalid):
        _one_output_column(np.array([1.5]), pa.int64())
    with pytest.raises(pa.ArrowInvalid):
        _one_output_column(np.array([SUB_SECOND], dtype="datetime64[us]"), pa.timestamp("s"))


def test_the_truncations_the_docstring_admits_for_a_list_are_the_ones_it_makes():
    """Check that a list truncates a fraction into ``int64`` and a sub-second timestamp into a seconds unit without an
    error, the two truncations that the factory docstring admits for a list."""
    assert "goes through ``pa.array``'s sequence converter" in FACTORY_DOC
    assert ("from a list alone, a fraction into an integer type and a timestamp unit change "
            "that drops digits") in FACTORY_DOC
    assert _one_output_column([1.5], pa.int64()).to_pylist() == [1]
    assert _one_output_column([SUB_SECOND], pa.timestamp("s")).to_pylist() == [
        SUB_SECOND.replace(microsecond=0)
    ]


def test_a_float_under_a_declared_decimal_is_refused_as_the_docstring_says():
    """Check that a float under a declared ``decimal`` type is refused from a list and from an ndarray, as the factory
    docstring says, and that the docstring does not count it among the conversions that pass without an error. The
    sequence converter takes only an int or a ``decimal.Decimal`` under a decimal type, and the ndarray route refuses a
    float dtype; an exact ``Decimal`` passes."""
    assert "A float under a ``decimal`` type is refused on both routes" in FACTORY_DOC
    assert "a float into ``decimal``" not in FACTORY_DOC
    with pytest.raises(pa.ArrowException):
        _one_output_column([1.236], pa.decimal128(6, 2))
    with pytest.raises(pa.ArrowException):
        _one_output_column(np.array([1.236]), pa.decimal128(6, 2))
    exact = _one_output_column([decimal.Decimal("1.23")], pa.decimal128(6, 2))
    assert exact.to_pylist() == [decimal.Decimal("1.23")]


def _inferred_output_column(value):
    """Return the column that a UDF returning ``value`` produces with no ``output_schema``."""
    batch = pa.RecordBatch.from_pydict({"v": [1.0]})
    fn = make_mapinarrow_func(lambda d, b, br: {"out": value})
    return list(fn(iter([batch])))[0].column("out")


def test_an_inferred_datetime64_column_comes_back_as_the_docs_say():
    """Check that an inferred ``datetime64`` column comes back ``date32`` for the day unit, and a timestamp of its unit
    for seconds through nanoseconds, as the factory docstring and the README both say. ``pa.array`` infers ``date32``
    from a day-unit array, so a timestamp promised for every unit would be wrong for that one."""
    assert ("A ``datetime64`` array comes back a naive ``timestamp`` of its unit after the conversion above, so "
            "``datetime64[5s]`` comes back ``timestamp[s]`` and an hour or minute unit comes back in seconds. A day, "
            "week, month or year unit comes back ``date32``.") in FACTORY_DOC
    assert "A unit finer than a nanosecond raises :class:`TypeError`" in FACTORY_DOC
    assert ("a `datetime64` output becomes a timestamp of the same unit, without a time zone. A unit with a "
            "multiplier, such as `datetime64[5s]`, is converted to its base unit, and an hour or minute unit to "
            "seconds. A day, week, month or year unit becomes `date32`. A unit finer than a nanosecond raises "
            "`TypeError`.") in README_TEXT
    days = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")
    column = _inferred_output_column(days)
    assert column.type == pa.date32()
    assert column.to_pylist() == [datetime.date(2020, 1, 1), datetime.date(2020, 1, 2)]
    midnights = [datetime.datetime(2020, 1, 1), datetime.datetime(2020, 1, 2)]
    for unit in ("s", "ms", "us", "ns"):
        column = _inferred_output_column(days.astype(f"datetime64[{unit}]"))
        assert column.type == pa.timestamp(unit), unit
        assert column.to_pylist() == midnights, unit


def test_the_readme_names_the_pandas_floor_the_extras_declare():
    # The floors are read from pyproject.toml, so the test follows a change to the extras. The README must not
    # mention 1.5.0, a pandas floor that the extras do not declare.
    floors = set(re.findall(r'"pandas>=([\d.]+)"', PYPROJECT.read_text()))
    row = re.search(r"\| pandas \| ([\d.]+)\+", README.read_text()).group(1)
    assert floors == {row}, (floors, row)
    assert "1.5.0" not in README_TEXT


def test_the_cache_dir_is_read_at_numbas_import_as_the_readme_says(tmp_path):
    """Check that numba reads ``NUMBA_CACHE_DIR`` when numba is imported, as the README says, so that a directory set
    after numba is imported and before numbarrow is misses ``is_null``, the first function that numbarrow compiles."""
    # numba reads the variable again only when it compiles something, and it sets up a function's cache before it
    # compiles the function. The first function that numbarrow compiles therefore caches in the directory set first,
    # and every function does when the cache there is warm.
    assert "`NUMBA_CACHE_DIR` when numba is" in README_TEXT
    assert "misses at least the first function numbarrow compiles" in README_TEXT
    first, late = tmp_path / "first", tmp_path / "late"
    src = (f"import os; os.environ['NUMBA_CACHE_DIR'] = {str(first)!r}\n"
           "import numba\n"
           f"os.environ['NUMBA_CACHE_DIR'] = {str(late)!r}\n"
           "import numbarrow.core.is_null\n")
    env = {key: value for key, value in os.environ.items() if key not in ("NUMBA_CACHE_DIR", "NUMBARROW_JIT_OPTIONS")}
    env["PYTHONPATH"] = str(README.parent)
    run = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert run.returncode == 0, run.stderr
    indexed = {where: sorted(path.name.split(".")[1].split("-")[0] for path in (tmp_path / where).rglob("*.nbi"))
               for where in ("first", "late")}
    assert "is_null" in indexed["first"] and "is_null" not in indexed["late"], indexed


def _output_column(value):
    """Return the column that a UDF returning ``value`` produces from a two-row batch, with its type inferred."""
    batch = pa.RecordBatch.from_arrays([pa.array([0, 0], type=pa.int64())], names=["c"])
    fn = make_mapinarrow_func(lambda data, bitmap, broadcasts: {"o": value}, input_columns=["c"])
    return list(fn(iter([batch])))[0].column("o")


def test_the_shapes_the_docstring_says_carry_a_null_out_do():
    """Check that each shape the factory docstring lists carries a null into the output column: a list, a tuple and
    an object array holding ``None``, a pyarrow Array, a numpy masked array and a ``Nullable``. A plain integer ndarray
    carries none."""
    assert ("A null comes out of a list, a tuple or an object array holding ``None``, a :class:`pyarrow.Array`, "
            "a numpy masked array, and a :class:`Nullable`") in FACTORY_DOC
    shapes = {
        "list": [None, 1],
        "tuple": (None, 1),
        "object array": np.array([None, 1], dtype=object),
        "pyarrow Array": pa.array([None, 1]),
        "masked array": np.ma.array([0, 1], mask=[True, False]),
        "Nullable": Nullable(np.array([0, 1]), np.array([0b10], dtype=np.uint8)),
    }
    for label, value in shapes.items():
        assert _output_column(value).null_count == 1, label
    assert _output_column(np.array([0, 1])).null_count == 0
