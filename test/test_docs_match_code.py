"""Invariants tying the documented behaviour to the code.

These exist because the docs drifted from the code repeatedly: a type table
that did not match the dispatcher, a copy-versus-view claim that was wrong for
one type, and a conversion the docs promised would raise for every output
shape while one of the shapes truncated it. A prose fix closes one instance;
these fail the build the next time any of it drifts, which is the only thing
that stops it recurring.
"""
import datetime
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from numbarrow.core.adapters import arrow_array_adapter
from numbarrow.core.mapinarrow_factory import Nullable, make_mapinarrow_func

README = Path(__file__).resolve().parent.parent / "README.md"
PYPROJECT = README.parent / "pyproject.toml"

# The docstring as one line, so a claim that wraps is still one string.
FACTORY_DOC = " ".join(make_mapinarrow_func.__doc__.split())

# The README the same way, for the sentences it repeats.
README_TEXT = " ".join(README.read_text().split())

SUB_SECOND = datetime.datetime(2020, 1, 1, 12, 34, 56, 789012)

# One representative array per documented type. Kept beside the table it
# checks, so adding a row to the table without adding a sample fails here.
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
    """The Supported Types table as [(type names, copy answer)]."""
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
    """True when the result views an Arrow buffer rather than owning fresh memory."""
    node = array
    for _ in range(8):
        if isinstance(node, memoryview):
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
            # A struct or list row adapts per field; its children are covered
            # by the scalar rows, so the only claim to hold it to is the label.
            assert copy_answer.lower().startswith("per-field"), (
                f"{name}: adapts per field, README says {copy_answer!r}"
            )
            continue
        # Asserted rather than skipped: a reworded cell used to disarm its row.
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
    """The column a UDF returning *value* under a declared *arrow_type* yields."""
    batch = pa.RecordBatch.from_pydict({"v": [1.0]})
    fn = make_mapinarrow_func(lambda d, b, br: {"out": value},
                              output_schema=pa.schema([("out", arrow_type)]))
    return list(fn(iter([batch])))[0].column("out")


def test_the_refusals_hold_for_the_shapes_the_docstring_names_them_for():
    # The docstring anchored its promise to pa.array without saying that
    # pa.array has two converters: the typed one raises on these two, and the
    # sequence converter a list goes through truncates both without a word.
    assert "For an ndarray of a numeric or datetime dtype, or a :class:`pyarrow.Array`" in FACTORY_DOC
    with pytest.raises(pa.ArrowInvalid):
        _one_output_column(np.array([1.5]), pa.int64())
    with pytest.raises(pa.ArrowInvalid):
        _one_output_column(np.array([SUB_SECOND], dtype="datetime64[us]"), pa.timestamp("s"))


def test_the_truncations_the_docstring_admits_for_a_list_are_the_ones_it_makes():
    assert "goes through ``pa.array``'s sequence converter" in FACTORY_DOC
    assert ("from a list alone, a fraction into an integer type and a timestamp unit change "
            "that drops digits") in FACTORY_DOC
    assert _one_output_column([1.5], pa.int64()).to_pylist() == [1]
    assert _one_output_column([SUB_SECOND], pa.timestamp("s")).to_pylist() == [
        SUB_SECOND.replace(microsecond=0)
    ]


def _inferred_output_column(value):
    """The column a UDF returning *value* with no output_schema yields."""
    batch = pa.RecordBatch.from_pydict({"v": [1.0]})
    fn = make_mapinarrow_func(lambda d, b, br: {"out": value})
    return list(fn(iter([batch])))[0].column("out")


def test_an_inferred_datetime64_column_comes_back_as_the_docs_say():
    # Both sentences promised a timestamp of the array's unit for every unit,
    # and pa.array infers date32 for the day one, which the round-trip test's
    # own drift table admits by leaving date32 out of it.
    assert ("a ``datetime64`` array comes back a naive ``timestamp`` of its unit, except a "
            "day-unit one, which comes back ``date32``") in FACTORY_DOC
    assert ("a `datetime64` output becomes a naive timestamp of its unit, except "
            "`datetime64[D]`, which becomes `date32`") in README_TEXT
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
    # The README kept explaining a 2.2.2 row against a 1.5.0 extra after both
    # extras had moved to 2.2.2, so the floors are read from the file.
    floors = set(re.findall(r'"pandas>=([\d.]+)"', PYPROJECT.read_text()))
    row = re.search(r"\| pandas \| ([\d.]+)\+", README.read_text()).group(1)
    assert floors == {row}, (floors, row)
    assert "1.5.0" not in README_TEXT


def _output_column(value):
    batch = pa.RecordBatch.from_arrays([pa.array([0, 0], type=pa.int64())], names=["c"])
    fn = make_mapinarrow_func(lambda data, bitmap, broadcasts: {"o": value}, input_columns=["c"])
    return list(fn(iter([batch])))[0].column("o")


def test_the_shapes_the_docstring_says_carry_a_null_out_do():
    # The sentence named four shapes while a tuple and an object array
    # holding None carried a null out too.
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
