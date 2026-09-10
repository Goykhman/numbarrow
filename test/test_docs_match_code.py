"""Invariants tying the documented behaviour to the code.

These exist because the docs drifted from the code repeatedly: a type table
that did not match the dispatcher, and a copy-versus-view claim that was wrong
for one type. A prose fix closes one instance; these fail the build the next
time any type drifts, which is the only thing that stops it recurring.
"""
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from numbarrow.core.adapters import arrow_array_adapter

README = Path(__file__).resolve().parent.parent / "README.md"

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
    # "Per-field" rows return dicts; their children are covered by the scalar
    # rows above, so only the yes/no rows are checkable here.
    if not copy_answer.lower().startswith(("yes", "no")):
        pytest.skip(f"not a yes/no answer: {copy_answer!r}")
    documented_copy = copy_answer.lower().startswith("yes")
    for name in names:
        result = arrow_array_adapter(SAMPLES[name])
        data = result[1]
        assert isinstance(data, np.ndarray), name
        measured_copy = not _is_view(data)
        assert measured_copy == documented_copy, (
            f"{name}: README says {'copy' if documented_copy else 'view'}, "
            f"measured {'copy' if measured_copy else 'view'}"
        )
