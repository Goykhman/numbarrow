"""Tests of ``floor_interpreter`` in the extras gate, ``.github/scripts/extras_sufficiency_check.py``.

The extras gate is a CI script that installs each optional extra of numbarrow alone and checks that the extra delivers
what it promises. It runs on the oldest Python that ``requires-python`` admits, which ``floor_interpreter`` reads from
that specifier set. The workflow runs the script against this tree's own ``requires-python`` only, so the forms of the
set that the function also accepts, and the form it refuses, are tested here. The test is skipped when the script is
absent or cannot be imported. Python 3.10 has no ``tomllib`` for the script to import, so the test is skipped there.
"""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / ".github" / "scripts" / "extras_sufficiency_check.py"


def _gate():
    """Import the extras gate as a module, or skip the test where it is absent or cannot be imported."""
    if not SCRIPT.exists():
        pytest.skip("the extras gate is not in this tree")
    spec = importlib.util.spec_from_file_location("extras_sufficiency_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        # CI runs the gate on the floor interpreter, but it also runs the test suite below the floor, on Python 3.10,
        # which has no tomllib for the script to import.
        pytest.skip(f"the extras gate cannot load here: {exc}")
    return module


def test_the_floor_interpreter_comes_from_a_ge_or_a_compatible_release_clause():
    """Check that ``floor_interpreter`` takes the floor from a ``>=`` clause, with or without a space and beside an
    upper bound, or from a ``~=`` clause, and refuses a specifier set with neither. The workflow runs the script only
    against this tree's ``>=3.12``, so only this test exercises the other forms."""
    gate = _gate()
    assert gate.floor_interpreter(">=3.12") == "python3.12"
    assert gate.floor_interpreter(">= 3.12, <3.14") == "python3.12"
    assert gate.floor_interpreter("~=3.12") == "python3.12"
    with pytest.raises(SystemExit, match="names no >= or ~= floor"):
        gate.floor_interpreter("==3.12.*")
