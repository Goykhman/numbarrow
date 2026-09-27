"""The extras gate's own parsing, which the workflow runs only against this tree's requires-python."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / ".github" / "scripts" / "extras_sufficiency_check.py"


def _gate():
    if not SCRIPT.exists():
        pytest.skip("the extras gate is not in this tree")
    spec = importlib.util.spec_from_file_location("extras_sufficiency_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_floor_interpreter_comes_from_a_ge_or_a_compatible_release_clause():
    # The gate reads its interpreter from the floor of requires-python, and
    # the workflow runs it against this tree's own ">=3.12" alone, so the ~=
    # clause it also accepts and the refusal of a set naming neither had no
    # run behind them.
    gate = _gate()
    assert gate.floor_interpreter(">=3.12") == "python3.12"
    assert gate.floor_interpreter(">= 3.12, <3.14") == "python3.12"
    assert gate.floor_interpreter("~=3.12") == "python3.12"
    with pytest.raises(SystemExit, match="names no >= or ~= floor"):
        gate.floor_interpreter("==3.12.*")
