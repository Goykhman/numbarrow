import json
import os
import re
import subprocess
import sys

import pytest

from numbarrow.core.configurations import get_jit_options, invalid_jit_options_err


def test_unset_gives_the_cached_default(monkeypatch):
    monkeypatch.delenv("NUMBARROW_JIT_OPTIONS", raising=False)
    assert get_jit_options() == {"cache": True}


def test_a_json_object_is_passed_through_unchanged(monkeypatch):
    monkeypatch.setenv("NUMBARROW_JIT_OPTIONS", '{"cache": false, "nogil": true}')
    assert get_jit_options() == {"cache": False, "nogil": True}


def test_an_empty_value_is_treated_as_unset(monkeypatch):
    monkeypatch.setenv("NUMBARROW_JIT_OPTIONS", "")
    assert get_jit_options() == {"cache": True}


@pytest.mark.parametrize("value", ["{cache: false}", "[1, 2]", "null", '"cache"', "1", "   "])
def test_anything_that_is_not_a_json_object_is_rejected(monkeypatch, value):
    monkeypatch.setenv("NUMBARROW_JIT_OPTIONS", value)
    with pytest.raises(ValueError, match=re.escape(invalid_jit_options_err)):
        get_jit_options()


def test_importing_with_an_empty_value_uses_the_default():
    # jit_options is computed when the module is imported, so the value has to
    # be in the environment before the interpreter starts.
    src = "import json; from numbarrow.core.configurations import jit_options; print(json.dumps(jit_options))"
    env = dict(os.environ, NUMBARROW_JIT_OPTIONS="")
    out = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == {"cache": True}
