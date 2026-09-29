import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from numba import void

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


@pytest.mark.parametrize("value", ['{"cache": "false"}', '{"cache": 0}', '{"cache": null}'])
def test_a_cache_value_that_is_not_a_boolean_is_rejected(monkeypatch, value):
    # numba reads any non-empty string as true, so '{"cache": "false"}' left
    # caching on.
    monkeypatch.setenv("NUMBARROW_JIT_OPTIONS", value)
    with pytest.raises(ValueError, match="cache"):
        get_jit_options()


@pytest.mark.parametrize("value", ['{"boundscheck": "false"}', '{"nogil": "true"}', '{"parallel": "false"}'])
def test_a_string_for_any_other_option_is_rejected(monkeypatch, value):
    # The same reading: '{"boundscheck": "false"}' turned bounds checking on.
    monkeypatch.setenv("NUMBARROW_JIT_OPTIONS", value)
    with pytest.raises(ValueError, match=re.escape(json.loads(value).popitem()[0])):
        get_jit_options()


def test_the_two_string_options_are_passed_through(monkeypatch):
    monkeypatch.setenv("NUMBARROW_JIT_OPTIONS", '{"error_model": "numpy", "inline": "never", "cache": false}')
    assert get_jit_options() == {"error_model": "numpy", "inline": "never", "cache": False}


def test_an_explicit_object_replaces_the_default(monkeypatch):
    # Documented rather than merged: '{}' turns caching off.
    monkeypatch.setenv("NUMBARROW_JIT_OPTIONS", "{}")
    assert get_jit_options() == {}


def test_importing_with_an_empty_value_uses_the_default():
    # jit_options is computed when the module is imported, so the value has to
    # be in the environment before the interpreter starts.
    src = "import json; from numbarrow.core.configurations import jit_options; print(json.dumps(jit_options))"
    # PYTHONPATH names the tree under test: from any cwd but the checkout root
    # the child would otherwise import whichever numbarrow it finds installed.
    env = dict(os.environ, NUMBARROW_JIT_OPTIONS="", PYTHONPATH=str(Path(__file__).resolve().parent.parent))
    out = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == {"cache": True}


def test_the_refusal_names_the_requirement_and_shows_the_value(monkeypatch):
    # One message for both failures told a value that was valid JSON that it
    # must be valid JSON, and showed neither the value nor the rule.
    monkeypatch.setenv("NUMBARROW_JIT_OPTIONS", "[1, 2]")
    with pytest.raises(ValueError, match=r"JSON object.*'\[1, 2\]' is valid JSON but a list"):
        get_jit_options()
    monkeypatch.setenv("NUMBARROW_JIT_OPTIONS", "{cache: false}")
    with pytest.raises(ValueError, match=r"'\{cache: false\}' is not valid JSON"):
        get_jit_options()


def test_a_runtime_error_other_than_numbas_no_locator_one_propagates(monkeypatch):
    # The fallback is narrowed to numba's "no locator available", and nothing
    # tested the narrowing: with the term dropped, every RuntimeError at
    # decoration compiled uncached behind a warning about the cache.
    from numbarrow.core import configurations
    options_seen = []

    def njit_raising_once(message):
        def njit(*signature, **options):
            def decorate(func):
                options_seen.append(options)
                if len(options_seen) == 1:
                    raise RuntimeError(message)
                return func
            return decorate
        return njit

    monkeypatch.setattr(configurations, "jit_options", {"cache": True})
    monkeypatch.setattr(configurations, "njit", njit_raising_once("some other failure at decoration"))
    with pytest.raises(RuntimeError, match="some other failure at decoration"):
        configurations.jit_with_options(void())(lambda: None)
    assert options_seen == [{"cache": True}]
    options_seen.clear()
    monkeypatch.setattr(configurations, "njit", njit_raising_once("cannot cache function: no locator available"))
    with pytest.warns(RuntimeWarning, match="compiles without a cache"):
        configurations.jit_with_options(void())(lambda: None)
    assert options_seen == [{"cache": True}, {"cache": False}]


def test_a_cache_write_that_fails_at_decoration_compiles_uncached_only_when_caching_is_on(monkeypatch):
    # numba takes a .zip's cache location unchecked, so there the failure is
    # the first save's OSError and not the no-locator RuntimeError, and the
    # import died on it. With caching off no cache is involved, and the error
    # is the caller's to see.
    from numbarrow.core import configurations
    options_seen = []

    def njit(*signature, **options):
        def decorate(func):
            options_seen.append(options)
            if options.get("cache") or len(options_seen) == 1:
                raise PermissionError(13, "Permission denied", "/nowhere/numba")
            return func
        return decorate

    monkeypatch.setattr(configurations, "njit", njit)
    monkeypatch.setattr(configurations, "jit_options", {"cache": True})
    with pytest.warns(RuntimeWarning, match=r"Permission denied: '/nowhere/numba'.*compiles without a cache"):
        configurations.jit_with_options(void())(lambda: None)
    assert options_seen == [{"cache": True}, {"cache": False}]
    options_seen.clear()
    monkeypatch.setattr(configurations, "jit_options", {"cache": False})
    with pytest.raises(PermissionError):
        configurations.jit_with_options(void())(lambda: None)
    assert options_seen == [{"cache": False}]
