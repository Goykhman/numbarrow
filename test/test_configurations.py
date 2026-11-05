"""Tests of how ``numbarrow.core.configurations`` reads ``NUMBARROW_JIT_OPTIONS``, and of the fallback in
``jit_with_options`` that compiles without a cache where numba cannot cache.

The tests of ``get_jit_options`` set the variable with ``monkeypatch`` and call the function in this process, because
it reads the environment on every call. One test imports the module in a child process, because the module-level
``jit_options`` is read once, at the first import. The fallback tests replace numba's ``njit`` with a stub that raises,
so they need no archive and no read-only directory. ``test_cache.py`` runs the fallback against real archives and
read-only directories, and checks that the options reach the compiled functions.
"""
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
    # numba reads any non-empty string as true, so '{"cache": "false"}' would leave caching on.
    monkeypatch.setenv("NUMBARROW_JIT_OPTIONS", value)
    with pytest.raises(ValueError, match="cache"):
        get_jit_options()


@pytest.mark.parametrize("value", ['{"boundscheck": "false"}', '{"nogil": "true"}', '{"parallel": "false"}'])
def test_a_string_for_any_other_option_is_rejected(monkeypatch, value):
    # numba reads these strings the same way, so '{"boundscheck": "false"}' would turn bounds checking on.
    monkeypatch.setenv("NUMBARROW_JIT_OPTIONS", value)
    with pytest.raises(ValueError, match=re.escape(json.loads(value).popitem()[0])):
        get_jit_options()


def test_the_two_string_options_are_passed_through(monkeypatch):
    """Check that ``error_model`` and ``inline``, the two options that numba takes as strings, keep their values."""
    monkeypatch.setenv("NUMBARROW_JIT_OPTIONS", '{"error_model": "numpy", "inline": "never", "cache": false}')
    assert get_jit_options() == {"error_model": "numpy", "inline": "never", "cache": False}


def test_an_explicit_object_replaces_the_default(monkeypatch):
    # The object replaces the default instead of extending it, so '{}' turns caching off.
    monkeypatch.setenv("NUMBARROW_JIT_OPTIONS", "{}")
    assert get_jit_options() == {}


def test_importing_with_an_empty_value_uses_the_default():
    # The module reads jit_options once, at its first import, and this process has already imported it, so a child
    # process imports the module with the empty value set.
    src = "import json; from numbarrow.core.configurations import jit_options; print(json.dumps(jit_options))"
    # PYTHONPATH points the child at the tree under test. A child started outside the checkout root would otherwise
    # import whatever numbarrow its interpreter has installed.
    env = dict(os.environ, NUMBARROW_JIT_OPTIONS="", PYTHONPATH=str(Path(__file__).resolve().parent.parent))
    out = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == {"cache": True}


def test_the_refusal_names_the_requirement_and_shows_the_value(monkeypatch):
    """Check that each refusal shows the value and says which rule it breaks: the value is not valid JSON, or it is
    valid JSON but not an object."""
    monkeypatch.setenv("NUMBARROW_JIT_OPTIONS", "[1, 2]")
    with pytest.raises(ValueError, match=r"JSON object.*'\[1, 2\]' is valid JSON but a list"):
        get_jit_options()
    monkeypatch.setenv("NUMBARROW_JIT_OPTIONS", "{cache: false}")
    with pytest.raises(ValueError, match=r"'\{cache: false\}' is not valid JSON"):
        get_jit_options()


def test_a_runtime_error_other_than_numbas_no_locator_one_propagates(monkeypatch):
    """Check that ``jit_with_options`` compiles without a cache after numba's "no locator available"
    ``RuntimeError``, and lets any other ``RuntimeError`` at decoration propagate. Without that distinction, an error
    unrelated to the cache would compile uncached behind a warning about the cache."""
    from numbarrow.core import configurations
    options_seen = []

    def njit_raising_once(message):
        """Return a stand-in for numba's ``njit`` that records the options of each call in ``options_seen``, raises
        ``RuntimeError(message)`` at its first decoration, and returns the function unchanged after that."""
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
    """Check that an ``OSError`` at decoration compiles without a cache, with a warning that shows the error, when
    caching is on, and propagates when caching is off.

    From version 0.61, numba takes the user's cache directory for a ``.zip`` without checking that it can write there.
    The failure then comes from numba's first save as an ``OSError``, such as ``PermissionError``, not as the
    no-locator ``RuntimeError``, and without the fallback it fails the import. With caching off, the error has nothing
    to do with a cache, and the caller sees it.
    """
    from numbarrow.core import configurations
    options_seen = []

    # The stub raises at its first decoration whatever the options, and at every decoration with caching on.
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
