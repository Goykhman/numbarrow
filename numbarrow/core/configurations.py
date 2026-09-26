"""
Default configuration options for Numba JIT compilation used throughout numbarrow.
"""

import os
import json
import warnings

from numba import njit


invalid_jit_options_err = (
    """NUMBARROW_JIT_OPTIONS must be a JSON object, e.g., export NUMBARROW_JIT_OPTIONS='{"cache": false}'"""
)


def get_jit_options():
    """
    Numba JIT options taken from the ``NUMBARROW_JIT_OPTIONS`` environment variable.

    The value must be a JSON object and is passed to ``@njit`` as keyword arguments unchanged, for example
    ``export NUMBARROW_JIT_OPTIONS='{"cache": false}'``. Unset or empty means ``{"cache": True}``; anything else
    raises ``ValueError``. The object replaces the default rather than extending it, so ``'{}'`` turns caching
    off. ``"cache"`` must be a JSON boolean: numba reads the string ``"false"`` as true, so it is refused rather
    than passed on, and so is a string for any other option but ``"error_model"`` and ``"inline"``, the two
    numba reads as strings: ``{"boundscheck": "false"}`` would turn bounds checking on. Each call reads the
    environment afresh; the module-level ``jit_options`` holds the value read when this module was first
    imported, and that is what the decorators use.

    numba's on-disk cache index records a function's signature, target and bytecode but not the options it
    was compiled with, so a cache warmed under one set of options is a hit for a run that asked for another:
    ``{"cache": true, "boundscheck": true}`` over a directory warmed with the default loads the unchecked
    code. Point ``NUMBA_CACHE_DIR`` at a fresh directory when an option changes.
    """
    as_str = os.environ.get("NUMBARROW_JIT_OPTIONS")
    if not as_str:
        return {"cache": True}
    try:
        as_json = json.loads(as_str)
    except json.JSONDecodeError as error:
        raise ValueError(f"{invalid_jit_options_err}; {as_str!r} is not valid JSON: {error}") from None
    if not isinstance(as_json, dict):
        # One message for both failures told a value that was valid JSON that
        # it must be valid JSON, and showed neither the value nor the rule.
        raise ValueError(
            f"{invalid_jit_options_err}; {as_str!r} is valid JSON but a {type(as_json).__name__}, not an object"
        )
    if "cache" in as_json and not isinstance(as_json["cache"], bool):
        raise ValueError(
            f'NUMBARROW_JIT_OPTIONS "cache" must be true or false, not {as_json["cache"]!r}: numba reads any '
            f"non-empty string as true"
        )
    for name, value in as_json.items():
        if isinstance(value, str) and name not in ("error_model", "inline"):
            raise ValueError(
                f'NUMBARROW_JIT_OPTIONS "{name}" cannot be the string {value!r}: numba reads any non-empty string '
                f'as true, and only "error_model" and "inline" take a string'
            )
    return as_json


jit_options = get_jit_options()


def jit_with_options(*signature):
    """``njit`` under the options ``NUMBARROW_JIT_OPTIONS`` gives, compiling uncached where no cache can be written.

    numba sets a cached function up when it is decorated, and raises ``RuntimeError`` there when no cache
    location can be written: a read-only install, an unwritable ``site-packages`` and user cache directory, or
    an import from an ``.egg``, ``.whl`` or ``.pyz`` archive, which Spark's ``--py-files`` ships. Nothing then
    named the way out. Such a function compiles without a cache, with a warning naming ``NUMBA_CACHE_DIR`` and
    ``NUMBARROW_JIT_OPTIONS='{"cache": false}'``. A write that fails later, on a full disk, is numba's own error.
    """
    def decorate(func):
        try:
            return njit(*signature, **jit_options)(func)
        except RuntimeError as error:
            if "no locator available" not in str(error) or not jit_options.get("cache"):
                raise
            warnings.warn(
                f"numba cannot cache {func.__qualname__} here ({error}); it compiles without a cache. Set "
                f"NUMBA_CACHE_DIR to a writable directory, or NUMBARROW_JIT_OPTIONS='{{\"cache\": false}}' to "
                f"turn caching off and silence this warning",
                RuntimeWarning, stacklevel=2,
            )
            return njit(*signature, **{**jit_options, "cache": False})(func)
    return decorate
