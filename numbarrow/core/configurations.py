"""
Default configuration options for Numba JIT compilation used throughout numbarrow.
"""

import os
import json


invalid_jit_options_err = """Must be valid JSON, e.g., export NUMBARROW_JIT_OPTIONS='{"cache": false}'"""


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
    except json.JSONDecodeError:
        raise ValueError(invalid_jit_options_err)
    if not isinstance(as_json, dict):
        raise ValueError(invalid_jit_options_err)
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
