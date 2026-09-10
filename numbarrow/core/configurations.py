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
    raises ``ValueError``. Each call reads the environment afresh; the module-level ``jit_options`` holds the
    value read when this module was first imported, and that is what the decorators use.
    """
    as_str = os.environ.get("NUMBARROW_JIT_OPTIONS")
    if not as_str:
        return {"cache": True}
    try:
        as_json = json.loads(as_str)
        if not isinstance(as_json, dict):
            raise ValueError(invalid_jit_options_err)
        return as_json
    except json.JSONDecodeError:
        raise ValueError(invalid_jit_options_err)


jit_options = get_jit_options()
