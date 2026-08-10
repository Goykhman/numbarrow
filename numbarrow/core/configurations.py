"""
Default configuration options for Numba JIT compilation used throughout numbarrow.
"""

import os
import json


invalid_jit_options_err = """Must be valid JSON, e.g., export NUMBARROW_JIT_OPTIONS='{"cache": false}'"""


def get_jit_options():
    as_str = os.environ.get("NUMBARROW_JIT_OPTIONS")
    if as_str is None:
        return {"cache": True}
    try:
        as_json = json.loads(as_str)
        if not isinstance(as_json, dict):
            raise ValueError(invalid_jit_options_err)
        return as_json
    except json.JSONDecodeError:
        raise ValueError(invalid_jit_options_err)


jit_options = get_jit_options()
