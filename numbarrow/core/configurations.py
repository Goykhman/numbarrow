"""
Default configuration options for Numba JIT compilation used throughout numbarrow.
"""

import os
import json


def get_jit_options():
    """
    E.g., export NUMBARROW_JIT_OPTIONS='{"cache": false}'
    """
    as_str = os.environ.get("NUMBARROW_JIT_OPTIONS")
    if as_str is None:
        return {"cache": True}
    try:
        as_json = json.loads(as_str)
        return as_json
    except json.JSONDecodeError:
        raise ValueError("NUMBARROW_JIT_OPTIONS must be valid JSON")


jit_options = get_jit_options()
