"""
Default configuration options for Numba JIT compilation used throughout numbarrow.
"""

# Passed as **kwargs to @njit decorators. "cache=True" persists compiled
# functions to disk so subsequent imports skip recompilation.
default_jit_options = {
    "cache": True
}
