"""
Low-level pointer utilities for zero-copy access to Arrow memory buffers.

Provides Numba-compatible functions that reinterpret a raw memory address
(obtained from :attr:`pyarrow.Buffer.address`) as a typed NumPy array, enabling
``@njit`` code to read Arrow buffer data directly without copying.
"""

import numpy as np
from numba import carray, from_dtype, int64, intp, njit
from numba.core.types import Array, voidptr
from numba.extending import intrinsic

from numbarrow.core.configurations import default_jit_options


@intrinsic
def _ptr_as_int_to_voidptr(typingctx, arg_type):
    """Convert an integer memory address to a Numba ``voidptr``.

    This is a Numba intrinsic (compiler-level function) that emits an
    LLVM ``inttoptr`` instruction, converting a Python int holding a
    memory address into a void pointer that :func:`numba.carray` can
    dereference.
    """
    def codegen(context, builder, signature, args):
        return builder.inttoptr(args[0], context.get_value_type(voidptr))
    return voidptr(arg_type), codegen


def numpy_array_from_ptr_factory(dtype_):
    """Create a JIT-compiled function that views memory at a given address as a NumPy array.

    Returns an ``@njit`` function with signature ``(ptr_as_int, sz) -> ndarray``
    that uses :func:`numba.carray` to reinterpret *sz* elements starting at
    address *ptr_as_int* as a contiguous C-order NumPy array of *dtype_*.
    No data is copied — the returned array is a view over the original memory.

    :param dtype_: NumPy dtype for the resulting array (e.g. ``np.int32``)
    :returns: JIT-compiled function ``(int, int) -> np.ndarray``
    """
    @njit(Array(from_dtype(dtype_), 1, "C")(intp, int64), **default_jit_options)
    def _(ptr_as_int: int, sz: int):
        # carray interprets raw memory at ptr as a typed NumPy array (zero-copy view)
        return carray(_ptr_as_int_to_voidptr(ptr_as_int), shape=(sz,), dtype=dtype_)
    return _


# Pre-built viewers for common NumPy types. Each entry maps a dtype to a
# JIT-compiled function that views a memory address as an array of that type.
arrays_viewers = {
    np_type: numpy_array_from_ptr_factory(np_type) for np_type in [
        np.bool_,
        np.float64,
        np.int32,
        np.int64,
        np.uint8
    ]
}
