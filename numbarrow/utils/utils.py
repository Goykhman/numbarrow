"""
Low-level pointer utilities for zero-copy access to Arrow memory buffers.

Provides Numba-compatible functions that reinterpret a raw memory address
(obtained from :attr:`pyarrow.Buffer.address`) as a typed NumPy array, enabling
``@njit`` code to read Arrow buffer data directly without copying.
"""

import hashlib

import numpy as np
from numba import carray, from_dtype, int64, intp
from numba.core.types import Array, voidptr
from numba.extending import intrinsic

from numbarrow.core.configurations import jit_with_options


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
    No data is copied: the returned array is a bare view over the memory at
    that address, with no owner and no read-only flag, valid only while that
    memory is. Reading it after the source is gone is undefined, and a write
    through it changes the source. The supported way in is
    :func:`~numbarrow.core.adapters.arrow_array_adapter`, which returns
    read-only arrays tied to the Arrow array they view; this is the primitive
    it is built on.

    :param dtype_: NumPy dtype for the resulting array (e.g. ``np.int32``)
    :returns: JIT-compiled function ``(int, int) -> np.ndarray``
    """
    def viewer(ptr_as_int: int, sz: int):
        # carray interprets raw memory at ptr as a typed NumPy array (zero-copy view)
        return carray(_ptr_as_int_to_voidptr(ptr_as_int), shape=(sz,), dtype=dtype_)
    # numba names a function's cache files after its qualname and source line,
    # so every viewer this factory makes shared one index file and one set of
    # data files, and numba writes those without a lock. Processes importing
    # together on a cold cache read one index and picked the same data-file
    # name for different viewers, and every process afterwards loaded the wrong
    # machine code: an int32 column read as float64, or a crash in
    # NRT_adapt_ndarray_to_python. A qualname per dtype gives each viewer its
    # own index and data files, and one entry per index leaves nothing for two
    # writers to disagree about.
    dtype_ = np.dtype(dtype_)
    name = f"view_{dtype_.name}"
    if dtype_.fields is not None:
        # numpy names every structured dtype of one itemsize void<bits>, so
        # two of them shared one index, and a process loading both from the
        # cache ran the first one's code for the second; the description
        # tells them apart.
        name += "_" + hashlib.sha1(repr(dtype_.descr).encode()).hexdigest()[:12]
    viewer.__name__ = name
    viewer.__qualname__ = f"{numpy_array_from_ptr_factory.__qualname__}.<locals>.{name}"
    return jit_with_options(Array(from_dtype(dtype_), 1, "C")(intp, int64))(viewer)


class _Viewers(dict):
    """The viewers by dtype, each built the first time it is asked for and kept for the next time.

    ``arrays_viewers[np.int32]`` compiles the int32 viewer through
    :func:`numpy_array_from_ptr_factory` on the first request and returns the
    same function on every request after. A viewer carries an index and a
    data file of its own in the numba cache, so nothing is compiled that
    nothing asks for; the adapters ask for uint8 for validity bitmaps and
    packed booleans, and int32 and int64 for string offsets.
    """

    def __missing__(self, dtype_):
        viewer = self[dtype_] = numpy_array_from_ptr_factory(dtype_)
        return viewer


arrays_viewers = _Viewers()
