"""Compiled functions, called viewers, that return a numpy array over the memory at a raw address without a copy.

A viewer takes an address, such as ``pyarrow.Buffer.address``, and a number of elements, and returns a one-dimensional
numpy array of one dtype over the memory there. :func:`numpy_array_from_ptr_factory` compiles the viewer for a dtype,
and ``arrays_viewers`` keeps one viewer per key, such as ``np.uint8``, compiled the first time the key is requested. The
array that a viewer returns does not keep the memory alive, and it is writable. The adapters in
:mod:`numbarrow.utils.arrow_array_utils` and :mod:`numbarrow.core.adapters` therefore use viewers only to read
validity bitmaps, packed booleans and string offsets, and they return copies, never the views themselves.
"""

import hashlib

import numpy as np
from numba import carray, from_dtype, int64, intp
from numba.core.types import Array, voidptr
from numba.extending import intrinsic

from numbarrow.core.configurations import jit_with_options


@intrinsic
def _ptr_as_int_to_voidptr(typingctx, arg_type):
    """Convert an integer memory address to a ``voidptr`` that :func:`numba.carray` can read through.

    The function is a numba intrinsic, so only jitted code can call it. numba types a call as ``voidptr(arg_type)`` and
    compiles it to one LLVM ``inttoptr`` instruction. An intrinsic has no pure-Python form, so under
    ``NUMBA_DISABLE_JIT=1`` every viewer call raises ``NotImplementedError``.
    """
    def codegen(context, builder, signature, args):
        return builder.inttoptr(args[0], context.get_value_type(voidptr))
    return voidptr(arg_type), codegen


def numpy_array_from_ptr_factory(dtype_):
    """Compile and return the viewer for ``dtype_``: a function that views the memory at an address as a numpy array.

    The viewer takes ``(ptr_as_int, sz)``, typed ``intp`` and ``int64``, and returns a one-dimensional C-contiguous
    array of ``sz`` elements of ``dtype_`` that starts at the address ``ptr_as_int``. ``dtype_`` is anything that
    ``numpy.dtype`` accepts and numba supports, a structured dtype included; the object dtype fails to compile. The
    factory compiles the viewer under numbarrow's jit options, as
    :func:`~numbarrow.core.configurations.jit_with_options` describes, and each call builds a new viewer;
    ``arrays_viewers`` builds one per key and keeps it.

    The viewer copies nothing. Its array views the memory at the address without keeping that memory alive, and the
    array is writable. Reading it after the memory is freed is undefined behavior, and a write through it changes the
    memory. :func:`~numbarrow.core.adapters.arrow_array_adapter` is the supported way to view an Arrow array, because
    it returns read-only arrays that keep their Arrow buffers alive.
    """
    def viewer(ptr_as_int: int, sz: int):
        # carray makes an array of the memory at the address, without a copy.
        return carray(_ptr_as_int_to_voidptr(ptr_as_int), shape=(sz,), dtype=dtype_)
    # numba names the cache files of a function after its module file, its qualified name and its first line number.
    # Python gives every viewer that this factory makes the same first line and the same qualified name,
    # numpy_array_from_ptr_factory.<locals>.viewer. Under that name, the viewers would share one index file and one set
    # of data files, which numba writes without a lock. Processes that import numbarrow together on a cold cache could
    # read the same index and pick the same data-file name for different viewers. A later process could then load the
    # wrong machine code. A qualified name per dtype gives each viewer its own index and data files. With one entry per
    # index, two writers have nothing to disagree about.
    dtype_ = np.dtype(dtype_)
    name = f"view_{dtype_.name}"
    if dtype_.fields is not None:
        # numpy gives every structured dtype of one itemsize the same name, void<bits>, so two such dtypes would get
        # viewers of one name. If two processes compiled them into one cache, a later process that loads both can run
        # the code of the first for the second. The repr of a dtype tells them apart, so a hash of it goes into the
        # name. The hash is of the repr, not of dtype.descr, because numpy refuses to build descr for a dtype with
        # out-of-order or overlapping fields. Such a dtype is what multi-field indexing, rec[["b", "a"]], returns.
        name += "_" + hashlib.sha1(repr(dtype_).encode()).hexdigest()[:12]
    viewer.__name__ = name
    viewer.__qualname__ = f"{numpy_array_from_ptr_factory.__qualname__}.<locals>.{name}"
    return jit_with_options(Array(from_dtype(dtype_), 1, "C")(intp, int64))(viewer)


class _Viewers(dict):
    """A dict of viewers by dtype that compiles each viewer the first time its key is requested, then keeps it.

    ``arrays_viewers[np.int32]`` compiles the int32 viewer through :func:`numpy_array_from_ptr_factory` on the first
    request and returns that same function on every later request. A dtype that nothing requests is never compiled.
    The adapters request uint8 for validity bitmaps and packed booleans, and int32 and int64 for string offsets. Each
    dtype has an index file and data files of its own in numba's cache.
    """

    def __missing__(self, dtype_):
        viewer = self[dtype_] = numpy_array_from_ptr_factory(dtype_)
        return viewer


arrays_viewers = _Viewers()
