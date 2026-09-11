"""
Type-dispatched adapters that convert PyArrow arrays into NumPy arrays for use
in Numba ``@njit`` compiled functions.

Uses :func:`functools.singledispatch` to route each PyArrow array type
(BooleanArray, Int32Array, Date32Array, etc.) to a handler that extracts the
underlying data buffer as a NumPy array and the validity bitmap as a uint8 array.
Where possible, data is viewed without copying; types that require layout changes
(e.g. Date32 → datetime64[D]) produce a copy.
"""

import numpy as np
import pyarrow as pa

from functools import singledispatch

from numbarrow.core.is_null import unpack_booleans
from numbarrow.utils.arrow_array_utils import (
    create_bitmap, create_str_array, structured_array_adapter,
    structured_list_array_adapter, type_repr, uniform_arrow_array_adapter
)
from numbarrow.utils.utils import arrays_viewers


def cast_64bit_date_arrow_to_numpy_array(pa_array: pa.Array, np_dtype: np.dtype):
    """ Can be used to cast PyArrow arrays of date types that are represented by
    64-bit integers to numpy arrays of various date types (np.datetime64[...],
    which are always represented  by 64-bit integers whose meaning is determined
    by the precision, such as, 's', 'ms', 'us').

    Since underlying data layout of both arrays in int64, a copy is avoided,

    The associated bitmap (if any) is also returned.
    """
    int64_array = pa_array.cast(pa.int64())
    # A zero-length array's buffers are not required to survive a cast, and
    # nothing has been copied when there is nothing to copy.
    if len(pa_array):
        assert int64_array.buffers()[1].address == pa_array.buffers()[1].address, "got copied"
    bitmap, int64_data = uniform_arrow_array_adapter(int64_array)
    data = int64_data.view(np_dtype)
    assert data.ctypes.data == int64_data.ctypes.data, "got copied"
    return bitmap, data


@singledispatch
def arrow_array_adapter(pa_array: pa.Array):
    """ Dispatcher for PyArrow array adapters of various types. """
    # The array itself is deliberately not interpolated: its repr carries every
    # value it holds, which grows without bound and puts the caller's data into
    # whatever log catches the traceback.
    # Being the base implementation, this also receives anything that is not a
    # registered Array, so neither attribute the message wants is guaranteed to
    # be there: a pa.Scalar or a pa.Field has a type but no length, and an
    # object that is not a pyarrow type at all, a list or an ndarray, has a
    # length but no type. Both are read defensively, because reading either one
    # directly turns the documented NotImplementedError into an AttributeError
    # or a TypeError raised while building the message, which escapes any
    # caller that wraps this in `except NotImplementedError`.
    if isinstance(pa_array, pa.ChunkedArray):
        # A Table column. Described as one, rather than as "an array of N
        # elements of type int64", which blames a supported type and never
        # says chunked.
        raise NotImplementedError(
            f"Not implemented for a ChunkedArray of {pa_array.num_chunks} chunks of type "
            f"{type_repr(pa_array.type)}: pass one chunk, or combine_chunks() first"
        )
    arrow_type = getattr(pa_array, "type", None)
    if arrow_type is None:
        described = f"{type(pa_array).__name__}, which is not a pyarrow Array"
    elif hasattr(pa_array, "__len__"):
        described = f"an array of {len(pa_array)} elements of type {type_repr(arrow_type)}"
    else:
        described = f"{type(pa_array).__name__} of type {type_repr(arrow_type)}"
    raise NotImplementedError(f"Not implemented for {described}")


@arrow_array_adapter.register(pa.BooleanArray)
def _(pa_array: pa.BooleanArray):
    """ PyArrow stores boolean arrays bit-wise,
     following the same kind of layout it uses
     for bitmaps. This requires creating a copy
     when casting to numpy arrays of booleans.
    """
    bitmap_buf, data_buf = pa_array.buffers()
    num_of_bool_elements = len(pa_array)
    if num_of_bool_elements == 0:
        # The Arrow spec lets a zero-length array carry a null data buffer, and
        # pyarrow's own full validation accepts one, so there is nothing to
        # unpack. Answered the same way as the 0-byte-buffer array it is
        # logically identical to, rather than refused.
        data = np.empty((0,), dtype=np.bool_)
        data.flags.writeable = False
        return create_bitmap(bitmap_buf, pa_array.offset, 0), data
    if data_buf is None:
        # Unreachable through pyarrow, which refuses to build a non-empty array
        # with a null data buffer, but a foreign producer can hand one over the
        # C Data Interface. A typed error beats dereferencing None.
        raise ValueError(f"bool array of length {num_of_bool_elements} has no data buffer")
    data_buf_p = data_buf.address
    total_bits = pa_array.offset + num_of_bool_elements
    num_of_bytes = (total_bits + 7) // 8
    packed_boolean_data_viewer = arrays_viewers[np.uint8]
    packed_boolean_data = packed_boolean_data_viewer(data_buf_p, num_of_bytes)
    data = unpack_booleans(pa_array.offset, num_of_bool_elements, packed_boolean_data)
    bitmap = create_bitmap(bitmap_buf, pa_array.offset, len(pa_array))
    # A fresh allocation, but read-only all the same: the contract must not
    # depend on which Arrow type the caller happened to pass.
    data.flags.writeable = False
    return bitmap, data


@arrow_array_adapter.register(pa.Date32Array)
def _(pa_array: pa.Date32Array):
    """
    PyArrow Date32 dates are represented by 32bit integers.
    Since all numpy dates are represented by 64bit integers, this
    creates a copy when it re-interprets numpy array of int32 integers
     (number of days since 1970-01-01) as datetime64[D] (int64)"""
    int32_array = pa_array.cast(pa.int32())
    # A zero-length array's buffers are not required to survive a cast, and
    # numpy may hand back the same empty allocation twice.
    if len(pa_array):
        assert int32_array.buffers()[1].address == pa_array.buffers()[1].address, "got copied"
    bitmap, int32_data = uniform_arrow_array_adapter(int32_array)
    data = int32_data.astype(np.dtype("datetime64[D]"))
    if len(pa_array):
        assert int32_data.ctypes.data != data.ctypes.data
    data.flags.writeable = False
    return bitmap, data


@arrow_array_adapter.register(pa.Date64Array)
def _(pa_array: pa.Date64Array):
    return cast_64bit_date_arrow_to_numpy_array(pa_array, np.dtype("datetime64[ms]"))


@arrow_array_adapter.register(pa.lib.DoubleArray)
@arrow_array_adapter.register(pa.Int32Array)
@arrow_array_adapter.register(pa.Int64Array)
@arrow_array_adapter.register(pa.UInt8Array)
def _(pa_array: pa.lib.DoubleArray | pa.Int32Array | pa.Int64Array | pa.UInt8Array):
    return uniform_arrow_array_adapter(pa_array)


@arrow_array_adapter.register(pa.ListArray)
def _(pa_array: pa.ListArray):
    return structured_list_array_adapter(pa_array)


@arrow_array_adapter.register(pa.StructArray)
def _(pa_array: pa.StructArray):
    return structured_array_adapter(pa_array)


@arrow_array_adapter.register(pa.LargeStringArray)
@arrow_array_adapter.register(pa.StringArray)
def _(pa_array: pa.StringArray | pa.LargeStringArray):
    return create_str_array(pa_array)


@arrow_array_adapter.register(pa.TimestampArray)
def _(pa_array: pa.TimestampArray):
    timestamp_type: pa.TimestampType = pa_array.type
    timestamp_unit = timestamp_type.unit
    return cast_64bit_date_arrow_to_numpy_array(pa_array, np.dtype(f"datetime64[{timestamp_unit}]"))
