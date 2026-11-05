"""The adapters that turn pyarrow arrays into numpy arrays for numba-compiled code.

:func:`arrow_array_adapter` takes a pyarrow array and returns its validity bitmap and its values as numpy arrays, which
an ``@njit`` function can take as arguments. It dispatches on the class of the array, with
:func:`functools.singledispatch`, to a handler registered in this module. The values of an ``int32``, ``int64``,
``float64``, ``uint8``, ``date64`` or timestamp array come back as a view of the Arrow buffer. Booleans, ``date32``
values and strings are copied, because numpy lays them out differently. :func:`cast_64bit_date_arrow_to_numpy_array`
is the ``datetime64`` view that the ``date64`` and timestamp handlers share, and
:mod:`numbarrow.utils.arrow_array_utils` holds the functions that adapt numeric, string, struct and list arrays.
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
    """Return the bitmap of a ``date64`` or timestamp array and a ``datetime64`` view of its values, without a copy.

    Both types hold one int64 per value, as ``datetime64`` does, so the function views the Arrow buffer. ``np_dtype``
    must be ``datetime64`` at the array's own unit: ``ms`` for a ``date64``, and the unit of the type for a timestamp. A
    view cannot rescale the values, so any other ``np_dtype`` raises ``ValueError``. So does an array of another type,
    a ``time64`` or a ``duration`` included. The view is read-only, and the bitmap is None if the array has no validity
    buffer.
    """
    np_dtype = np.dtype(np_dtype)
    # A time64 or a duration array has a unit as well, but its values are a time of day or an elapsed span, not
    # instants. Viewed as datetime64, they would fall on 1970-01-01, so the type is checked, not the presence of a unit.
    if pa.types.is_date64(pa_array.type):
        unit = "ms"
    elif pa.types.is_timestamp(pa_array.type):
        unit = pa_array.type.unit
    else:
        raise ValueError(f"{type_repr(pa_array.type)} is not a date64 or timestamp type")
    # A view reads the int64 values as they are, so at another unit every value would name the wrong instant: noon on
    # 2020-01-01 as a timestamp[ms], viewed as datetime64[s], falls in the year 51971.
    if np_dtype != np.dtype(f"datetime64[{unit}]"):
        raise ValueError(
            f"{type_repr(pa_array.type)} holds {unit} instants, which view as datetime64[{unit}], not as {np_dtype}"
        )
    int64_array = pa_array.cast(pa.int64())
    # The cast reinterprets the buffer without copying it, which the assert checks. A cast of a zero-length array need
    # not keep the source's buffers, and an empty array has nothing to copy, so the check skips it.
    if len(pa_array):
        assert int64_array.buffers()[1].address == pa_array.buffers()[1].address, "got copied"
    bitmap, int64_data = uniform_arrow_array_adapter(int64_array)
    if not len(pa_array):
        # The cast of a zero-length array drops its validity buffer. The bitmap comes from the source instead, so that
        # it is None only if the source has no validity buffer.
        bitmap = create_bitmap(pa_array.buffers()[0], pa_array.offset, 0)
    data = int64_data.view(np_dtype)
    assert data.ctypes.data == int64_data.ctypes.data, "got copied"
    return bitmap, data


@singledispatch
def arrow_array_adapter(pa_array: pa.Array):
    """Return the validity bitmap and the values of a pyarrow array as numpy arrays for numba-compiled code.

    The function dispatches on the class of ``pa_array``, and a subclass of a registered class takes that class's
    handler: a ``MapArray`` adapts as the ``ListArray`` of key and value structs that it is. For the following classes,
    the result is ``(bitmap, data)``:

    - An ``Int32Array``, ``Int64Array``, ``DoubleArray`` or ``UInt8Array`` gives a view of its values.
    - A ``Date64Array`` or ``TimestampArray`` gives a ``datetime64`` view at its own unit, ``ms`` for a ``date64``. A
      timestamp's time zone is dropped: the values are the UTC instants that Arrow stores, as pyarrow's ``to_numpy``
      returns them.
    - A ``BooleanArray`` gives a copy as ``bool_``, and a ``Date32Array`` a copy as ``datetime64[D]``.
    - A ``StringArray`` or ``LargeStringArray`` gives a copy as a fixed-width numpy unicode array, as many characters
      wide as its longest non-null value, and at least one. The width differs from one array to the next, as between
      two batches of a Spark column, so compile a numba function that takes the strings lazily, without a signature.

    For a ``StructArray``, and for a ``ListArray`` whose elements are structs, the result is
    ``(struct_bitmap, bitmaps, datas)``: the struct's own bitmap, then two dicts that map each field's name to the
    field's bitmap and to its values, each adapted as above. A field's bitmap need not mark the struct's null rows, so
    pass it with ``struct_bitmap`` to :func:`~numbarrow.core.is_null.is_null_struct`. For a list array, the values are
    the elements of all its rows in order, without the offsets that separate the rows.

    A bitmap is None if the array has no validity buffer, and every value is then valid. Otherwise it is a writable
    uint8 array that owns its memory, with the bit of element 0 first, even for a slice. An array without nulls can
    still have a validity buffer: a ``slice``, ``take`` or ``filter`` of an array that has one keeps it. An array read
    from Arrow IPC, the format of Spark's ``mapInArrow`` batches, has none if it has no nulls. A numba signature should
    therefore declare a bitmap parameter ``Optional``.

    Every data array is read-only, a copy as well as a view, so declare it with ``readonly=True`` in a numba signature.
    A view keeps the Arrow buffer that it reads alive.

    Any other object raises ``NotImplementedError`` with a message that names its type. A ``ChunkedArray``, such as a
    ``Table`` column, and a pyarrow scalar raise it too, with a message that says what to pass instead. The functions
    that adapt strings, structs and lists refuse some values, such as a string that ends in a NUL character or a list
    array whose rows differ in length. :func:`~numbarrow.utils.arrow_array_utils.create_str_array`,
    :func:`~numbarrow.utils.arrow_array_utils.structured_array_adapter` and
    :func:`~numbarrow.utils.arrow_array_utils.structured_list_array_adapter` list the values they refuse.
    """
    # This base implementation receives every object that matches no registered class. Its message describes the
    # object without interpolating it, because the repr of an array holds every value: the message would grow with the
    # array and put the caller's data into any log that records the traceback.
    #
    # The object may lack an attribute that the message reads. A pa.Field has a type but no length, and a list or an
    # ndarray has a length but no type. Each attribute is therefore read with a check. Reading a missing one would
    # raise AttributeError or TypeError while the message is built, instead of the NotImplementedError that the
    # docstring promises, and a caller that catches NotImplementedError would not catch it.
    if isinstance(pa_array, pa.ChunkedArray):
        # A ChunkedArray, such as a Table column, has a type and a length. The generic message would describe it as an
        # array of a type that is supported, and never mention the chunks, so this message names them and the remedies.
        raise NotImplementedError(
            f"Not implemented for a ChunkedArray of {pa_array.num_chunks} chunks of type "
            f"{type_repr(pa_array.type)}: pass one chunk, or combine_chunks() first"
        )
    if isinstance(pa_array, pa.Scalar):
        # A scalar is one row of a column. The generic message cannot describe every scalar: reading the length of a
        # null list scalar raises TypeError, and a struct or map scalar has a length, so it would read as an
        # unsupported array of a type that is supported.
        raise NotImplementedError(
            f"Not implemented for a {type(pa_array).__name__} of type {type_repr(pa_array.type)}: "
            f"pass the Array, not one of its rows"
        )
    arrow_type = getattr(pa_array, "type", None)
    if not isinstance(arrow_type, pa.DataType):
        # A pandas DataFrame or a numpy record array with a column named "type" returns that column for the attribute,
        # so only a pyarrow DataType counts as a type.
        described = f"{type(pa_array).__name__}, which is not a pyarrow Array"
    elif hasattr(pa_array, "__len__"):
        described = f"an array of {len(pa_array)} elements of type {type_repr(arrow_type)}"
    else:
        described = f"{type(pa_array).__name__} of type {type_repr(arrow_type)}"
    raise NotImplementedError(f"Not implemented for {described}")


@arrow_array_adapter.register(pa.BooleanArray)
def _(pa_array: pa.BooleanArray):
    """Return the bitmap and a read-only ``bool_`` copy of the values.

    Arrow packs boolean values one bit per value, as it packs a validity bitmap, and numpy stores a ``bool_`` in a
    byte, so numpy cannot view the Arrow buffer.
    """
    bitmap_buf, data_buf = pa_array.buffers()
    num_of_bool_elements = len(pa_array)
    if num_of_bool_elements == 0:
        # pyarrow's full validation accepts a zero-length array without a data buffer. There is nothing to unpack, and
        # the result is the same as for a zero-length array with a data buffer.
        data = np.empty((0,), dtype=np.bool_)
        data.flags.writeable = False
        return create_bitmap(bitmap_buf, pa_array.offset, 0), data
    if data_buf is None:
        # pyarrow refuses to build a non-empty array without a data buffer, and pyarrow 14 and 24 also refuse one
        # imported through the Arrow C Data Interface. The check is a guard for a pyarrow that does not: the handler
        # then raises ValueError, instead of the AttributeError that reading the address of None would raise.
        raise ValueError(f"bool array of length {num_of_bool_elements} has no data buffer")
    data_buf_p = data_buf.address
    total_bits = pa_array.offset + num_of_bool_elements
    num_of_bytes = (total_bits + 7) // 8
    packed_boolean_data_viewer = arrays_viewers[np.uint8]
    packed_boolean_data = packed_boolean_data_viewer(data_buf_p, num_of_bytes)
    data = unpack_booleans(pa_array.offset, num_of_bool_elements, packed_boolean_data)
    bitmap = create_bitmap(bitmap_buf, pa_array.offset, len(pa_array))
    # The unpacked values are a new array, made read-only all the same, so that the data array is read-only whatever
    # the Arrow type.
    data.flags.writeable = False
    return bitmap, data


@arrow_array_adapter.register(pa.Date32Array)
def _(pa_array: pa.Date32Array):
    """Return the bitmap and a read-only ``datetime64[D]`` copy of the values.

    A ``date32`` value is the number of days since 1970-01-01 as an int32, and numpy's ``datetime64`` is an int64, so
    the values are converted, not viewed.
    """
    int32_array = pa_array.cast(pa.int32())
    # The cast reinterprets the buffer without copying it, which the assert checks. A cast of a zero-length array need
    # not keep the source's buffers, so the check skips it.
    if len(pa_array):
        assert int32_array.buffers()[1].address == pa_array.buffers()[1].address, "got copied"
    bitmap, int32_data = uniform_arrow_array_adapter(int32_array)
    if not len(pa_array):
        # As in cast_64bit_date_arrow_to_numpy_array, the cast of a zero-length array drops its validity buffer, so the
        # bitmap comes from the source.
        bitmap = create_bitmap(pa_array.buffers()[0], pa_array.offset, 0)
    data = int32_data.astype(np.dtype("datetime64[D]"))
    # astype copies the values into new memory, which the assert checks. numpy can give two empty arrays the same
    # address, so the check skips a zero-length array.
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
