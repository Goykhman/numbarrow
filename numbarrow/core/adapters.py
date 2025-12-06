import numpy as np
import pyarrow as pa

from functools import singledispatch
from typing import Union

from numbarrow.core.is_null import is_null
from numbarrow.utils.arrow_array_utils import (
    create_bitmap, create_str_array, structured_array_adapter,
    structured_list_array_adapter, uniform_arrow_array_adapter
)
from numbarrow.utils.utils import arrays_viewers


@singledispatch
def arrow_array_adapter(pa_array: pa.Array):
    """ Dispatcher for PyArrow array adapters of various types. """
    raise NotImplementedError(f"Not implemented for {pa_array} of type {pa_array.type}")


@arrow_array_adapter.register(pa.BooleanArray)
def _(pa_array: pa.BooleanArray):
    """ PyArrow stores boolean arrays bit-wise,
     following the same kind of layout it uses
     for bitmaps. This requires creating a copy.
    """
    bitmap_buf, data_buf = pa_array.buffers()
    data_buf_p = data_buf.address
    num_of_bool_elements = len(pa_array)
    num_of_bytes = num_of_bool_elements // 8
    if num_of_bool_elements % 8 > 0:
        num_of_bytes += 1
    packed_boolean_data_viewer = arrays_viewers[np.uint8]
    packed_boolean_data = packed_boolean_data_viewer(data_buf_p, num_of_bytes)
    data_lst = [not is_null(i, packed_boolean_data) for i in range(num_of_bool_elements)]
    data = np.array(data_lst, dtype=np.bool_)
    bitmap = create_bitmap(bitmap_buf)
    return bitmap, data


@arrow_array_adapter.register(pa.Date32Array)
def _(pa_array: pa.Date32Array):
    """ Creates a copy when re-interprets numpy array of integers
     (number of days since 1970-01-01) as datetime64[D] """
    int32_array = pa_array.cast(pa.int32())
    assert int32_array.buffers()[1].address == pa_array.buffers()[1].address, "got copied"
    bitmap, int32_data = uniform_arrow_array_adapter(int32_array)
    data = int32_data.astype(np.dtype("datetime64[D]"))
    # assert data.ctypes.data == int32_data.ctypes.data, "got copied"
    return bitmap, data


@arrow_array_adapter.register(pa.Date64Array)
def _(pa_array: pa.Date64Array):
    """ Creates a copy when re-interprets numpy array of integers
     (number of milliseconds since 1970-01-01) as datetime64[ms] """
    int64_array = pa_array.cast(pa.int64())
    assert int64_array.buffers()[1].address == pa_array.buffers()[1].address, "got copied"
    bitmap, int64_data = uniform_arrow_array_adapter(int64_array)
    data = int64_data.astype(np.dtype("datetime64[ms]"))
    # assert data.ctypes.data == int32_data.ctypes.data, "got copied"
    return bitmap, data


@arrow_array_adapter.register(pa.DoubleArray)
@arrow_array_adapter.register(pa.Int32Array)
@arrow_array_adapter.register(pa.Int64Array)
def _(pa_array: Union[
    pa.DoubleArray, pa.Int32Array, pa.Int64Array
]):
    return uniform_arrow_array_adapter(pa_array)


@arrow_array_adapter.register(pa.ListArray)
def _(pa_array: pa.ListArray):
    return structured_list_array_adapter(pa_array)


@arrow_array_adapter.register(pa.StructArray)
def _(pa_array: pa.StructArray):
    return structured_array_adapter(pa_array)


@arrow_array_adapter.register(pa.StringArray)
def _(pa_array: pa.StringArray):
    return {}, create_str_array(pa_array)
