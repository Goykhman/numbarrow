import numpy as np
import pyarrow as pa

from numba import njit
from numba.core.types import Array, boolean, int64, uint8
from typing import Dict, Optional, Tuple

from configurations import default_jit_options
from utils import arrays_viewers


@njit(boolean(int64, Array(uint8, 1, "C")), **default_jit_options)
def is_null(index_: int, bitmap: np.ndarray) -> bool:
    byte_for_index = bitmap[index_ // 8]
    bit_position_in_byte = index_ % 8
    return not (byte_for_index >> bit_position_in_byte) % 2


def uniform_arrow_array_adapter(pa_array: pa.Array) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """ NumPy adapter for PyArrow arrays with uniformly sized elements.
     Returns views over bitmap and data contiguous memory regions as numpy arrays. """
    bitmap_buf, data_buf = pa_array.buffers()
    data_arrow_ty = pa_array.type
    data_np_ty = data_arrow_ty.to_pandas_dtype()
    data_viewer = arrays_viewers.get(data_np_ty, None)
    if data_viewer is None:
        raise ValueError(f"There is no {data_np_ty} in `utils.arrays_viewers`. Add it?")
    data_p = data_buf.address
    data_buf_byte_size = data_buf.size
    data_item_byte_size = np.dtype(data_np_ty).itemsize
    data_len = data_buf_byte_size // data_item_byte_size
    data = data_viewer(data_p, data_len)

    if bitmap_buf is not None:
        bitmap_p = bitmap_buf.address
        bitmap_len = bitmap_buf.size
        bitmap_viewer = arrays_viewers[np.uint8]
        bitmap = bitmap_viewer(bitmap_p, bitmap_len)
    else:
        bitmap = None
    return bitmap, data


def structured_array_adapter(struct_array: pa.StructArray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    NumPy adapter of PyArrow `StructArray`.

    Returns tuple of two dictionaries, the first dictionary maps names of
    the structure fields to the contiguous bitmap array, the second maps
    these names to the contiguous values arrays.
    """
    assert isinstance(struct_array, pa.StructArray)
    data_type: pa.StructType = struct_array.type
    assert isinstance(data_type, pa.StructType)
    bitmaps = {}
    datas = {}
    for field_ind in range(len(data_type)):
        field: pa.Field = data_type[field_ind]
        field_name = field.name
        pa_array = struct_array.field(field_name)
        bitmap, data = uniform_arrow_array_adapter(pa_array)
        bitmaps[field_name] = bitmap
        datas[field_name] = data
    return bitmaps, datas


def structured_list_array_adapter(list_array: pa.ListArray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    NumPy adapter of PyArrow array of same-length lists of structures.

    :param list_array: PyArrow array with elements being of `pa.ListType`.
    Each list is in turn of the same length, and each element of the list
    is of `pa.StructType`.

    Returns tuple of two dictionaries, the first dictionary maps names of
    the structure fields to the contiguous bitmap array, the second maps
    these names to the contiguous values arrays.

    No data is copied as the data is uniform and is stored in columnar
    format, meaning that the underlying values are stored contiguously
    in a `pa.StructArray`.
    """
    assert isinstance(list_array, pa.ListArray)
    data_values: pa.StructArray = list_array.values
    return structured_array_adapter(data_values)
