import ctypes
import numpy as np
import pyarrow as pa

from typing import Dict, Optional, Tuple

from numbarrow.utils.utils import arrays_viewers


def create_bitmap(bitmap_buf: Optional[pa.Buffer]):
    """ Create numpy array of uint8 type containing
    bit-map of valid array entries """
    if bitmap_buf is None:
        return None
    bitmap_p = bitmap_buf.address
    bitmap_len = bitmap_buf.size
    bitmap_viewer = arrays_viewers[np.uint8]
    bitmap = bitmap_viewer(bitmap_p, bitmap_len)
    return bitmap


def create_str_array(pa_str_array: pa.StringArray) -> np.ndarray:
    """ Copy data from densely packed `pa.StringArray` into
     padded numpy array of the character sequence type determined
     by the length of the longest string. """
    bitmap_buf, offsets_buf, data_buf = pa_str_array.buffers()
    data_p = data_buf.address
    offsets_p = offsets_buf.address
    offsets_len = offsets_buf.size // np.dtype(np.int32).itemsize
    offsets_array = arrays_viewers[np.int32](offsets_p, offsets_len)
    diffs = np.diff(offsets_array)
    str_array_len = len(diffs)
    item_sz = diffs.max()
    str_array = np.empty((str_array_len,), dtype=f"|U{item_sz}")
    for i in range(len(offsets_array) - 1):
        start = offsets_array[i]
        end = offsets_array[i + 1]
        s = (ctypes.c_char * int(end - start)).from_address(data_p + int(start)).value
        str_array[i] = s
    return str_array


def structured_array_adapter(struct_array: pa.StructArray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    NumPy adapter of PyArrow `StructArray`.

    Returns tuple of two dictionaries, the first dictionary maps names of
    the structure fields to the contiguous bitmap arrays, the second maps
    these names to the contiguous value arrays.
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

    Data is not copied as it is uniformly stored in a columnar format,
    that is, the underlying values are stored contiguously in a
    `pa.StructArray`.
    """
    assert isinstance(list_array, pa.ListArray)
    data_values: pa.StructArray = list_array.values
    return structured_array_adapter(data_values)


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
    bitmap = create_bitmap(bitmap_buf)
    return bitmap, data
