"""
Utilities for extracting data from PyArrow array buffers as NumPy arrays.

Handles uniform arrays (fixed-width elements), string arrays (variable-length
with offset buffers), struct arrays, and list-of-struct arrays. Validity
bitmaps are extracted as uint8 arrays for use with :func:`~numbarrow.core.is_null.is_null`.
"""

import ctypes
import numpy as np
import pyarrow as pa

from typing import Dict, Optional, Tuple

from numbarrow.utils.utils import arrays_viewers


def create_bitmap(bitmap_buf: Optional[pa.Buffer], offset: int = 0, length: int = 0):
    """ Create numpy array of uint8 type containing
    bit-map of valid array entries, adjusted for array offset. """
    if bitmap_buf is None:
        return None
    bitmap_p = bitmap_buf.address
    bitmap_len = bitmap_buf.size
    bitmap_viewer = arrays_viewers[np.uint8]
    raw_bitmap = bitmap_viewer(bitmap_p, bitmap_len)
    if length == 0:
        return raw_bitmap[:0]
    num_bytes = (length + 7) // 8
    if offset == 0:
        return raw_bitmap[:num_bytes]
    # Re-pack bitmap bits starting from the offset bit position,
    # only reading the byte range that covers [offset, offset+length)
    first_byte = offset // 8
    last_byte = (offset + length + 7) // 8
    relevant = raw_bitmap[first_byte:last_byte]
    bit_offset = offset % 8
    bits = np.unpackbits(relevant, bitorder="little")
    sliced_bits = bits[bit_offset:bit_offset + length]
    pad = (-length) % 8
    if pad:
        sliced_bits = np.pad(sliced_bits, (0, pad), mode="constant")
    result = np.packbits(sliced_bits, bitorder="little")
    return result[:num_bytes]


def create_str_array(pa_str_array: pa.StringArray) -> np.ndarray:
    """ Copy data from densely packed `pa.StringArray` into
     padded numpy array of the character sequence type determined
     by the length of the longest string. """
    # Arrow StringArray layout: [validity_bitmap, offsets (int32), char_data (uint8)]
    # offsets[i] and offsets[i+1] delimit the byte range of string i in char_data
    bitmap_buf, offsets_buf, data_buf = pa_str_array.buffers()
    data_p = data_buf.address
    offsets_p = offsets_buf.address
    offsets_len = offsets_buf.size // np.dtype(np.int32).itemsize
    offsets_array = arrays_viewers[np.int32](offsets_p, offsets_len)
    offset = pa_str_array.offset
    n = len(pa_str_array)
    logical_offsets = offsets_array[offset:offset + n + 1]
    diffs = np.diff(logical_offsets)
    if len(diffs) == 0:
        return np.empty((0,), dtype="|U1")
    item_sz = diffs.max()
    str_array = np.empty((n,), dtype=f"|U{item_sz}")
    for i in range(n):
        start = logical_offsets[i]
        end = logical_offsets[i + 1]
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
    data_item_byte_size = np.dtype(data_np_ty).itemsize
    data_p = data_buf.address + pa_array.offset * data_item_byte_size
    data_len = len(pa_array)
    data = data_viewer(data_p, data_len)
    bitmap = create_bitmap(bitmap_buf, pa_array.offset, len(pa_array))
    return bitmap, data
