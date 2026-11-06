"""Functions that adapt pyarrow arrays of several layouts to numpy arrays, and helpers for readable error messages.

:func:`~numbarrow.core.adapters.arrow_array_adapter` picks one of these adapters by the type of the array it is given.
Each adapter returns the array's data and its validity bitmap: a uint8 array of one bit per element, which
:func:`~numbarrow.core.is_null.is_null` reads, or None for an array without one. :func:`uniform_arrow_array_adapter`
views fixed-width data without a copy, and :func:`create_str_array` copies strings into a fixed-width Unicode array.
:func:`structured_array_adapter` and :func:`structured_list_array_adapter` adapt a struct array and a list of structs
field by field. :func:`create_bitmap` copies a validity bitmap. :func:`type_repr`, :func:`renamed` and
:class:`MissingKeyError` keep the error messages of numbarrow short and readable.
"""

import numpy as np
import pyarrow as pa

from numbarrow.utils.utils import arrays_viewers


# uniform_arrow_array_adapter views each Arrow type in this map as the numpy dtype that the map gives it. The map is
# written out instead of derived from pa.DataType.to_pandas_dtype(), because that method imports pandas. numbarrow
# does not require pandas, and without it such a lookup would fail for every int and float column. pa.bool_() is
# absent on purpose: Arrow packs booleans one bit per element, so a view of one byte per element would read wrong
# values. The boolean handler in numbarrow.core.adapters unpacks the bits instead.
arrow_to_numpy_dtypes = {
    pa.float64(): np.float64,
    pa.int32(): np.int32,
    pa.int64(): np.int64,
    pa.uint8(): np.uint8,
}

# The length, in characters, of the longest type text that an error message quotes in full; type_repr cuts a longer
# one.
TYPE_REPR_WIDTH = 120


def type_repr(arrow_type) -> str:
    """Return ``str(arrow_type)``; if it is longer than ``TYPE_REPR_WIDTH`` characters, cut it there and add a note.

    The note counts the characters cut and, for a struct type, the struct's fields. A message that quotes a pyarrow type
    grows with the schema: the text of a struct of a thousand fields with short names runs to about 13,000 characters,
    and a message that long fills every log that records the traceback. A type whose text fits within the width, such
    as a struct of a few fields, comes back unchanged.
    """
    text = str(arrow_type)
    if len(text) <= TYPE_REPR_WIDTH:
        return text
    cut = text[:TYPE_REPR_WIDTH]
    more = len(text) - TYPE_REPR_WIDTH
    if pa.types.is_struct(arrow_type):
        return f"{cut}... ({arrow_type.num_fields} fields, {more} more characters)"
    return f"{cut}... ({more} more characters)"


class MissingKeyError(KeyError):
    """A ``KeyError`` whose message prints as written, without the quotes that a plain ``KeyError`` adds.

    A plain ``KeyError`` prints the repr of its one argument, so a sentence raised through it prints inside a second
    pair of quotes, and a sentence that holds quotes of its own can print with backslashes before them.
    ``except KeyError`` still catches a ``MissingKeyError``; only the printed message differs. With no argument or with
    several, it prints as a plain ``KeyError`` does.
    """

    def __str__(self):
        return str(self.args[0]) if len(self.args) == 1 else super().__str__()


def renamed(exc: Exception, prefix: str) -> Exception:
    """Return a new exception whose message is ``prefix``, a colon and the message of ``exc``.

    A ``KeyError`` that is not a pyarrow error comes back as a :class:`MissingKeyError`, because a ``KeyError`` prints
    the repr of its argument, and one rebuilt from that text would quote the message twice. Otherwise the result keeps
    the class of ``exc`` if ``exc`` is a pyarrow error, or if its class is exactly ``TypeError``, ``ValueError``,
    ``NotImplementedError`` or ``OverflowError``; each of these can be built from one string. Any other exception, a
    subclass of those four included, comes back as a ``ValueError``. A class such as ``UnicodeDecodeError``, whose
    constructor takes five arguments, would raise a second error while the result is built, and the prefix would be
    lost. Raise the result ``from exc`` to keep the original traceback.
    """
    if isinstance(exc, KeyError) and not isinstance(exc, pa.ArrowException):
        text = str(exc.args[0]) if len(exc.args) == 1 else str(exc)
        return MissingKeyError(f"{prefix}: {text}")
    cls = type(exc)
    kept = (TypeError, ValueError, NotImplementedError, OverflowError)
    if not (isinstance(exc, pa.ArrowException) or cls in kept):
        cls = ValueError
    return cls(f"{prefix}: {exc}")


def create_bitmap(bitmap_buf: pa.Buffer | None, offset: int = 0, length: int = 0):
    """Copy the validity bits of ``length`` elements, from element ``offset`` on, into a new uint8 array.

    ``bitmap_buf`` is an Arrow validity buffer, such as ``array.buffers()[0]``, and ``offset`` and ``length`` are
    usually the array's own. A ``bitmap_buf`` of ``None`` means that the array carries no validity buffer, and the
    result is then ``None``. Otherwise the result holds one bit per element, from the least significant bit of its first
    byte on: bit ``i`` is 1 if element ``offset + i`` is valid, that is, not null, and 0 if it is null. This is the
    layout that :func:`~numbarrow.core.is_null.is_null` reads. The result has ``(length + 7) // 8`` bytes, none for a
    ``length`` of 0, and the bits after the last element have no defined value. The result owns its memory, so it stays
    correct after the Arrow array is freed. It is writable, and a write to it leaves the Arrow array unchanged.
    """
    if bitmap_buf is None:
        return None
    bitmap_p = bitmap_buf.address
    bitmap_len = bitmap_buf.size
    bitmap_viewer = arrays_viewers[np.uint8]
    raw_bitmap = bitmap_viewer(bitmap_p, bitmap_len)
    if length == 0:
        return raw_bitmap[:0].copy()
    num_bytes = (length + 7) // 8
    if offset == 0:
        # raw_bitmap views the Arrow buffer through its address, so it neither owns the buffer nor keeps it alive. The
        # bytes are therefore copied. A view would change under a caller who only reads it. Once the source array is
        # freed and its memory reused, the view reads other bytes, and valid elements can then look null. A write
        # through a view would also change the source's validity, even for elements outside a short slice.
        return raw_bitmap[:num_bytes].copy()
    # For a non-zero offset, the bytes that cover elements offset to offset + length - 1 are unpacked, cut to those
    # elements, and packed again, so that element offset lands on bit 0.
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
    # np.packbits returns a new array that owns its memory, and sliced_bits was padded to whole bytes above, so the
    # array holds exactly num_bytes bytes. A slice to num_bytes would remove nothing and would turn the result into a
    # view whose base is the packed array, so the array is returned as it is.
    return result


def create_str_array(pa_str_array: pa.StringArray | pa.LargeStringArray) -> tuple[np.ndarray | None, np.ndarray]:
    """Copy a string array into a read-only numpy array of fixed-width Unicode strings; return ``(bitmap, data)``.

    ``bitmap`` is the array's validity bitmap from :func:`create_bitmap`, or ``None`` if the array carries no validity
    buffer. ``data`` holds one string per element, and an empty string for a null element. Its width is the length, in
    characters, of the longest valid value, and at least one. Every element takes four bytes per character of that
    width, so one long value makes the whole array large. The bytes under a null element are never decoded and do not
    count toward the width.

    ``ValueError`` is raised if a valid value is not UTF-8, if a valid value ends in a NUL character, and if a non-empty
    array lacks its offsets or data buffer. numpy pads a shorter value with NUL characters, so a value's trailing NUL
    would be lost; a NUL at the start or in the middle of a value is kept.
    """
    bitmap_buf, offsets_buf, data_buf = pa_str_array.buffers()
    offset = pa_str_array.offset
    n = len(pa_str_array)
    if n == 0:
        # A zero-length array returns before any buffer is read, because pyarrow's full validation accepts one whose
        # offsets buffer is null. The result is read-only, as it is for a non-empty array, so that a caller gets a
        # read-only array whatever the batch's length.
        empty = np.empty((0,), dtype="|U1")
        empty.flags.writeable = False
        return create_bitmap(bitmap_buf, offset, 0), empty
    if offsets_buf is None or data_buf is None:
        # pyarrow refuses to build a non-empty string array without an offsets or data buffer. Imported through the
        # Arrow C Data Interface, a NULL data pointer under offsets that are all zero arrives as a buffer of 0 bytes,
        # not as None. The check is a guard for a pyarrow that does build one: the ValueError names the type, as the
        # uniform and boolean adapters do for a missing data buffer. Without this check, the first use of the missing
        # buffer raises an AttributeError or a TypeError that names neither the array nor its type.
        missing = "offsets" if offsets_buf is None else "data"
        raise ValueError(
            f"{pa_str_array.type} array of length {n} has no {missing} buffer"
        )
    buf_view = memoryview(data_buf)
    offsets_p = offsets_buf.address
    # A string array has int32 offsets, and a large string array has int64 offsets.
    offset_dtype = np.int64 if pa.types.is_large_string(pa_str_array.type) else np.int32
    offsets_len = offsets_buf.size // np.dtype(offset_dtype).itemsize
    offsets_array = arrays_viewers[offset_dtype](offsets_p, offsets_len)
    logical_offsets = offsets_array[offset:offset + n + 1]
    bounds = np.asarray(logical_offsets, dtype=np.int64)
    # The offsets of a null element may still span bytes: pc.if_else, for one, leaves the bytes of each value that it
    # masks out in place. Those bytes belong to no value. Decoding them would return a value that the caller masked
    # out and would size the array for a string that no caller can see. If the bytes are not UTF-8, it would also raise
    # UnicodeDecodeError on an array that the Arrow format allows. So live marks the valid elements, and only those are
    # measured and decoded.
    live = np.ones(n, dtype=bool)
    if pa_str_array.null_count:
        live = pa_str_array.is_valid().to_numpy(zero_copy_only=False).astype(bool)
    # numpy measures the width of a Unicode array in characters, and the difference of two Arrow offsets counts UTF-8
    # bytes. A width taken from the offsets would allocate up to four times the memory needed for non-ASCII text, for
    # every element of the array. Every UTF-8 byte except a continuation byte, 0b10xxxxxx, starts one character. One
    # vectorized pass over the bytes therefore counts the characters of every value, without decoding any value twice.
    span_start, span_stop = int(bounds[0]), int(bounds[n])
    if span_stop > span_start:
        raw = np.frombuffer(buf_view, dtype=np.uint8, count=span_stop - span_start, offset=span_start)
        starts_a_char = (raw & 0xC0) != 0x80
        # np.add.reduceat sums the values from each element's start index up to the next element's start index. The
        # last element's range runs to the end of the array. For an empty range, reduceat returns the value at the start
        # index instead of 0, so the count of every empty element is then set to 0. An empty last element starts at the
        # end of the span, and reduceat refuses an index equal to the array's length. The False appended to padded puts
        # that index inside the array. It adds nothing to the count of the last element, whose range now includes it.
        # Clamping the index instead would also move the end of the previous element's range, which would then lose its
        # last character.
        padded = np.append(starts_a_char, False)
        non_empty = bounds[1:] > bounds[:n]
        char_counts = np.add.reduceat(padded, bounds[:n] - span_start)
        char_counts = np.where(non_empty, char_counts, 0)
        # numpy pads each element of a fixed-width Unicode array with NUL characters, so an element cannot end in NUL:
        # storing "ab\x00" stores "ab". A NUL at the start or in the middle of a value is kept. A valid value that ends
        # in NUL is refused, because a silently shortened string is a wrong answer. The check leaves out two kinds of
        # element: an empty one, whose last_byte index of 0 points into another value, and a null one, whose bytes
        # belong to no value.
        last_byte = np.where(non_empty, bounds[1:] - 1 - span_start, 0)
        trailing_nul = non_empty & live & (raw[last_byte] == 0)
        if trailing_nul.any():
            index_ = int(np.flatnonzero(trailing_nul)[0])
            raise ValueError(
                f"element {index_} of a {n}-element string array ends in a NUL, "
                f"which a numpy fixed-width unicode array cannot represent: a "
                f"trailing NUL is how it pads a shorter element"
            )
    else:
        char_counts = np.zeros(n, dtype=np.int64)
    item_sz = int(char_counts[live].max()) if live.any() else 0
    # numpy fills a new unicode array with empty strings, because the U dtype needs initialization, so a null row that
    # the loop below skips holds the empty string.
    str_array = np.empty((n,), dtype=f"|U{item_sz}")
    for i in range(n):
        if not live[i]:
            continue
        start = int(bounds[i])
        length = int(bounds[i + 1]) - start
        try:
            str_array[i] = bytes(buf_view[start:start + length]).decode("utf-8")
        except UnicodeDecodeError as exc:
            # The position that a UnicodeDecodeError reports is a byte offset inside the element, so the message adds
            # the index of the element in the array.
            raise ValueError(
                f"element {i} of a {n}-element {pa_str_array.type} array is not valid UTF-8: {exc}"
            ) from exc
    bitmap = create_bitmap(bitmap_buf, offset, n)
    # The array is a new allocation, but it is read-only like the views that the fixed-width adapters return, so that a
    # caller gets a read-only array whatever the Arrow type.
    str_array.flags.writeable = False
    return bitmap, str_array


def _is_union_layout(arrow_type):
    """Return whether ``arrow_type`` is a union type, looking through any extension type to its storage type."""
    while isinstance(arrow_type, pa.BaseExtensionType):
        arrow_type = arrow_type.storage_type
    return pa.types.is_union(arrow_type)


# The struct adapters keep the validity of a struct's rows apart from the validity of each field, as Awkward Array's
# BitMaskedArray over a RecordArray does:
# https://awkward-array.org/doc/main/reference/generated/ak.contents.BitMaskedArray.html
def structured_array_adapter(struct_array: pa.StructArray) -> tuple[
    np.ndarray | None, dict[str, np.ndarray | None], dict[str, np.ndarray]
]:
    """Adapt a struct array field by field; return ``(struct_bitmap, bitmaps, datas)``.

    ``struct_bitmap`` is the validity bitmap of the struct's rows, or ``None`` if the struct carries no validity
    buffer. ``bitmaps`` and ``datas`` map each field name, in field order, to the field's validity bitmap and its data,
    adapted by :func:`~numbarrow.core.adapters.arrow_array_adapter` as an array of the field's type. A field's bitmap
    marks only the field's own nulls, and it is ``None`` if the field carries no validity buffer. A row that is null as
    a whole need not be marked in a field's bitmap, so pass both bitmaps to
    :func:`~numbarrow.core.is_null.is_null_struct`. Under such a row, a string field holds an empty string, and any
    other field holds whatever its buffer holds there.

    A field's data is a read-only view of the field's buffer if the field holds int32, int64, float64, uint8, date64 or
    timestamp values; the int64 values of a date64 or timestamp field need no conversion to ``datetime64``. The data is
    a read-only copy for three kinds of field: numbarrow unpacks the bits of a boolean field,
    turns the int32 days of a date32 field into ``datetime64[D]``, and copies a string field into fixed-width Unicode.

    ``TypeError`` is raised if ``struct_array`` is not a ``pyarrow.StructArray``. ``NotImplementedError`` is raised
    for repeated field names and for a field of a union type. A ``NotImplementedError``, ``ValueError``, ``TypeError``
    or pyarrow error from the adapter of a field, such as the error for an unsupported type, is raised again through
    :func:`renamed`, with ``struct field '<name>'`` and a colon in front of its message. A field that adapts but is
    itself a struct, a list of structs or a map raises ``NotImplementedError``.
    """
    # numbarrow.core.adapters imports this module, so this function imports the dispatcher when it runs. An import at
    # module scope would be circular.
    from numbarrow.core.adapters import arrow_array_adapter

    if not isinstance(struct_array, pa.StructArray):
        # The type is checked with an if, not an assert, because python -O removes asserts. Any other array would then
        # reach the field loop and fail there with an error about something else.
        raise TypeError(
            f"structured_array_adapter takes a StructArray, not {type(struct_array).__name__}"
        )
    data_type: pa.StructType = struct_array.type
    struct_bitmap_buf = struct_array.buffers()[0]
    struct_bitmap = create_bitmap(
        struct_bitmap_buf, struct_array.offset, len(struct_array)
    )
    names = [data_type[i].name for i in range(len(data_type))]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        # Arrow allows repeated field names, but the dicts of the result are keyed by field name, so they cannot hold
        # two fields of one name.
        raise NotImplementedError(
            f"Not implemented for a struct with repeated field names "
            f"{sorted(duplicates)} in {type_repr(struct_array.type)}"
        )
    bitmaps = {}
    datas = {}
    # struct_array.field(i) returns the child with only its own validity bits, so a row that is null at the struct
    # level can look valid in the child. Adapting that child would treat the bytes under such a row as a value. A
    # string field would take its width from those bytes and decode them. It would raise UnicodeDecodeError if they
    # are not UTF-8, and refuse them if they end in NUL. flatten() marks the struct's null rows as null in every child,
    # so the data is adapted from the flattened children. The bitmap that this function returns for a field still
    # comes from the raw child. The function returns the struct's validity and each field's apart, and is_null_struct
    # takes both, so merging them here would lose which of the two made a value null.
    raw_children = [struct_array.field(i) for i in range(len(data_type))]
    for field_ind, raw_child in enumerate(raw_children):
        if _is_union_layout(raw_child.type):
            # flatten() writes the struct's validity into each child, and a union has no validity buffer to write it
            # to. In pyarrow 24, flatten() aborts the process for a struct that has a null row and a union field. The
            # dispatcher would refuse the union with a NotImplementedError anyway, so a union field is refused before
            # anything is flattened, whether or not the struct has a null row.
            raise NotImplementedError(
                f"struct field {data_type[field_ind].name!r}: Not implemented for an array of "
                f"{len(raw_child)} elements of type {type_repr(raw_child.type)}, a union layout, which "
                f"cannot take the struct's validity"
            )
    masked = list(struct_array.flatten()) if struct_array.null_count else raw_children
    for field_ind in range(len(data_type)):
        field: pa.Field = data_type[field_ind]
        field_name = field.name
        raw_child = raw_children[field_ind]
        pa_array = masked[field_ind]
        # Each child goes through the dispatcher, not straight to uniform_arrow_array_adapter. A boolean child holds
        # one bit per element, which a uniform view would read as one byte per element. A child with more than two
        # buffers, such as a string child, has no uniform layout at all. It needs its own handler, or the dispatcher's
        # NotImplementedError for an unsupported type.
        try:
            adapted = arrow_array_adapter(pa_array)
        except (NotImplementedError, ValueError, TypeError, pa.ArrowException) as exc:
            # The dispatcher's message names the child's type but not the field. Many common Arrow types, such as
            # float32 and int16, have no handler, so the field's name is what tells the caller which field to change.
            raise renamed(exc, f"struct field {field_name!r}") from exc
        if len(adapted) != 2:
            raise NotImplementedError(
                f"Not implemented for struct field {field_name!r} of type {type_repr(pa_array.type)}: "
                f"a structured child adapts to per-field dictionaries, which do not fit "
                f"a single field's bitmap and data"
            )
        bitmap, data = adapted
        if struct_array.null_count:
            # The bitmap of a flattened child includes the struct's null rows, so the field's own bitmap is rebuilt
            # from buffers()[0] of the raw child. A field then gets the same bitmap whether or not the struct has null
            # rows.
            bitmap = create_bitmap(raw_child.buffers()[0], raw_child.offset, len(raw_child))
        bitmaps[field_name] = bitmap
        datas[field_name] = data
    return struct_bitmap, bitmaps, datas


def structured_list_array_adapter(list_array: pa.ListArray) -> tuple[
    np.ndarray | None, dict[str, np.ndarray | None], dict[str, np.ndarray]
]:
    """Adapt a list array of structs whose rows all have one length; return ``(struct_bitmap, bitmaps, datas)``.

    The result is what :func:`structured_array_adapter` returns for the elements of all the array's rows, in row order,
    so a field's data is a view or a copy as that function describes. The result carries no offsets: element ``k`` of
    row ``r`` is at index ``r * width + k``, where ``width`` is the common row length. ``struct_bitmap`` therefore marks
    null elements, not null rows. A ``pyarrow.MapArray`` is a list array whose elements are key and value structs, so
    it adapts to the two fields ``key`` and ``value``.

    ``TypeError`` is raised if ``list_array`` is not a ``pyarrow.ListArray``. ``NotImplementedError`` is raised if its
    elements are not structs, if a row is null, and if the rows differ in length, an empty row among longer ones
    included. Without offsets, the order of the elements alone cannot say which row an element belongs to.
    """
    if not isinstance(list_array, pa.ListArray):
        # The type is checked with an if, not an assert, because python -O removes asserts. Any other array would then
        # reach .values below and fail there with an AttributeError.
        raise TypeError(
            f"structured_list_array_adapter takes a ListArray, not {type(list_array).__name__}"
        )
    values: pa.Array = list_array.values
    if not pa.types.is_struct(values.type):
        # The message names the list's type and length, not the array. The repr of an array holds every value in it, so
        # it can be of any size, and it would copy the caller's data into whatever log records the traceback. The list
        # type already names the element type, so the message does not repeat it.
        raise NotImplementedError(
            f"Not implemented for a list array of {len(list_array)} rows of "
            f"type {type_repr(list_array.type)}: its elements are not structs"
        )
    # The result holds the elements of all the rows with no offsets, so a caller can map element i back to its row only
    # by assuming that every row has the same number of elements. A null row, an empty row and a row of another length
    # each break that assumption, and each shifts every element after it. The struct-level bitmap of the result covers
    # the elements, not the rows, so the result could not report such a row either. Such an array is refused instead
    # of answered wrongly.
    if list_array.null_count:
        raise NotImplementedError(
            f"Not implemented for a list array of {len(list_array)} rows of type "
            f"{type_repr(list_array.type)} with {list_array.null_count} null row(s): a "
            f"null row contributes no elements, so the flattened result would "
            f"silently misalign with the rows"
        )
    if len(list_array):
        bounds = list_array.offsets.to_numpy(zero_copy_only=False).astype(np.int64)
        widths = bounds[1:] - bounds[:-1]
        low, high = int(widths.min()), int(widths.max())
        if low != high:
            raise NotImplementedError(
                f"Not implemented for a list array of {len(list_array)} rows of type "
                f"{type_repr(list_array.type)} whose rows are not all the same length: "
                f"they run from {low} to {high}. Element order alone cannot say which "
                f"row an element belongs to unless every row is the same length, so "
                f"the result would silently misalign with the rows"
            )
    # values is the whole child array, whatever the offset and length of this array, so a sliced list would return the
    # elements of rows outside the slice. The offsets index into values directly, so the first and the last offset of
    # this array bound the elements of its rows.
    if len(list_array) == 0:
        # pyarrow's full validation accepts a zero-length list array whose validity and offsets buffers are null. In
        # pyarrow 24, reading the offsets of such an array crashes the process instead of raising, so the empty result
        # is built without reading them.
        data_values: pa.StructArray = values.slice(0, 0)
    else:
        offsets = list_array.offsets
        start = offsets[0].as_py()
        stop = offsets[len(list_array)].as_py()
        data_values = values.slice(start, stop - start)
    return structured_array_adapter(data_values)


def uniform_arrow_array_adapter(pa_array: pa.Array) -> tuple[np.ndarray | None, np.ndarray]:
    """View the data of a fixed-width array as a read-only numpy array, without a copy; return ``(bitmap, data)``.

    The array's type must be one that ``arrow_to_numpy_dtypes`` maps to a numpy dtype: float64, int32, int64 or uint8.
    Any other type raises ``ValueError``, a boolean one included, because Arrow stores one bit per boolean value.
    A non-empty array without a data buffer raises ``ValueError`` as well. ``bitmap`` is the array's validity bitmap
    from :func:`create_bitmap`, or ``None`` if the array carries no validity buffer. ``data`` views the array's data
    buffer from the array's offset on, and it keeps that buffer alive.

    ``data`` is read-only, because Arrow arrays are immutable, and its ``WRITEABLE`` flag cannot be set. pyarrow's own
    ``Array.to_numpy(zero_copy_only=True)`` returns the same kind of view. A slice or reshape of ``data`` that an
    ``@njit`` function returns is a new array whose flag can be set, and a store through it then changes the Arrow
    array. Declare a numba signature that receives ``data`` with ``readonly=True``; such a signature accepts writable
    arrays as well.
    """
    data_arrow_ty = pa_array.type
    data_np_ty = arrow_to_numpy_dtypes.get(data_arrow_ty, None)
    if data_np_ty is None:
        # The dtype is looked up before the buffers are unpacked, so that an array with more than two buffers, such as
        # a string array, is refused with a message that names its type instead of "too many values to unpack".
        raise ValueError(
            f"There is no numpy view type for {data_arrow_ty} in "
            f"`arrow_array_utils.arrow_to_numpy_dtypes`. Add it?"
        )
    bitmap_buf, data_buf = pa_array.buffers()
    data_item_byte_size = np.dtype(data_np_ty).itemsize
    data_len = len(pa_array)
    if data_len == 0:
        # A zero-length array returns before any buffer is read. Its offset may point past the end of an empty buffer,
        # where np.frombuffer raises a ValueError that names neither the array nor its type. pyarrow's full validation
        # also accepts one with a null data buffer. Such an array is the same logical array as one with empty buffers,
        # so both get the same empty result. structured_list_array_adapter returns early for a zero-length list for
        # the same reason. The result is read-only, as it is for a non-empty array, so that a caller gets a read-only
        # array whatever the batch's length.
        empty = np.empty((0,), dtype=data_np_ty)
        empty.flags.writeable = False
        return create_bitmap(bitmap_buf, pa_array.offset, 0), empty
    if data_buf is None:
        # pyarrow refuses to build a non-empty array without a data buffer, and pyarrow 14 and 24 also refuse one
        # imported through the Arrow C Data Interface. The check is a guard for a pyarrow that does not: the ValueError
        # names the type, where memoryview(None) below would raise a TypeError that names neither the array nor its
        # type.
        raise ValueError(f"{data_arrow_ty} array of length {data_len} has no data buffer")
    # The data is viewed through the pyarrow buffer, not through its address, so that the .base chain of the result
    # holds the buffer and keeps it alive. A view of a bare address owns nothing: once the caller drops the source
    # array, the view reads freed memory, which can give wrong values or crash the process.
    #
    # The view is read-only whatever the origin of the buffer. np.frombuffer makes a read-only view only of a buffer
    # that pyarrow marks immutable, and that depends on where the buffer came from. A locally built array is mutable,
    # while the same data after an Arrow IPC round trip, the transport that mapInArrow uses, is not. Clearing the
    # WRITEABLE flag after the view is made would not be enough, because a caller can set the flag again and then write
    # into the source array. numpy refuses to set WRITEABLE on an array whose base is read-only, so the memoryview is
    # made read-only first. The same rule makes pyarrow's own to_numpy(zero_copy_only=True) refuse the flag.
    #
    # The read-only memoryview is wrapped in a pyarrow buffer, and that buffer becomes the base of the result. Given the
    # memoryview itself, np.frombuffer would keep only a wrapper of it as .base. Calling release() on that wrapper ends
    # its hold on the source buffer, and a caller who did so and dropped the array would then read freed memory. A
    # pa.Buffer has no release() method, holds the memoryview and through it the source buffer, and exports its memory
    # read-only, so numpy still refuses to set WRITEABLE.
    data = np.frombuffer(
        pa.py_buffer(memoryview(data_buf).toreadonly()),
        dtype=data_np_ty,
        count=data_len,
        offset=pa_array.offset * data_item_byte_size
    )
    bitmap = create_bitmap(bitmap_buf, pa_array.offset, len(pa_array))
    return bitmap, data
