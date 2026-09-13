"""
Utilities for extracting data from PyArrow array buffers as NumPy arrays.

Handles uniform arrays (fixed-width elements), string arrays (variable-length
with offset buffers), struct arrays, and list-of-struct arrays. Validity
bitmaps are extracted as uint8 arrays for use with :func:`~numbarrow.core.is_null.is_null`.
"""

import numpy as np
import pyarrow as pa

from numbarrow.utils.utils import arrays_viewers


# The numpy view type for each Arrow type that has one, covering exactly the
# keys of ``arrays_viewers``. Resolved directly rather than through
# ``pa.DataType.to_pandas_dtype()``, which imports pandas, and so made a stock
# install unable to adapt an int or a float column.
# ``pa.bool_()`` is deliberately absent even though ``np.bool_`` is a key of
# ``arrays_viewers``: Arrow packs booleans one bit per element, so there is no
# uniform view of them, and the old ``to_pandas_dtype()`` lookup succeeded with
# an itemsize-1 viewer and returned wrong values. Boolean arrays go through the
# dispatcher's own handler, which unpacks the bits.
arrow_to_numpy_dtypes = {
    pa.float64(): np.float64,
    pa.int32(): np.int32,
    pa.int64(): np.int64,
    pa.uint8(): np.uint8,
}

# The longest pyarrow type an error message carries in full.
TYPE_REPR_WIDTH = 120


def type_repr(arrow_type) -> str:
    """``str(arrow_type)``, cut at a fixed width with a note of what was cut.

    A message that interpolates a pyarrow type grows with the schema: a
    thousand-field struct is 13,000 characters at the dispatcher, and a message
    that size lands in every log line that catches the traceback. The bound
    leaves the one-field case, which is the one people read, untouched.
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
    """A KeyError whose message reads as written.

    ``KeyError.__str__`` reprs its single argument, so a sentence raised
    through a plain KeyError arrives wrapped in a second pair of quotes, and
    one that was already rendered arrives mangled. ``except KeyError`` still
    catches this; only the rendering changes.
    """

    def __str__(self):
        return str(self.args[0]) if len(self.args) == 1 else super().__str__()


def renamed(exc: Exception, prefix: str) -> Exception:
    """The same exception with *prefix* in front of its message.

    The class is kept when it can be rebuilt from one string, which every
    pyarrow error and a plain TypeError, ValueError, NotImplementedError or
    OverflowError can; anything else, such as a UnicodeDecodeError with its
    five constructor arguments, comes back as a ValueError so that the prefix
    is never lost to a second error raised while building the message. A
    plain KeyError comes back as a :class:`MissingKeyError`, since its str()
    is the repr of its argument and a KeyError rebuilt from that would repr
    it again. Raise the result ``from exc`` to keep the original traceback.
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
    """ Create numpy array of uint8 type containing
    bit-map of valid array entries, adjusted for array offset.

    The returned array always owns its memory. The offset path below already
    produced a fresh array through ``np.packbits`` and is returned as-is; the
    offset-0 path has to copy, because there it would otherwise be a view over
    the Arrow validity buffer reached by raw address. That aliasing is not only
    a hazard for a caller who writes: the bits change under a caller who only
    reads, once the source array is collected and its buffer is reused, so a
    bitmap that read 0b11011011 comes back all-zero and every row looks null.
    Writing through it nulls out the source, and a short slice hands back bits
    the slice does not own. """
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
        return raw_bitmap[:num_bytes].copy()
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
    # Returned as-is. sliced_bits was padded to a whole number of bytes just
    # above, so packbits already produces exactly num_bytes of them, in a fresh
    # array that owns them. Slicing to num_bytes trimmed nothing and only cost
    # the copy that put base back to None.
    return result


def create_str_array(pa_str_array: pa.StringArray | pa.LargeStringArray) -> tuple[np.ndarray | None, np.ndarray]:
    """Copy data from a densely packed PyArrow string array into a padded
    NumPy Unicode array.

    StringArray uses int32 offsets; LargeStringArray uses int64 offsets.

    The padding is NUL, so a value whose last character is NUL cannot be told
    from a shorter one and raises ``ValueError`` rather than coming back
    truncated. Leading and interior NULs are representable and are preserved.
    """
    bitmap_buf, offsets_buf, data_buf = pa_str_array.buffers()
    offset = pa_str_array.offset
    n = len(pa_str_array)
    if n == 0:
        # Decided before the buffers are touched: the Arrow spec lets a
        # zero-length array carry null buffers, and pyarrow's own full
        # validation accepts one, so there is nothing to read here and nothing
        # that needs reading. Read-only like the non-empty path below, so that
        # the contract does not depend on whether the batch happened to be
        # empty.
        empty = np.empty((0,), dtype="|U1")
        empty.flags.writeable = False
        return create_bitmap(bitmap_buf, offset, 0), empty
    if offsets_buf is None or data_buf is None:
        # The uniform and boolean adapters name the type for a missing buffer;
        # without the same guard here the array dies on a bare AttributeError
        # naming neither the array nor its type. Unreachable through pyarrow,
        # which refuses to build a non-empty string array with null offsets,
        # but a foreign producer can hand one over the C Data Interface.
        missing = "offsets" if offsets_buf is None else "data"
        raise ValueError(
            f"{pa_str_array.type} array of length {n} has no {missing} buffer"
        )
    buf_view = memoryview(data_buf)
    offsets_p = offsets_buf.address
    offset_dtype = np.int64 if pa.types.is_large_string(pa_str_array.type) else np.int32
    offsets_len = offsets_buf.size // np.dtype(offset_dtype).itemsize
    offsets_array = arrays_viewers[offset_dtype](offsets_p, offsets_len)
    logical_offsets = offsets_array[offset:offset + n + 1]
    bounds = np.asarray(logical_offsets, dtype=np.int64)
    # A null slot's offsets are not required to be empty: pc.if_else leaves the
    # masked-out value's bytes in place. Those bytes belong to no logical value,
    # so decoding them would hand the caller a value it explicitly masked out,
    # size the array from a string that is not there, and raise
    # UnicodeDecodeError on a spec-legal array whose dead bytes are not utf-8.
    live = np.ones(n, dtype=bool)
    if pa_str_array.null_count:
        live = pa_str_array.is_valid().to_numpy(zero_copy_only=False).astype(bool)
    # The width is a character count. A difference of two Arrow offsets counts
    # utf-8 bytes instead, which over-allocates by up to 4x on non-ASCII input,
    # and numpy holds the whole array resident at that width. In utf-8 every
    # byte that is not a 0b10xxxxxx continuation byte starts exactly one
    # character, so the character counts come from one vectorised pass over the
    # bytes rather than from a second decode of every element.
    span_start, span_stop = int(bounds[0]), int(bounds[n])
    if span_stop > span_start:
        raw = np.frombuffer(buf_view, dtype=np.uint8, count=span_stop - span_start, offset=span_start)
        starts_a_char = (raw & 0xC0) != 0x80
        # reduceat sums raw[idx[i]:idx[i+1]]; for an empty range it yields
        # raw[idx[i]] instead of 0, which can only overstate an empty slot by
        # one and so cannot make the width too small.
        # An empty trailing element starts at the very end of the span, and
        # reduceat rejects an index equal to the length. One padding element
        # makes that index legal; it can only ever be summed into an empty
        # range, which the next line zeroes. Clamping instead would be wrong,
        # since an element's start index is also the previous element's upper
        # bound, so lowering it drops a character from the element before.
        padded = np.append(starts_a_char, False)
        non_empty = bounds[1:] > bounds[:n]
        char_counts = np.add.reduceat(padded, bounds[:n] - span_start)
        char_counts = np.where(non_empty, char_counts, 0)
        # A fixed-width |U element pads with NUL, so a trailing NUL in the
        # value is indistinguishable from that padding: assigning "ab\x00"
        # stores "ab", and the value never round trips. Leading and interior
        # NULs are unaffected and stay supported. Refused rather than
        # truncated, since a silently shortened string is still a wrong answer.
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
    str_array = np.empty((n,), dtype=f"|U{item_sz}")
    for i in range(n):
        if not live[i]:
            continue
        start = int(bounds[i])
        length = int(bounds[i + 1]) - start
        try:
            str_array[i] = bytes(buf_view[start:start + length]).decode("utf-8")
        except UnicodeDecodeError as exc:
            # The bare error's "position" is a byte offset inside the element,
            # which says nothing about which element of the batch it is in.
            raise ValueError(
                f"element {i} of a {n}-element {pa_str_array.type} array is not valid UTF-8: {exc}"
            ) from exc
    bitmap = create_bitmap(bitmap_buf, offset, n)
    # A fresh numpy allocation, but read-only all the same, so that the
    # contract does not depend on which Arrow type the caller happened to pass.
    str_array.flags.writeable = False
    return bitmap, str_array


# Two-layer struct nullability inspired by Awkward Array's BitMaskedArray(RecordArray)
# design. See: https://awkward-array.org/doc/main/reference/generated/ak.contents.BitMaskedArray.html


def structured_array_adapter(struct_array: pa.StructArray) -> tuple[
    np.ndarray | None, dict[str, np.ndarray | None], dict[str, np.ndarray]
]:
    """
    NumPy adapter of PyArrow `StructArray`.

    Returns a 3-tuple:
    - struct-level validity bitmap (None when the array carries no validity buffer)
    - dict mapping field names to per-field validity bitmaps
    - dict mapping field names to per-field value arrays
    """
    # Imported here rather than at module scope because the dispatcher's
    # handlers are defined in terms of the adapters in this module.
    from numbarrow.core.adapters import arrow_array_adapter

    if not isinstance(struct_array, pa.StructArray):
        # A bare assert here vanished under -O and let any array reach the
        # field loop, which then died on an error about something else.
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
        # Arrow permits repeated field names; keying by name cannot represent
        # them, and struct_array.field(name) raises a bare KeyError naming
        # neither the struct nor the cause.
        raise NotImplementedError(
            f"Not implemented for a struct with repeated field names "
            f"{sorted(duplicates)} in {type_repr(struct_array.type)}"
        )
    bitmaps = {}
    datas = {}
    # `field()` hands back the raw child, which keeps its own validity bits and
    # knows nothing about a row that is null at the struct layer. Adapting that
    # child treats such a row's bytes as a live value: it sizes a string
    # column's width from a value no caller can observe, decodes those bytes
    # and so raises UnicodeDecodeError on non-utf8 ones, and trips the
    # trailing-NUL refusal over a value the caller already nulled out.
    # `flatten()` applies the struct's validity to each child, which supplies
    # exactly that missing layer, so the DATA is read from the flattened child.
    # The reported bitmap still comes from the child's own buffer: this
    # function returns the two validity layers separately, on purpose, and
    # `is_null_struct` takes both, so folding the struct layer into the field
    # bitmap here would collapse a distinction the caller needs.
    raw_children = [struct_array.field(i) for i in range(len(data_type))]
    masked = list(struct_array.flatten()) if struct_array.null_count else raw_children
    for field_ind in range(len(data_type)):
        field: pa.Field = data_type[field_ind]
        field_name = field.name
        raw_child = raw_children[field_ind]
        pa_array = masked[field_ind]
        # Each child goes through the dispatcher rather than straight to
        # uniform_arrow_array_adapter. A boolean child is bit-packed, so the
        # uniform viewer would read it at one byte per element over a buffer
        # holding one bit per element; and a child with more than two buffers
        # has no uniform layout to view at all, so it needs its own handler or
        # the dispatcher's typed error.
        try:
            adapted = arrow_array_adapter(pa_array)
        except (NotImplementedError, ValueError, TypeError, pa.ArrowException) as exc:
            # The dispatcher's message names the child's type but not which
            # field of the struct it is, and thirteen of the common Arrow types
            # are unregistered, so the field name is what a caller needs.
            raise renamed(exc, f"struct field {field_name!r}") from exc
        if len(adapted) != 2:
            raise NotImplementedError(
                f"Not implemented for struct field {field_name!r} of type {type_repr(pa_array.type)}: "
                f"a structured child adapts to per-field dictionaries, which do not fit "
                f"a single field's bitmap and data"
            )
        bitmap, data = adapted
        if struct_array.null_count:
            # Recomputed from the raw child so the reported bitmap stays this
            # field's own validity, matching what every adapter derives from
            # buffers()[0] for an unmasked child.
            bitmap = create_bitmap(raw_child.buffers()[0], raw_child.offset, len(raw_child))
        bitmaps[field_name] = bitmap
        datas[field_name] = data
    return struct_bitmap, bitmaps, datas


def structured_list_array_adapter(list_array: pa.ListArray) -> tuple[
    np.ndarray | None, dict[str, np.ndarray | None], dict[str, np.ndarray]
]:
    """
    NumPy adapter of PyArrow array of same-length lists of structures.

    :param list_array: PyArrow array with elements being of `pa.ListType`.
        Each list is in turn of the same length, and each element of the list
        is of `pa.StructType`.

    Returns a 3-tuple of: the struct-level validity bitmap (or ``None`` when
    the elements carry no validity buffer), a dictionary mapping field names to
    per-field validity bitmaps (each ``None`` when that field carries no
    validity buffer), and a dictionary mapping field names to the contiguous
    field data arrays.

    Whether a field's data is copied depends on the field's type. A
    fixed-width child is a zero-copy view over the contiguous
    `pa.StructArray` values, except a `date32` child, whose int32 days are cast
    up to `datetime64[D]` and so allocate; `date64` and `timestamp` children
    are views, since their int64 values need no cast. A boolean child is
    bit-unpacked and a string child is repacked into fixed-width Unicode, so
    those are copies too.
    Every returned data array is read-only either way.

    The returned arrays are the flattened elements of every list in
    ``list_array``, with no offsets telling a caller where one row's elements
    end and the next row's begin. Mapping an element back to its row therefore
    assumes every list has the same length, which is what the ``:param:``
    contract above requires. A null row breaks that assumption outright, since
    it contributes no elements, so ``list_array.null_count`` must be zero and
    a non-zero one raises ``NotImplementedError``.
    """
    if not isinstance(list_array, pa.ListArray):
        # A bare assert here vanished under -O and let any array through to
        # `.values`, which then died on an AttributeError.
        raise TypeError(
            f"structured_list_array_adapter takes a ListArray, not {type(list_array).__name__}"
        )
    values: pa.Array = list_array.values
    if not pa.types.is_struct(values.type):
        # The array itself is deliberately not interpolated: its repr carries
        # every value it holds, which grows without bound and puts the
        # caller's data into whatever log catches the traceback. The element
        # type is already part of the list type, so it is not repeated.
        raise NotImplementedError(
            f"Not implemented for a list array of {len(list_array)} rows of "
            f"type {type_repr(list_array.type)}: its elements are not structs"
        )
    # The result is the flattened elements with no offsets, so a caller can
    # only map element i back to a row by assuming every row holds the same
    # number of elements. Three shapes break that assumption identically, by
    # shifting every element after them: a null row, an empty row and a row of
    # a different length. The struct-level fold covers the elements rather than
    # the rows, so none of it is reported anywhere either. Refused rather than
    # answered wrongly.
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
    # `values` is the whole child array and ignores this array's own offset,
    # so a sliced or offset list column would hand back the elements of rows
    # it does not contain. The offsets are absolute into `values`, so the
    # first and last of them bound exactly the elements these rows own.
    if len(list_array) == 0:
        # The Arrow spec lets a zero-length array carry null buffers, and
        # pyarrow's own full validation accepts one. Indexing a null offsets
        # buffer segfaults rather than raising, so there is nothing to read
        # here and nothing that needs reading.
        data_values: pa.StructArray = values.slice(0, 0)
    else:
        offsets = list_array.offsets
        start = offsets[0].as_py()
        stop = offsets[len(list_array)].as_py()
        data_values = values.slice(start, stop - start)
    return structured_array_adapter(data_values)


def uniform_arrow_array_adapter(pa_array: pa.Array) -> tuple[np.ndarray | None, np.ndarray]:
    """ NumPy adapter for PyArrow arrays with uniformly sized elements.

    Returns the validity bitmap, which owns its memory, and a zero-copy numpy
    view over the array's data buffer. The view is read-only and cannot be made
    writable: Arrow buffers are immutable by contract, and this is what
    pyarrow's own ``Array.to_numpy(zero_copy_only=True)`` returns. Declare numba
    signatures that receive it with ``readonly=True``, which accepts writable
    arrays too.
    """
    data_arrow_ty = pa_array.type
    data_np_ty = arrow_to_numpy_dtypes.get(data_arrow_ty, None)
    if data_np_ty is None:
        # Resolved before unpacking the buffers, so that a type with a
        # different buffer count reports its type rather than dying on the
        # unpack with "too many values to unpack".
        raise ValueError(
            f"There is no numpy view type for {data_arrow_ty} in "
            f"`arrow_array_utils.arrow_to_numpy_dtypes`. Add it?"
        )
    bitmap_buf, data_buf = pa_array.buffers()
    data_item_byte_size = np.dtype(data_np_ty).itemsize
    data_len = len(pa_array)
    if data_len == 0:
        # An empty array's offset may still point past the end of an empty
        # buffer, where np.frombuffer raises a bare "buffer is smaller than
        # requested size" naming neither the array nor the type. The spec also
        # lets a zero-length array carry a null data buffer, and pyarrow's own
        # full validation accepts one, so this answers that shape the same way
        # as the 0-byte-buffer array it is logically identical to.
        # `structured_list_array_adapter` reasons the same way about the same
        # shape; deciding it here keeps the two paths agreeing.
        empty = np.empty((0,), dtype=data_np_ty)
        # Read-only like the non-empty path below, so that the contract does
        # not depend on whether the batch happened to be empty.
        empty.flags.writeable = False
        return create_bitmap(bitmap_buf, pa_array.offset, 0), empty
    if data_buf is None:
        # Unreachable through pyarrow, which refuses to build a non-empty array
        # with a null data buffer, but a foreign producer can hand one over the
        # C Data Interface. A typed error beats dereferencing None.
        raise ValueError(f"{data_arrow_ty} array of length {data_len} has no data buffer")
    # Viewed through the buffer rather than through its raw address, so that
    # the result's ``.base`` chain reaches the pyarrow buffer and keeps it
    # alive. A view built over a bare address owns nothing and refers to
    # nothing, so once the caller drops the last reference to the source the
    # result reads freed memory: correct at tiny sizes, wrong from a few
    # hundred elements, and a segfault once the allocation is large enough to
    # be returned to the system.
    # The buffer is viewed read-only. np.frombuffer only reports read-only for
    # buffers pyarrow marks immutable, which depends on where the buffer came
    # from: a locally built array is mutable, while the same data after an
    # Arrow IPC round trip, which is the transport mapInArrow uses, is not.
    # Clearing the flag afterwards made the result the same either way, but a
    # flag is the caller's to flip back, and a store through the flipped view
    # changed the source Arrow array. numpy refuses to set WRITEABLE on an
    # array whose base is read-only, which is also what makes pyarrow's own
    # to_numpy(zero_copy_only=True) refuse the flip.
    data = np.frombuffer(
        memoryview(data_buf).toreadonly(),
        dtype=data_np_ty,
        count=data_len,
        offset=pa_array.offset * data_item_byte_size
    )
    bitmap = create_bitmap(bitmap_buf, pa_array.offset, len(pa_array))
    return bitmap, data
