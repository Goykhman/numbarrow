"""
Factory for PySpark ``mapInArrow`` UDF functions.

Bridges PySpark's Arrow-based batch processing with Numba JIT-compiled functions
by converting each :class:`pyarrow.RecordBatch` column through
:func:`~numbarrow.core.adapters.arrow_array_adapter` before passing the data
to a user-supplied computation function.
"""

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from collections.abc import Mapping
from types import MappingProxyType
from typing import Callable, NamedTuple

from numbarrow.core.adapters import arrow_array_adapter
from numbarrow.utils.arrow_array_utils import MissingKeyError, renamed, type_repr


class Nullable(NamedTuple):
    """An output column with its validity, ``Nullable(data, bitmap)``.

    ``data`` is anything a column may be: an ndarray, a list, a
    :class:`pyarrow.Array` or a numpy record array. ``bitmap`` is a packed
    uint8 validity bitmap in the layout ``bitmap_dict`` hands out,
    ``(rows + 7) // 8`` bytes with a set bit for a valid row, or ``None``,
    which is what ``bitmap_dict`` holds for a column with no validity buffer.
    A bare array carries no nulls out of a UDF; this does.
    """

    data: object
    bitmap: np.ndarray | None


# How many of the keys no declared field has a refusal lists. The listing is
# sized by the data, not the schema: a UDF keying a dict by a row value put
# every key of a 100,000-row batch, 1.5 MB, into the exception and twice into
# the executor logs.
KEYS_SHOWN = 10


def _struct_fields(struct_type):
    return [struct_type[i] for i in range(struct_type.num_fields)]


def _storage(arrow_type):
    """The type under any extension wrapping: a cast and a key check work on the storage."""
    while isinstance(arrow_type, pa.BaseExtensionType):
        arrow_type = arrow_type.storage_type
    return arrow_type


def _is_list_view(arrow_type):
    # The view layouts arrived in pyarrow 16; on an older one nothing is a view.
    is_view = getattr(pa.types, "is_list_view", None)
    is_large_view = getattr(pa.types, "is_large_list_view", None)
    return bool(is_view and is_view(arrow_type)) or bool(is_large_view and is_large_view(arrow_type))


def _is_list_like(arrow_type):
    return (pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type)
            or pa.types.is_fixed_size_list(arrow_type) or _is_list_view(arrow_type))


def _carries_keys(arrow_type):
    """Whether a value of this type is built from dicts somewhere inside it."""
    arrow_type = _storage(arrow_type)
    if pa.types.is_struct(arrow_type):
        return True
    if _is_list_like(arrow_type):
        return _carries_keys(arrow_type.value_type)
    if pa.types.is_map(arrow_type):
        return _carries_keys(arrow_type.key_type) or _carries_keys(arrow_type.item_type)
    return False


def _unexpected_fields(source_type, declared_type):
    """Field names the source type carries, at any depth, that the declared type does not.

    The kinds are paired through their layouts: an extension type through its
    storage, and a map with a declared list of key/value structs through its
    entries struct, since a cast matches those by name too and filled the
    value struct with nulls behind either wrapping.
    """
    source_type = _storage(source_type)
    declared_type = _storage(declared_type)
    if pa.types.is_map(source_type) and _is_list_like(declared_type):
        entries = pa.struct([source_type.key_field, source_type.item_field])
        return _unexpected_fields(entries, declared_type.value_type)
    if _is_list_like(source_type) and pa.types.is_map(declared_type):
        entries = pa.struct([declared_type.key_field, declared_type.item_field])
        return _unexpected_fields(source_type.value_type, entries)
    if pa.types.is_dictionary(source_type) and pa.types.is_dictionary(declared_type):
        # A dictionary is a layout: the cast decodes it and matches the value
        # structs by name, and filled a whole column with nulls the same way.
        return _unexpected_fields(source_type.value_type, declared_type.value_type)
    if pa.types.is_struct(source_type) and pa.types.is_struct(declared_type):
        declared = {field.name: field.type for field in _struct_fields(declared_type)}
        found = []
        for field in _struct_fields(source_type):
            if field.name not in declared:
                found.append(field.name)
            else:
                found.extend(_unexpected_fields(field.type, declared[field.name]))
        return found
    if _is_list_like(source_type) and _is_list_like(declared_type):
        return _unexpected_fields(source_type.value_type, declared_type.value_type)
    if pa.types.is_map(source_type) and pa.types.is_map(declared_type):
        return (_unexpected_fields(source_type.key_type, declared_type.key_type)
                + _unexpected_fields(source_type.item_type, declared_type.item_type))
    return []


def _iterable_rows(rows):
    """The rows a key check can look inside; the rest carry no keys of their own.

    A missing row is ``None`` in a list, ``NaN`` in a pandas Series of object
    dtype and ``pd.NA`` in a nullable one, and ``pa.array`` turns every one of
    them into a null. None of them is iterable, so a row that cannot be
    iterated is passed over here and left to ``pa.array``, which converts what
    it can and names the column when it cannot. A str, a bytes and an ndarray
    of a non-object dtype iterate, but over scalars that carry no keys, so
    they are passed over too: ``pa.array`` refuses such a row at its first
    element, where spreading it into a list here first took seconds and
    hundreds of megabytes for a long one.

    Iterability is tested with ``iter`` rather than by the ``__iter__``
    attribute: a sequence by ``__getitem__`` alone has no such attribute and
    iterates all the same, and ``pa.array`` reads it item by item. A pyarrow
    scalar row is passed over as well: ``pa.array`` checks one against the
    declared type itself, and from pyarrow 21 a MapScalar is a Mapping whose
    ``values`` is an array rather than a method.
    """
    kept = []
    for row in rows:
        if isinstance(row, (str, bytes, pa.Scalar)) or (isinstance(row, np.ndarray) and row.dtype.kind != "O"):
            continue
        try:
            iter(row)
        except TypeError:
            continue
        kept.append(row)
    return kept


def _check_keys(rows, arrow_type, where=""):
    """Refuse, at any depth, a dict key that no declared struct field has, naming the path to it.

    Arrow matches struct fields by exact name and fills a missing one with
    null, so a list of dicts keyed ``Amount`` against a field called
    ``amount`` builds a whole column of nulls under an identical schema,
    without a word. The same typo on a top-level key raises; this makes the
    nested one raise too, however deep the struct sits inside a list, a map or
    another struct. A row given as a tuple, a namedtuple or a pyspark Row
    binds by position, so its elements are checked against the fields in
    declared order, and one that names the declared fields in another order,
    which would swap every same-typed field without a word, is refused.
    """
    arrow_type = _storage(arrow_type)
    if pa.types.is_struct(arrow_type):
        fields = {field.name: field.type for field in _struct_fields(arrow_type)}
        names = list(fields)
        dicts = [row for row in rows if isinstance(row, Mapping) and not isinstance(row, pa.Scalar)]
        tuples = [row for row in rows if isinstance(row, tuple) and not isinstance(row, pa.Scalar)]
        seen = set()
        for row in dicts:
            seen.update(row)
        unexpected_keys = sorted(str(key) for key in seen - set(fields))
        if unexpected_keys:
            shown = unexpected_keys[:KEYS_SHOWN]
            more = f" and {len(unexpected_keys) - KEYS_SHOWN} more" if len(unexpected_keys) > KEYS_SHOWN else ""
            raise ValueError(
                f"{where}declared {type_repr(arrow_type)} but the dicts carry keys {shown}{more} "
                f"that no declared field has; Arrow matches struct fields by exact name and "
                f"fills a missing one with null"
            )
        for row in tuples:
            given = getattr(row, "_fields", None) or getattr(row, "__fields__", None)
            if given is not None and set(given) == set(names) and list(given) != names:
                raise ValueError(
                    f"{where}declared {type_repr(arrow_type)} but a row names its fields {list(given)}; a "
                    f"tuple's fields bind by position, so build it in the declared order or return dicts"
                )
        for index, (name, child_type) in enumerate(fields.items()):
            if _carries_keys(child_type):
                children = [row[name] for row in dicts if name in row]
                children.extend(row[index] for row in tuples if index < len(row))
                _check_keys(children, child_type, f"{where}field {name!r}: ")
    elif _is_list_like(arrow_type):
        _check_keys([item for row in _iterable_rows(rows) for item in row], arrow_type.value_type, f"{where}list item: ")
    elif pa.types.is_map(arrow_type):
        keys, items = _map_entries(rows)
        if _carries_keys(arrow_type.key_type):
            _check_keys(keys, arrow_type.key_type, f"{where}map key: ")
        if _carries_keys(arrow_type.item_type):
            _check_keys(items, arrow_type.item_type, f"{where}map value: ")


def _map_entries(rows):
    """The keys and the values of map rows given as dicts or as lists of pairs.

    Anything that is not a pair is left for ``pa.array`` to refuse, which
    names the column; indexing it here would not.
    """
    keys, items = [], []
    for row in _iterable_rows(rows):
        if isinstance(row, Mapping):
            keys.extend(row)
            items.extend(row.values())
        else:
            for pair in row:
                if isinstance(pair, Mapping) and set(pair) == {"key", "value"}:
                    # The entry shape Spark's map_entries produces, which
                    # pa.array reads as a pair.
                    keys.append(pair["key"])
                    items.append(pair["value"])
                elif isinstance(pair, (tuple, list)) and len(pair) == 2:
                    keys.append(pair[0])
                    items.append(pair[1])
    return keys, items


def _struct_column(children, rows, **layout):
    """A struct array from its children, or *rows* empty structs when there is no child to take a length from."""
    if not children:
        return pa.StructArray.from_buffers(pa.struct([]), rows, [None], children=[])
    return pa.StructArray.from_arrays(children, **layout)


def _record_field(value, name, arrow_type):
    """One field of a record array as an Arrow array; a failure names the field."""
    try:
        return _convert(value[name], arrow_type)
    except (pa.ArrowException, TypeError, ValueError, OverflowError, KeyError) as exc:
        raise renamed(exc, f"field {name!r}") from exc


def _record_to_struct(value, arrow_type):
    """A numpy record array as a struct column, one child per field.

    A record array is what an ``@njit`` function returns for a numba record
    type, and the one ndarray shape that means struct, but ``pa.array``
    refuses it with "Unsupported numpy type". Each field goes through the
    same conversion as a column of its own, so a unicode field keeps its NULs,
    a declared child type is honoured, and a field that fails to convert is
    named whether or not a type was declared. A record array with no fields
    becomes that many empty structs: a struct array with no children has no
    length of its own.
    """
    names = list(value.dtype.names)
    if arrow_type is None:
        children = [_record_field(value, name, None) for name in names]
        return _struct_column(children, len(value), names=names)
    if not pa.types.is_struct(arrow_type):
        raise TypeError(f"a record array with fields {names} cannot become {type_repr(arrow_type)}")
    fields = _struct_fields(arrow_type)
    unexpected_keys = sorted(set(names) - {field.name for field in fields})
    if unexpected_keys:
        raise ValueError(
            f"declared {type_repr(arrow_type)} but the record array carries fields "
            f"{unexpected_keys} that no declared field has"
        )
    children = []
    for field in fields:
        if field.name not in names:
            children.append(pa.nulls(len(value), type=field.type))
            continue
        children.append(_record_field(value, field.name, field.type))
    return _struct_column(children, len(value), fields=fields)


def _convert(value, arrow_type):
    """Build one Arrow array from a UDF output value.

    With a declared type the array is built as that type from the start
    rather than inferred and cast afterwards: a string column declared
    ``large_string`` is built as one, a list of dicts declared ``struct`` is
    built field by field and its keys are checked at every depth, and a list
    of dicts or of pairs declared ``map`` becomes a map, which no inferred
    struct can be cast to. A ready-built Arrow array is cast to the declared
    type only once its field names, at every depth, are found among the
    declared ones, since a cast matches struct fields by name as well.

    A numpy fixed-width unicode or bytes array handed straight to ``pa.array``
    is read with C string semantics, so a value is cut at its first NUL:
    ``"a\x00b"`` arrives as ``"a"`` and a leading NUL empties the value
    outright. Going via ``tolist()`` hands Arrow real Python strings and
    bytes, which carry NULs, at about 18% more time on a 200k-row column. A
    trailing NUL is dropped by ``tolist()`` itself, as numpy's own element
    access drops it, which matches the adapter refusing one on the way in;
    under a declared fixed-size binary type the array goes to ``pa.array``
    directly, which keeps every byte there.

    Without a declared type a unicode or bytes column is still named
    ``string`` or ``binary`` rather than inferred, because ``pa.array([])``
    infers ``null`` where the same column with rows infers ``string``: a UDF
    that filters a whole batch away would yield a schema its other batches do
    not share, and Spark's writer refuses the second schema it sees.
    """
    if isinstance(value, pa.ChunkedArray):
        value = value.combine_chunks()
    if isinstance(value, pa.Array):
        if arrow_type is None or value.type == arrow_type:
            return value
        unexpected = _unexpected_fields(value.type, arrow_type)
        if unexpected:
            raise ValueError(
                f"declared {type_repr(arrow_type)} but the array is {type_repr(value.type)}, whose "
                f"fields {unexpected} no declared field has; a cast matches struct fields by exact "
                f"name and fills a missing one with null"
            )
        return value.cast(arrow_type)
    if isinstance(value, np.ndarray) and value.dtype.kind != "O":
        return _ndarray_to_arrow(value, arrow_type)
    # An object array, a list, a tuple, a pandas Series or any other iterable
    # of Python objects.
    if not hasattr(value, "__len__"):
        # A generator would be consumed by the checks, so it is read once.
        value = list(value)
    if isinstance(value, (list, tuple)):
        for row in value:
            if _is_pandas(row, "Series", "DataFrame"):
                # pa.array reads a Series row by its index labels, so a sorted
                # or filtered one came back reordered, and one whose labels
                # were not 0..n-1 died on a bare KeyError.
                raise TypeError(
                    f"a row is a pandas {type(row).__name__}, which pa.array reads by its labels "
                    f"rather than in order; hand it over as row.to_numpy() or list(row)"
                )
    if arrow_type is not None and _carries_keys(arrow_type):
        _check_keys(value, arrow_type)
    array = pa.array(value, type=arrow_type)
    if isinstance(array, pa.ChunkedArray):
        # A pandas Series over a multi-chunk pyarrow array comes back as one,
        # which RecordBatch.from_arrays refused naming no column.
        array = array.combine_chunks()
    return array


def _at_arrow_unit(value):
    """A datetime64 or timedelta64 array at a unit pyarrow models, with any multiplier folded in.

    ``pa.array`` reads numpy's base unit and ignores a multiplier, so a
    ``datetime64[5s]`` column of five-second bins came back at one-second
    steps, 2020 read as 1980, and ``datetime64[2D]`` slipped past the day-unit
    inference in ``_ndarray_to_arrow`` into the misread it exists to prevent.
    A multiplier folds into its base unit exactly; an hour or minute unit
    becomes seconds and a week, month or year unit becomes days, exactly too,
    where ``pa.array`` refused them outright. A unit finer than a nanosecond,
    and a month or year timedelta, which has no fixed length, would not
    convert exactly and are refused instead.
    """
    family = "datetime64" if value.dtype.kind == "M" else "timedelta64"
    unit, count = np.datetime_data(value.dtype)
    if unit in ("ps", "fs", "as"):
        raise TypeError(
            f"a {value.dtype} array has no Arrow type: pyarrow models seconds down to nanoseconds; "
            f"convert it to {family}[ns] first, which drops the finer digits"
        )
    if family == "timedelta64" and unit in ("M", "Y"):
        raise TypeError(
            f"a {value.dtype} array has no fixed length in seconds; convert it to {family}[D] or "
            f"{family}[s] first"
        )
    if unit in ("h", "m") or (family == "timedelta64" and unit in ("W", "D")):
        target = "s"
    elif unit in ("W", "M", "Y"):
        target = "D"
    else:
        target = unit
    if target == unit and count == 1:
        return value
    return value.astype(f"{family}[{target}]")


def _ndarray_to_arrow(value, arrow_type):
    """An ndarray of a non-object dtype as an Arrow array; see ``_convert``."""
    if value.dtype.names is not None:
        return _record_to_struct(value, arrow_type)
    kind = value.dtype.kind
    if kind in ("U", "S") and value.ndim != 1:
        # A 0-d unicode or bytes array's tolist() is a bare scalar, which
        # pa.array spreads one character per row, and a 2-d one's is nested
        # lists; both defeat the one-dimensional refusal pa.array gives the
        # array itself, which every other dtype still gets, naming the column.
        raise TypeError(f"a {value.ndim}-dimensional {value.dtype} array; an output column is one-dimensional")
    if kind == "U":
        return pa.array(value.tolist(), type=arrow_type or pa.string())
    if kind == "S":
        if arrow_type is not None and pa.types.is_fixed_size_binary(arrow_type):
            # tolist() drops a trailing NUL, so a digest ending in 0x00 came
            # back a byte short and was refused under its fixed width.
            # pa.array keeps every byte of a fixed-width array under a
            # fixed-width type, and only there; variable-width binary still
            # cuts at the first NUL, so it keeps the tolist() route.
            return pa.array(value, type=arrow_type)
        return pa.array(value.tolist(), type=arrow_type or pa.binary())
    if kind in ("M", "m"):
        value = _at_arrow_unit(value)
    if value.dtype == np.dtype("datetime64[D]") and arrow_type is not None:
        # Under a declared timestamp or int32 ``pa.array`` reads a day-unit
        # array's 8-byte values as the 4-byte days of a date32, so every other
        # row came back the epoch, or a zero. Inferred first the array is a
        # date32, and the cast gives the day numbers under int32, midnights
        # under a timestamp, and leaves pyarrow's own refusals in place.
        return pa.array(value).cast(arrow_type)
    return pa.array(value, type=arrow_type)


def _is_pandas(value, *names):
    """Whether *value* is a pandas object of one of the given class names, without importing pandas."""
    cls = type(value)
    return cls.__name__ in names and cls.__module__.split(".")[0] == "pandas"


def _split_pair(value):
    """The data and the bitmap of a :class:`Nullable`; any other shape carries no bitmap.

    A bare tuple is not read as a pair: it goes to ``pa.array`` as the
    sequence it is, since a 2-tuple was a two-row column before ``Nullable``
    existed and any rule on a bare tuple would misread one shape or another.
    """
    if isinstance(value, Nullable):
        return value.data, value.bitmap
    return value, None


def _with_validity(array, bitmap):
    """The same array with *bitmap*, a packed validity bitmap, folded in.

    The bitmap has the layout ``bitmap_dict`` hands out and
    :func:`~numbarrow.core.is_null.is_null` reads: one bit per row, LSB first,
    set for a valid row, ``(rows + 7) // 8`` bytes of uint8. A fixed-width or
    string array with no nulls yet takes the bitmap as its validity buffer
    and keeps its data buffer, so neither side is copied when the bitmap is
    contiguous, which is the case for every bitmap ``bitmap_dict`` hands out.
    Any other array, one that already carries nulls, a sliced Arrow array or
    a nested type, is masked through ``if_else``, which keeps the nulls it
    had; an extension array is masked through its storage and rewrapped. The
    bitmap carries no row count of its own: the length check here is
    per byte, eight rows to a byte, and the caller checks a bitmap the batch
    handed out against the count it was handed out for. A bitmap that is not
    an ndarray at all is refused before any attribute of it is read, since the
    AttributeError that reading one raises is outside the classes the caller
    catches and would escape naming neither the column nor the layout.
    """
    if not isinstance(bitmap, np.ndarray):
        raise TypeError(
            f"the bitmap of a (data, bitmap) pair must be the packed uint8 array bitmap_dict "
            f"hands out, or None, not a {type(bitmap).__name__}"
        )
    if bitmap.dtype != np.uint8 or bitmap.ndim != 1:
        raise TypeError(
            f"the bitmap of a (data, bitmap) pair must be the packed uint8 array bitmap_dict "
            f"hands out, or None, not a {bitmap.ndim}-dimensional {bitmap.dtype} array"
        )
    rows = len(array)
    if len(bitmap) != (rows + 7) // 8:
        raise ValueError(
            f"the bitmap has {len(bitmap)} bytes, which covers {8 * len(bitmap)} rows, but the "
            f"column has {rows} rows"
        )
    if rows == 0:
        return array
    if isinstance(array, pa.ExtensionArray):
        # The flat test below reads the extension type, which reports no
        # fields and no dictionary whatever its storage, so a dictionary
        # storage took the from_buffers path and aborted the interpreter, and
        # a null or a slice went to if_else, which has no extension kernel.
        # The storage carries the layout; the result is rewrapped.
        return pa.ExtensionArray.from_storage(array.type, _with_validity(array.storage, bitmap))
    flat = (array.null_count == 0 and array.offset == 0 and array.type.num_fields == 0
            and not pa.types.is_dictionary(array.type) and not pa.types.is_null(array.type))
    if flat:
        buffers = [pa.py_buffer(np.ascontiguousarray(bitmap))] + list(array.buffers()[1:])
        return pa.Array.from_buffers(array.type, rows, buffers)
    valid = pa.array(np.unpackbits(bitmap, bitorder="little")[:rows].astype(bool))
    return pc.if_else(valid, array, pa.scalar(None, type=array.type))


def _to_arrow(value, name, arrow_type=None, handed=MappingProxyType({})):
    """Convert one UDF output column to an Arrow array, naming the column on any failure.

    A :class:`Nullable` is built from its data and then given its bitmap as
    validity. A bitmap the batch handed out is right only for a column of the
    count it was handed out for, which is the batch's rows for a column's own
    bitmap and the flattened elements for a struct field's, and a packed
    bitmap cannot tell one row count from another inside the same byte, so
    that case is refused here by identity rather than left to the byte check.
    ``pa.array`` iterates a Mapping, so a dict of arrays returned under one
    key silently became a string column of the dict's keys, with a different
    row count and nothing raised; it is refused outright. A str or bytes
    returned as a column is refused the same way: ``pa.array`` spreads it one
    character per row, so ``{"country": "US"}`` over a two-row batch was the
    rows ``U`` and ``S``. Every other failure on the output side named no
    column at all.
    """
    value, bitmap = _split_pair(value)
    if isinstance(value, Nullable):
        # A helper's Nullable wrapped once more went to pa.array as the 2-tuple
        # it is: two rows, the data as one and the bitmap's bytes as the other.
        raise TypeError(
            f"output column {name!r} is a Nullable inside a Nullable, which pa.array would read as "
            f"a two-row column of its data and its bitmap; wrap the data once"
        )
    if isinstance(value, Mapping):
        raise TypeError(
            f"output column {name!r} is a {type(value).__name__}, which pa.array would read as "
            f"its keys; return an ndarray, a list or a pyarrow Array per column, and for a "
            f"struct column a list of dicts or a record array"
        )
    if isinstance(value, (str, bytes)):
        raise TypeError(
            f"output column {name!r} is a {type(value).__name__}, which pa.array would spread one "
            f"character per row; a constant column is np.full(rows, value)"
        )
    if _is_pandas(value, "DataFrame"):
        # Read by its column labels: a one-column frame died on a bare
        # KeyError(0), and one with integer labels came back transposed.
        raise TypeError(
            f"output column {name!r} is a DataFrame, which pa.array reads by its column labels; "
            f"return one Series or ndarray per column"
        )
    try:
        array = _convert(value, arrow_type)
        if bitmap is None:
            return array
        _, covers = handed.get(id(bitmap), (None, None))
        if covers is not None and len(array) != covers:
            raise ValueError(
                f"the bitmap is one this batch handed out for {covers} rows, but the column has "
                f"{len(array)} rows; a resized column needs a bitmap of its own"
            )
        return _with_validity(array, bitmap)
    except (pa.ArrowException, TypeError, ValueError, OverflowError, KeyError) as exc:
        raise renamed(exc, f"output column {name!r}") from exc


def _handed_bitmaps(data_dict, bitmap_dict):
    """Each bitmap the batch hands out, by id, with the bitmap itself and the count of rows it covers.

    A column's bitmap covers the batch's rows, a struct field's covers the
    field's own elements, and for a list of structs those are the flattened
    elements rather than the outer rows. The count therefore comes from the
    data handed out beside the bitmap, never from the batch. The bitmap rides
    along to stay alive for the batch, and no longer: an id is reusable once its object is
    freed, and a UDF that drops a bitmap from ``bitmap_dict`` frees it, after
    which a bitmap of its own could land on that id and be refused as the
    handed-out one.
    """
    handed = {}
    for name, bitmaps in bitmap_dict.items():
        datas = data_dict[name]
        leaves = bitmaps.items() if isinstance(bitmaps, dict) else [(None, bitmaps)]
        for field, bitmap in leaves:
            if bitmap is not None:
                handed[id(bitmap)] = (bitmap, len(datas if field is None else datas[field]))
    return handed


def _build_batch(outputs, output_schema, handed=MappingProxyType({})):
    """The RecordBatch a UDF's result becomes, bound to ``output_schema`` when there is one.

    ``handed`` maps the id of each bitmap the batch handed the UDF to that
    bitmap and the row count it covers; see ``_to_arrow``.
    """
    if not isinstance(outputs, Mapping):
        raise TypeError(
            f"main_func must return a dict of column name to array, not "
            f"{type(outputs).__name__}"
        )
    if output_schema is None:
        names = list(outputs)
        for name in names:
            if not isinstance(name, str):
                raise TypeError(
                    f"output column {name!r} is a {type(name).__name__}, not a str: the keys of the dict "
                    f"main_func returns are its column names"
                )
        arrays = [_to_arrow(outputs[name], name, None, handed) for name in names]
    else:
        extra = [name for name in outputs if name not in output_schema.names]
        if extra:
            raise ValueError(
                f"main_func returned columns {extra} that output_schema does not name; "
                f"it names {output_schema.names}"
            )
        names = output_schema.names
        arrays = []
        for field in output_schema:
            if field.name not in outputs:
                raise MissingKeyError(
                    f"output_schema names column {field.name!r}, which main_func did not "
                    f"return; it returned {list(outputs)}"
                )
            arrays.append(_to_arrow(outputs[field.name], field.name, field.type, handed))
    lengths = {name: len(array) for name, array in zip(names, arrays)}
    if len(set(lengths.values())) > 1:
        # pyarrow's own refusal says "2 vs 3" and names neither column.
        raise ValueError(f"output columns differ in length: {lengths}")
    if output_schema is None:
        return pa.RecordBatch.from_arrays(arrays, names=names)
    return pa.RecordBatch.from_arrays(arrays, schema=output_schema)


def _repeated_names(fields):
    """Names declared more than once among *fields* or inside any of their types, at any depth."""
    names = [field.name for field in fields]
    repeated = sorted({name for name in names if names.count(name) > 1})
    for field in fields:
        arrow_type = _storage(field.type)
        if pa.types.is_struct(arrow_type):
            repeated.extend(_repeated_names(_struct_fields(arrow_type)))
        elif _is_list_like(arrow_type):
            repeated.extend(_repeated_names([arrow_type.value_field]))
        elif pa.types.is_map(arrow_type):
            repeated.extend(_repeated_names([arrow_type.key_field, arrow_type.item_field]))
    return repeated


def _schema_drift(first, later):
    """Why a batch built by inference differs from the partition's first, naming the column."""
    remedy = ("Spark's writer refuses a batch whose schema differs from the first it wrote, so return the "
              "same columns every batch and declare output_schema where a batch may be empty or all null")
    if first.names != later.names:
        return f"this batch built columns {later.names} where the first built {first.names}; {remedy}"
    for name in first.names:
        before, now = first.field(name).type, later.field(name).type
        if before != now:
            return (f"output column {name!r} was inferred as {type_repr(before)} from the first batch and "
                    f"{type_repr(now)} from this one; {remedy}")
    return f"this batch's schema differs from the first batch's; {remedy}"


def _fold_struct_validity(struct_bitmap, field_bitmap):
    """Combine a struct's own validity bits into one field's bits.

    A row that is null as a whole leaves its fields' own bitmaps untouched, so
    a field bitmap alone cannot see it. Arrow validity is 1 for valid, so the
    two layers combine with a bitwise and.
    """
    if struct_bitmap is None:
        return field_bitmap
    if field_bitmap is None:
        # Copied per field so that two fields do not share one array, where a
        # caller writing through one would change the other.
        return struct_bitmap.copy()
    return struct_bitmap & field_bitmap


def make_mapinarrow_func(
    main_func: Callable,
    input_columns: list[str] | None = None,
    broadcasts: dict | None = None,
    output_schema: pa.Schema | None = None
):
    """
    Creates a function that can be given as an argument to `mapInArrow`

    :param main_func: called once per :class:`pyarrow.RecordBatch` as
        ``main_func(data_dict, bitmap_dict, broadcasts)``, returning a dict
        that maps each output column's name to an ndarray, a list or a
        :class:`pyarrow.Array`, from which a PyArrow ``RecordBatch`` is built.
        A numpy record array becomes a struct column, one child per field.
        A null comes out of a list, a tuple or an object array holding
        ``None``, a :class:`pyarrow.Array`, a numpy masked array, and a
        :class:`Nullable`, ``Nullable(data, bitmap)``, whose ``bitmap`` is a
        packed uint8 validity bitmap in the layout ``bitmap_dict`` hands out,
        ``(rows + 7) // 8`` bytes with a set bit for a valid row, or ``None``.
        A bare array carries no nulls out: a row that came in null goes out
        valid, holding whatever the UDF computed from the placeholder under
        the null.  A result that is null exactly where one input column is
        passes that column's bitmap through,
        ``{"out": Nullable(result, bitmap_dict["value"])}``, or
        ``bitmap_dict["column"]["field"]`` for a struct field, and any other
        result builds its own in that layout, the one
        :func:`~numbarrow.core.is_null.is_null` reads, for instance
        ``np.packbits(valid, bitorder="little")`` from a boolean array.  A bitmap
        that is not an ndarray, or is one of another length or dtype, raises
        naming the column, and so does a bitmap the batch handed out on a
        column whose row count is not the count that bitmap covers, the
        batch's rows for a column's own bitmap and the flattened elements for
        a struct field's, since a packed bitmap cannot tell row counts apart
        inside one byte.

        Spark binds the columns of that batch to the declared output schema by
        POSITION, not by name, and checks nothing about their names: it reads
        each Arrow vector through the accessor its declared type expects, so
        an int64 column declared as a timestamp reads as a timestamp, and two
        columns whose types share an accessor family swap silently when the
        dict is built in the other order.  A column read through the accessor
        of another family fails in the JVM with
        ``java.lang.UnsupportedOperationException`` whatever its width, as
        float64 under ``LongType`` and int64 under ``DoubleType`` do, both 64
        bits wide, and so does a width mismatch inside one family, such as
        int32 under ``LongType``.  Build the returned dict in the order the
        output schema declares, or pass ``output_schema`` and let Arrow bind
        it by name instead.

        ``data_dict`` maps each selected column's name to its data.  A column
        of a uniform type maps to one array.  A struct or list-of-struct
        column maps to a dict of its fields, ``data_dict[column][field]``, so
        a field never shares a namespace with another column or with another
        struct's fields.

        ``bitmap_dict`` has the same shape: a uint8 aligned array of bitmap
        data, or ``None`` where the column carries no validity buffer, and for
        a struct column a dict of those keyed by field.  Every key of
        ``data_dict`` is present, so a null-free batch is indexable exactly
        like a batch containing nulls.

        For a ``StructArray`` column the struct-level validity is folded into
        each field's bitmap, so one
        :func:`~numbarrow.core.is_null.is_null` call per field sees both a null
        field and a row that is null as a whole.

        For a ``ListArray`` of structs the fold covers the flattened struct
        elements, NOT the outer list rows.  A null outer row can be reported
        nowhere, and because the adapter returns the flattened elements with
        no offsets, such a row also shifts the element-to-row mapping, so a
        list column whose ``null_count`` is non-zero raises
        ``NotImplementedError``.

    :param input_columns: optional list of column names that will be expected
        to be needed for in `data_dict` for the calculation done by
        `main_func`. When not given, all columns in the iterated over PySpark
        DataFrame will be used.  Names are matched exactly; a name the batch
        does not have raises :class:`KeyError` listing the batch's columns,
        since Spark's case-insensitive projection may have spelled it
        differently, and a name the batch carries more than once, as an
        unaliased join produces, raises :class:`ValueError`.  The names are
        read once, when the function is made, so a one-shot iterable such as
        a generator serves as well as a list.
    :param broadcasts: optional dictionary of broadcast values
    :param output_schema: optional :class:`pyarrow.Schema` for the batch that is
        yielded.  When given, the dict returned by ``main_func`` is bound to it
        BY NAME, so insertion order stops deciding, and every column is built
        with its declared type rather than inferred and cast: a string column
        declared ``large_string`` is built as one, a list of dicts declared
        ``struct`` is built field by field, and a list of dicts or of pairs
        declared ``map`` becomes a map, which no inferred type can be cast to.
        A name the schema declares but the dict omits raises :class:`KeyError`,
        a key the schema does not name raises :class:`ValueError`, and a dict
        key that no declared struct field has raises :class:`ValueError` too,
        since Arrow matches struct fields by exact name and would otherwise
        fill the column with nulls.

        What the declared type refuses is what ``pa.array`` refuses, and that
        depends on the shape the column arrives in.  For an ndarray of a
        numeric or datetime dtype, or a :class:`pyarrow.Array`, an integer
        out of the declared type's range, a float with a fraction into an
        integer type and a timestamp unit change that drops digits all raise
        :class:`pyarrow.ArrowInvalid`.  A Python list, and any other sequence
        of Python objects, an object-dtype ndarray included, goes through
        ``pa.array``'s sequence converter instead: an integer out of the
        declared type's range still raises :class:`pyarrow.ArrowInvalid` and
        one beyond int64 altogether raises :class:`OverflowError`, but a
        float's fraction and a timestamp's extra digits are dropped silently.
        So the lossy conversions that pass without a word are a timestamp
        into ``date32`` or ``date64``, which floors to the day, ``float64``
        into ``float32``, which overflows to ``inf``, and, from a list alone,
        a fraction into an integer type and a timestamp unit change that
        drops digits.

        Left as ``None`` the batch is built from the dict alone: insertion
        order decides, and every type is inferred from the value, so a unicode
        or bytes array comes back ``string`` or ``binary`` whatever type went
        in, a ``datetime64`` array comes back a naive ``timestamp`` of its
        unit, with a multiplier such as ``datetime64[5s]`` folded in and an
        hour or minute unit taken to seconds; a day, week, month or year unit
        comes back ``date32``, a unit finer than a nanosecond is refused, and
        an object array holding only ``None`` comes back ``null``.  The
        first batch's inferred schema is held for the partition, and a later
        batch whose inferred types differ, an all-``None`` list beside one
        holding values, or ints beside floats, is refused naming the column,
        since Spark's writer would refuse it naming nothing; declare
        ``output_schema`` where a batch may be empty or all null.
    """
    broadcasts = broadcasts if broadcasts is not None else {}
    if isinstance(input_columns, str):
        # A str is iterable, so "value" was read as the columns v, a, l, u, e.
        raise TypeError(
            f"input_columns must be a list of column names, not the string {input_columns!r}"
        )
    # dict.fromkeys keeps first-seen order. Naming a column twice produces the
    # same arrays twice, so it stays harmless. Read once, here: read inside the
    # batch loop, a generator, map() or filter() handed in was used up by the
    # first batch, and every later batch then saw no columns at all.
    named = None if input_columns is None else list(dict.fromkeys(input_columns))
    if output_schema is not None and not isinstance(output_schema, pa.Schema):
        # A PySpark StructType is the schema mapInArrow itself takes, and it
        # carries .names too, so one handed here got as far as the first batch
        # and died on a field's missing .type.
        raise TypeError(
            f"output_schema must be a pyarrow.Schema, not a {type(output_schema).__name__}"
        )
    if output_schema is not None:
        repeated = _repeated_names(list(output_schema))
        if repeated:
            # A dict holds one value per name, so every copy was filled from it
            # and Spark died in the JVM naming neither the column nor the copy.
            raise ValueError(
                f"output_schema names {repeated} more than once; the dict main_func returns holds one value "
                f"per name, so alias one of them"
            )

    def _(iterator):
        inferred = None
        for batch in iterator:
            if not isinstance(batch, pa.RecordBatch):
                # Handed a RecordBatch or a Table instead of an iterator of
                # them, the loop walked the columns and died on an attribute
                # of the first one; mapInPandas hands over pandas frames.
                raise TypeError(
                    f"pass an iterator of pyarrow.RecordBatch, as mapInArrow does, such as [batch] or "
                    f"table.to_batches(), not one yielding a {type(batch).__name__}"
                )
            data_dict: dict[str, np.ndarray | dict[str, np.ndarray]] = {}
            bitmap_dict: dict[str, np.ndarray | None | dict[str, np.ndarray | None]] = {}
            names = batch.schema.names
            input_columns_ = named if named is not None else list(dict.fromkeys(names))
            for col in input_columns_:
                if col not in names:
                    # Spark's projection is case-insensitive and may have
                    # rewritten the name it was given; the batch's own names
                    # make that visible.
                    raise MissingKeyError(f"column {col!r} is not in this batch, whose columns are {names}")
                if names.count(col) > 1:
                    # An unaliased join produces this shape, and batch.column
                    # dies on pyarrow's own KeyError, which names no remedy.
                    raise ValueError(
                        f"column {col!r} appears {names.count(col)} times in this batch; alias one of "
                        f"them in the projection that feeds mapInArrow"
                    )
                col_pa: pa.Array = batch.column(col)
                try:
                    adapted = arrow_array_adapter(col_pa)
                except (NotImplementedError, ValueError, TypeError, pa.ArrowException) as exc:
                    raise renamed(exc, f"column {col!r}") from exc
                if len(adapted) == 3:
                    struct_bitmap, field_bitmaps, field_datas = adapted
                    # The struct-level bitmap is folded into each field rather
                    # than published on its own: one is_null call per field
                    # then sees both a null field and a row that is null as a
                    # whole, and for a list-of-struct column a bitmap of its
                    # own would suggest it covers the outer list rows, which
                    # it does not.
                    data_dict[col] = field_datas
                    bitmap_dict[col] = {
                        name: _fold_struct_validity(struct_bitmap, field_bitmap)
                        for name, field_bitmap in field_bitmaps.items()
                    }
                else:
                    bitmap_dict[col], data_dict[col] = adapted
            handed = _handed_bitmaps(data_dict, bitmap_dict)
            built = _build_batch(main_func(data_dict, bitmap_dict, broadcasts), output_schema, handed)
            if output_schema is None:
                # An all-None list beside one holding values, or ints beside
                # floats, inferred a second schema, and Spark's writer refused
                # it naming nothing.
                if inferred is None:
                    inferred = built.schema
                elif built.schema != inferred:
                    raise ValueError(_schema_drift(inferred, built.schema))
            # Nothing of this batch is held across the yield: the adapted
            # arrays, the views and the handed-out bitmaps stayed bound in the
            # frame while the consumer wrote the batch out and the next one
            # was adapted, so a string column's |U copy was live twice at the
            # peak and a handed-out bitmap outlived its batch.
            data_dict = bitmap_dict = handed = col_pa = adapted = None
            struct_bitmap = field_bitmaps = field_datas = None
            yield built
            built = None
    return _
