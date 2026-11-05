"""A factory of functions for PySpark's ``mapInArrow`` that compute on each batch's columns as numpy arrays.

:func:`make_mapinarrow_func` wraps a function of the caller's, ``main_func``, into the function that
``DataFrame.mapInArrow`` calls on each partition. For each :class:`pyarrow.RecordBatch`, the wrapper adapts the
columns with :func:`~numbarrow.core.adapters.arrow_array_adapter` and calls ``main_func`` with the arrays and their
validity bitmaps. ``main_func`` typically runs numba-compiled code on them and returns a dict of output columns, from
which the wrapper builds the output batch. :class:`Nullable` pairs an output column with a validity bitmap, so that the
column carries nulls out. The private functions convert each output column to an Arrow array. They refuse several kinds
of value that pyarrow would misread without an error, with a message that names the column.
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
    """An output column paired with a validity bitmap, so that the column carries nulls out of ``main_func``.

    ``data`` is any value an output column may be: an ndarray, a list, a :class:`pyarrow.Array` or a numpy record array.
    ``bitmap`` is a packed uint8 bitmap in the layout of the input bitmaps that ``main_func`` receives:
    ``(rows + 7) // 8`` bytes, one bit per row, least significant bit first, and a set bit for a valid row. A row is
    null in the output if ``data`` holds a null there or its bit is clear. ``bitmap`` may also be ``None``, the bitmap
    that ``main_func`` receives for an input column without a validity buffer. The column then carries only the nulls
    of ``data``. A plain ndarray returned without ``Nullable`` carries no nulls out, except a ``NaT``, which pyarrow
    reads as null.
    """

    data: object
    bitmap: np.ndarray | None


# The largest number of undeclared struct keys that the error of _check_keys lists. The keys come from the data, not
# from the schema: a UDF that keys its dicts by a row value has one key per row. Listing every key of a 100,000-row
# batch would put about a megabyte into the message, and into every log that records it.
KEYS_SHOWN = 10


def _struct_fields(struct_type):
    return [struct_type[i] for i in range(struct_type.num_fields)]


def _storage(arrow_type):
    """Return the storage type under any extension types of ``arrow_type``, the type a cast and the key check read."""
    while isinstance(arrow_type, pa.BaseExtensionType):
        arrow_type = arrow_type.storage_type
    return arrow_type


def _is_list_view(arrow_type):
    # pyarrow 16 added the list-view layouts and their predicates. With an older pyarrow, no type is a view.
    is_view = getattr(pa.types, "is_list_view", None)
    is_large_view = getattr(pa.types, "is_large_list_view", None)
    return bool(is_view and is_view(arrow_type)) or bool(is_large_view and is_large_view(arrow_type))


def _is_list_like(arrow_type):
    return (pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type)
            or pa.types.is_fixed_size_list(arrow_type) or _is_list_view(arrow_type))


def _carries_keys(arrow_type):
    """Return whether ``arrow_type`` has a struct at any depth, so that its values may hold dicts to check."""
    arrow_type = _storage(arrow_type)
    if pa.types.is_struct(arrow_type):
        return True
    if _is_list_like(arrow_type):
        return _carries_keys(arrow_type.value_type)
    if pa.types.is_map(arrow_type):
        return _carries_keys(arrow_type.key_type) or _carries_keys(arrow_type.item_type)
    return False


def _unexpected_fields(source_type, declared_type):
    """Return the field names that ``source_type`` has at any depth and ``declared_type`` lacks.

    The two types are paired through their layouts, because from pyarrow 19 on, a cast fills a declared struct field
    that the source lacks with nulls, behind every wrapping. An extension type is paired through its storage type. A
    map is paired with a declared list of key and value structs through its entries struct, and a list of structs with
    a declared map the same way.
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
        # A dictionary type is a layout of its value type. The cast converts the dictionary's values, so the value
        # structs are paired as plain structs are.
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
    """Return the rows that may hold dicts, for the key check to look inside.

    A row is kept if ``iter`` accepts it and it is not a str, a bytes, an ndarray of a non-object dtype or a pyarrow
    scalar. ``pa.array`` converts or refuses every row that is skipped, and ``_to_arrow`` names the column if it fails.
    """
    kept = []
    for row in rows:
        # A str, a bytes and an ndarray of a non-object dtype iterate over scalars, which hold no keys. pa.array
        # refuses such a row at its first item, while spreading a long one into a list here would cost time and
        # memory in proportion to its length. pa.array checks a pyarrow scalar against the declared type itself. A
        # MapScalar is a Mapping from pyarrow 21 on, but its values is an array, not a method, so _map_entries cannot
        # read one.
        if isinstance(row, (str, bytes, pa.Scalar)) or (isinstance(row, np.ndarray) and row.dtype.kind != "O"):
            continue
        # A missing row is None in a list, NaN in a pandas Series of object dtype and pd.NA in a nullable one.
        # pa.array turns each of them into a null, and none of them is iterable. iter() decides rather than an
        # __iter__ attribute, because a sequence that defines __len__ and __getitem__ but no __iter__ still iterates,
        # and pa.array reads it item by item.
        try:
            iter(row)
        except TypeError:
            continue
        kept.append(row)
    return kept


def _refuse_permuted_fields(tuples, names, arrow_type, where):
    """Raise ``ValueError`` for a namedtuple or pyspark Row whose field names differ from the declared names or order.

    ``pa.array`` binds the items of a tuple to the struct fields by position and ignores their names. A row that names
    the declared fields in another order would swap the values of fields of the same type without an error. A row that
    names a field the declared type lacks, such as a misspelled or renamed field, would put its value into the field
    at the same position. A plain tuple has no names and is not checked. ``where`` starts the message with the path
    to the struct.
    """
    for row in tuples:
        given = getattr(row, "_fields", None) or getattr(row, "__fields__", None)
        if given is None or list(given) == names:
            continue
        how = "in another order" if set(given) == set(names) else "which are not the declared fields"
        raise ValueError(
            f"{where}declared {type_repr(arrow_type)} but a row names its fields {list(given)}, {how}; a "
            f"tuple's fields bind by position, so name them as declared, in the declared order, or return dicts"
        )


def _check_keys(rows, arrow_type, where=""):
    """Raise ``ValueError`` for a dict key that no declared struct field has, at any depth, naming the path to it.

    Arrow matches struct fields by exact name and fills a missing field with nulls. A list of dicts keyed ``Amount``
    under a field named ``amount`` would therefore become a column of nulls, with the declared schema and no error.
    ``_build_batch`` already refuses a misspelled output column name. This check refuses the same mistake in a struct
    at any depth inside lists, maps and other structs. A tuple, a namedtuple or a pyspark Row binds to the fields by
    position, so its items are checked against the fields in declared order. ``_refuse_permuted_fields`` refuses a
    namedtuple or Row that names other fields, or the declared fields in another order. ``where`` is the path so far,
    which starts the message.
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
        _refuse_permuted_fields(tuples, names, arrow_type, where)
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
    """Return the keys and the values of map rows given as dicts or as lists of pairs.

    A pair is a tuple or a list of two items, or a dict with the keys ``key`` and ``value``. Any other item is skipped
    and left to ``pa.array``, which refuses it, and ``_to_arrow`` then names the column. Indexing the item here could
    raise an ``IndexError``, which ``_to_arrow`` does not catch, so the error would not name the column.
    """
    keys, items = [], []
    for row in _iterable_rows(rows):
        if isinstance(row, Mapping):
            keys.extend(row)
            items.extend(row.values())
        else:
            for pair in row:
                if isinstance(pair, Mapping) and set(pair) == {"key", "value"}:
                    # Spark's map_entries produces entries of this shape, and pa.array reads one as a key and a value.
                    keys.append(pair["key"])
                    items.append(pair["value"])
                elif isinstance(pair, (tuple, list)) and len(pair) == 2:
                    keys.append(pair[0])
                    items.append(pair[1])
    return keys, items


def _struct_column(children, rows, **layout):
    """Return a struct array of ``children``, or of ``rows`` empty structs if no child gives the array a length.

    ``layout`` holds the ``names`` or the ``fields`` that ``pa.StructArray.from_arrays`` takes.
    """
    if not children:
        return pa.StructArray.from_buffers(pa.struct([]), rows, [None], children=[])
    return pa.StructArray.from_arrays(children, **layout)


def _record_field(value, name, arrow_type):
    """Convert the field ``name`` of a record array to an Arrow array, and name the field in any failure."""
    try:
        return _convert(value[name], arrow_type)
    except (pa.ArrowException, TypeError, ValueError, OverflowError, KeyError) as exc:
        raise renamed(exc, f"field {name!r}") from exc


def _record_to_struct(value, arrow_type):
    """Convert a numpy record array to a struct column, with one child per field.

    A record array is the one ndarray that means a struct, and an ``@njit`` function returns one for a numba record
    type. ``pa.array`` refuses it with "Unsupported numpy type". Each field is converted as a column of its own, so a
    unicode field keeps its leading and interior NULs, a declared child type is applied, and a failure names the field
    with or without a declared type. A declared field that the record array lacks becomes a column of nulls. A field
    that the declared struct lacks raises ``ValueError``, and a declared type that is not a struct raises
    ``TypeError``. A record array without fields becomes one empty struct per row, because a struct array without
    children has no length of its own.
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
    """Convert one output value of the UDF to an Arrow array, of the type ``arrow_type`` if it is not ``None``.

    A list or another sequence of Python objects is built as the declared type, not inferred and then cast. A string
    column declared ``large_string`` is built as one. A list of dicts declared as a struct is built field by field,
    after its keys are checked at every depth. A list of dicts or of pairs declared as a map becomes a map, which no
    inferred struct can be cast to. A :class:`pyarrow.Array` is cast to the declared type, but only after its field
    names at every depth are found among the declared ones. An ndarray of a non-object dtype is converted by
    ``_ndarray_to_arrow``.
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
    # The rest is an object array, a list, a tuple, a pandas Series or any other iterable of Python objects.
    if not hasattr(value, "__len__"):
        # The checks below iterate the value, which would use up a generator, so a value without a length is read
        # into a list once.
        value = list(value)
    _refuse_pandas_rows(value, arrow_type)
    if arrow_type is not None and _carries_keys(arrow_type):
        _check_keys(value, arrow_type)
    array = pa.array(value, type=arrow_type)
    if isinstance(array, pa.ChunkedArray):
        # pa.array returns a ChunkedArray for a pandas Series over a multi-chunk pyarrow array.
        # RecordBatch.from_arrays would refuse it with an error that names no column.
        array = array.combine_chunks()
    return array


def _at_arrow_unit(value, arrow_type):
    """Return ``value``, a datetime64 or timedelta64 array, at a unit that pyarrow models and without a multiplier.

    A multiplier is applied exactly: ``datetime64[5s]`` becomes ``datetime64[s]`` with five times the counts. An hour or
    minute unit becomes seconds. A datetime64 week, month or year becomes days, and a timedelta64 day or week becomes
    seconds, since Arrow has no duration in days. Each of these conversions is exact. A unit finer than a nanosecond
    raises ``TypeError``, and so does a timedelta64 month or year, which has no fixed length.

    If the declared type is not temporal, such as an integer or a string, a change of unit raises ``TypeError``,
    because it would change the counts: under ``int64``, two days would become 172800. A multiplier is still applied,
    because the counts of ``datetime64[5s]`` are not counts of seconds.
    """
    # pa.array reads numpy's base unit and ignores the multiplier. A datetime64[5s] column of five-second bins would
    # come back at one-second steps, with 2020 read as 1980. A datetime64[2D] array would also miss the day-unit branch
    # of _ndarray_to_arrow and get the misread that the branch prevents. pa.array refuses outright a unit that it does
    # not model, such as an hour or a week.
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
    if target != unit and arrow_type is not None and not pa.types.is_temporal(_storage(arrow_type)):
        raise TypeError(
            f"a {value.dtype} array under {type_repr(arrow_type)}: pyarrow has no such unit, and taking the "
            f"counts to {family}[{target}] would change them; convert the array to the values you mean, or "
            f"declare a temporal type"
        )
    if target == unit and count == 1:
        return value
    return value.astype(f"{family}[{target}]")


def _ndarray_to_arrow(value, arrow_type):
    """Convert an ndarray of a non-object dtype to an Arrow array, of the type ``arrow_type`` if it is not ``None``.

    A record array becomes a struct column. A unicode or bytes array must be one-dimensional, and without a declared
    type it becomes ``string`` or ``binary``. It keeps every NUL except a trailing one, which a bytes array keeps only
    under a fixed-size binary type. ``_at_arrow_unit`` first takes a datetime64 or timedelta64 array to a unit that
    pyarrow models.
    """
    if value.dtype.names is not None:
        return _record_to_struct(value, arrow_type)
    kind = value.dtype.kind
    if kind in ("U", "S") and value.ndim != 1:
        # The unicode and bytes paths below convert through tolist(). For a 0-d array, tolist() returns a bare str or
        # bytes, which pa.array spreads one character or byte per row, and for a 2-d array it returns nested lists.
        # pa.array refuses an array of any other dtype that is not one-dimensional, and these two are refused here in
        # the same way.
        raise TypeError(f"a {value.ndim}-dimensional {value.dtype} array; an output column is one-dimensional")
    # pa.array reads a numpy unicode or bytes array with C string semantics and cuts each value at its first NUL:
    # "a\x00b" becomes "a", and a leading NUL empties the value. tolist() gives pa.array Python str and bytes objects,
    # which keep their NULs, at the cost of one Python object per value. tolist() drops a trailing NUL, as numpy's
    # element access does, because numpy pads a fixed-width value with NULs. The adapter refuses a trailing NUL on
    # input for the same reason.
    # Without a declared type, a unicode array becomes string and a bytes array binary, rather than an inferred type.
    # pa.array infers null for an empty list, so a UDF that filters away a whole batch would yield a schema that the
    # other batches do not share, and Spark's writer refuses the second schema it sees.
    if kind == "U":
        return pa.array(value.tolist(), type=arrow_type or pa.string())
    if kind == "S":
        if arrow_type is not None and pa.types.is_fixed_size_binary(arrow_type):
            # Under a fixed-size binary type, pa.array keeps every byte of a fixed-width array, NULs included.
            # tolist() would drop a trailing NUL, so a digest that ends in 0x00 would come back a byte short and be
            # refused under its fixed width. Under a variable-width binary type, pa.array cuts at the first NUL, so
            # that case takes the tolist() route.
            return pa.array(value, type=arrow_type)
        return pa.array(value.tolist(), type=arrow_type or pa.binary())
    if kind in ("M", "m"):
        value = _at_arrow_unit(value, arrow_type)
    if value.dtype == np.dtype("datetime64[D]") and arrow_type is not None:
        # With a declared timestamp or int32 type, pa.array reads the 8-byte values of a day-unit array as the
        # 4-byte days of a date32, so every other row comes back as the epoch or as zero. Built without a type, the
        # array is a correct date32. The cast then gives day numbers under int32 and midnights under a timestamp,
        # and pyarrow's own refusals of other casts still apply.
        return pa.array(value).cast(arrow_type)
    return pa.array(value, type=arrow_type)


def _refuse_pandas_rows(value, arrow_type):
    """Raise ``TypeError`` if a row of ``value``, a column of Python objects, is a pandas Series or DataFrame.

    ``pa.array`` reads a Series row by its index labels, not by position. A sorted or filtered Series would come back
    reordered, and one whose labels are not ``0`` to ``n - 1`` would raise a bare ``KeyError``. The check scans every
    column that reaches it: a list, a tuple, an object array or any other iterable. The message suggests
    ``row.to_dict()`` under a struct type, because ``pa.array`` refuses an ndarray or a list as a struct value. Under
    any other type it suggests ``row.to_numpy()`` or ``list(row)``.
    """
    for row in value:
        if _is_pandas(row, "Series", "DataFrame"):
            under_struct = arrow_type is not None and pa.types.is_struct(_storage(arrow_type))
            remedy = "row.to_dict()" if under_struct else "row.to_numpy() or list(row)"
            raise TypeError(
                f"a row is a pandas {type(row).__name__}, which pa.array reads by its labels "
                f"rather than in order; hand it over as {remedy}"
            )


def _is_pandas(value, *names):
    """Return whether ``value`` is a pandas object of one of the class names ``names``, without importing pandas."""
    cls = type(value)
    return cls.__name__ in names and cls.__module__.split(".")[0] == "pandas"


def _split_pair(value):
    """Return ``(data, bitmap)`` for a :class:`Nullable`, and ``(value, None)`` for any other value.

    A plain tuple is not read as a pair. It goes to ``pa.array`` as a sequence of rows, so a 2-tuple is a two-row
    column, and any rule that read some tuples as pairs would misread one of the two shapes.
    """
    if isinstance(value, Nullable):
        return value.data, value.bitmap
    return value, None


def _with_validity(array, bitmap):
    """Return ``array`` with ``bitmap`` applied as validity, so that a row is null if either of the two marks it null.

    ``bitmap`` has the layout of the bitmaps in ``bitmap_dict``, which :func:`~numbarrow.core.is_null.is_null` reads:
    ``(rows + 7) // 8`` bytes of uint8, one bit per row, least significant bit first, and a set bit for a valid row.
    A bitmap that is not a one-dimensional uint8 ndarray raises ``TypeError``, and one of another byte length raises
    ``ValueError``. The bitmap holds no row count, so this check cannot tell apart row counts within one byte.
    ``_to_arrow`` checks an input bitmap against the row count it covers. A flat array without nulls takes the bitmap
    as its validity buffer, and nothing is copied if the bitmap is contiguous.
    """
    # The type is checked before any attribute is read, because the AttributeError that a read would raise is not
    # among the classes that _to_arrow catches. It would escape without naming the column or the layout.
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
        # An extension type reports no fields and no dictionary, whatever its storage type, so the flat test below
        # cannot read its layout. A dictionary storage would take the from_buffers path, which aborts the interpreter
        # on that layout. An array with nulls or an offset would reach if_else, which has no kernel for extension
        # types. The bitmap is therefore applied to the storage array, and the result is wrapped in the extension
        # type again.
        return pa.ExtensionArray.from_storage(array.type, _with_validity(array.storage, bitmap))
    # A flat array has no nulls, no offset, no child arrays and no dictionary, and is not of the null type. It takes
    # the bitmap as its validity buffer and keeps its data buffers. Nothing is copied if the bitmap is contiguous, as
    # every bitmap in bitmap_dict is. Any other array is masked with if_else, which keeps the nulls that the array
    # already has.
    flat = (array.null_count == 0 and array.offset == 0 and array.type.num_fields == 0
            and not pa.types.is_dictionary(array.type) and not pa.types.is_null(array.type))
    if flat:
        buffers = [pa.py_buffer(np.ascontiguousarray(bitmap))] + list(array.buffers()[1:])
        return pa.Array.from_buffers(array.type, rows, buffers)
    valid = pa.array(np.unpackbits(bitmap, bitorder="little")[:rows].astype(bool))
    return pc.if_else(valid, array, pa.scalar(None, type=array.type))


def _to_arrow(value, name, arrow_type=None, handed=MappingProxyType({})):
    """Convert the output column ``name`` to an Arrow array, of the type ``arrow_type`` if it is not ``None``.

    Any failure is raised again with the column's name. A :class:`Nullable` is converted from its data, and its bitmap
    is then applied as validity. ``handed`` maps the id of each input bitmap, one that ``bitmap_dict`` held for this
    batch, to the bitmap and the row count it covers. An input bitmap on a column of any other length raises
    ``ValueError``, because a packed bitmap cannot tell apart row counts within one byte. A Nullable inside a
    Nullable, a Mapping, a str or bytes, and a pandas DataFrame raise ``TypeError`` before they reach ``pa.array``,
    which would misread each of them.
    """
    value, bitmap = _split_pair(value)
    if isinstance(value, Nullable):
        # pa.array would read the inner Nullable as the 2-tuple it is: a two-row column of the data and the bitmap.
        raise TypeError(
            f"output column {name!r} is a Nullable inside a Nullable, which pa.array would read as "
            f"a two-row column of its data and its bitmap; wrap the data once"
        )
    # pa.array iterates a Mapping over its keys. A dict returned as a column would become a string column of its
    # keys, of another length, without an error.
    if isinstance(value, Mapping):
        raise TypeError(
            f"output column {name!r} is a {type(value).__name__}, which pa.array would read as "
            f"its keys; return an ndarray, a list or a pyarrow Array per column, and for a "
            f"struct column a list of dicts or a record array"
        )
    # pa.array spreads a str one character per row, and a bytes one byte per row: {"country": "US"} over a two-row
    # batch would give the rows "U" and "S".
    if isinstance(value, (str, bytes)):
        raise TypeError(
            f"output column {name!r} is a {type(value).__name__}, which pa.array would spread one "
            f"character per row; a constant column is np.full(rows, value)"
        )
    if _is_pandas(value, "DataFrame"):
        # pa.array reads a DataFrame by its column labels, as rows. A frame with one column named x raises a bare
        # KeyError(0), and a frame with integer labels can come back transposed.
        raise TypeError(
            f"output column {name!r} is a DataFrame, which pa.array reads by its column labels; "
            f"return one Series or ndarray per column"
        )
    try:
        array = _convert(value, arrow_type)
        if bitmap is None:
            return array
        # An input bitmap fits only a column of the row count it covers: the batch's rows for a column's bitmap or a
        # struct field's, and the flattened elements for a field of a list of structs. _with_validity can check only
        # the byte count.
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
    """Return a dict that maps the id of each bitmap in ``bitmap_dict`` to the bitmap and the row count it covers.

    A column's bitmap covers the batch's rows. A struct field's bitmap covers the field's elements, which for a list of
    structs are the flattened elements rather than the list rows. The count is therefore the length of the data beside
    the bitmap in ``data_dict``, not the length of the batch. The dict holds a reference to each bitmap, which keeps the
    bitmap alive while the batch is processed, and the caller drops the dict with the batch. Python reuses the id of a
    freed object, and a UDF that deletes a bitmap from ``bitmap_dict`` would otherwise free it. A new bitmap of the
    UDF's own could then get that id and be refused as an input bitmap.
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
    """Build the output RecordBatch from ``outputs``, the dict that ``main_func`` returned.

    With ``output_schema``, the columns are taken by the schema's names, in its order, and built as its types. A column
    that the schema names and ``outputs`` lacks raises ``KeyError``, and a key that the schema does not name raises
    ``ValueError``. Without a schema, every key must be a str, and the columns keep the dict's order and inferred types.
    A value that is not a Mapping raises ``TypeError``, and columns of different lengths raise ``ValueError``.
    ``handed`` maps the id of each input bitmap to the bitmap and the row count it covers, for ``_to_arrow``.
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
        # pyarrow's own error gives only the two lengths, such as "3 vs 2", and names neither column.
        raise ValueError(f"output columns differ in length: {lengths}")
    if output_schema is None:
        return pa.RecordBatch.from_arrays(arrays, names=names)
    return pa.RecordBatch.from_arrays(arrays, schema=output_schema)


def _repeated_names(fields):
    """Return the names that two sibling fields share, among ``fields`` and inside their types at any depth."""
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


def _refuse_repeated_names(output_schema):
    """Raise ``ValueError`` if two columns of ``output_schema``, or two sibling fields at any depth, share a name.

    The dict that ``main_func`` returns holds one value per name, so every copy would be filled from that value. Spark
    would then fail in the JVM with an error that names neither the column nor the copy.
    """
    if output_schema is None:
        return
    repeated = _repeated_names(list(output_schema))
    if repeated:
        raise ValueError(
            f"output_schema names {repeated} more than once; the dict main_func returns holds one value "
            f"per name, so alias one of them"
        )


def _refuse_empty_selection(named):
    """Raise ``ValueError`` if ``named``, the list of selected column names, is empty.

    An empty selection, such as a filter that matched nothing or a generator already used up, would pass ``main_func``
    no columns. The result would be a batch without rows or columns and no error, or a bare ``KeyError`` inside
    ``main_func``.
    """
    if named == []:
        raise ValueError(
            "input_columns names no column, so main_func would be handed none; pass None for every column, "
            "and check for a generator already used up or a filter that matched nothing"
        )


def _held_schema(output_schema, inferred, built):
    """Return the schema to hold for the rest of the partition, and raise ``ValueError`` if ``built`` departs from it.

    With ``output_schema`` nothing is held, and the result is ``None``. Otherwise the first batch's inferred schema is
    held, and a later batch with another inferred schema raises an error that names the column. Inference can differ
    between batches: an all-None list infers ``null`` where a list with values does not, and a list of ints infers
    ``int64`` where a list of floats infers ``double``. Spark's writer refuses a second schema with an error that names
    nothing.
    """
    if output_schema is not None:
        return None
    if inferred is None:
        return built.schema
    if built.schema != inferred:
        raise ValueError(_schema_drift(inferred, built.schema))
    return inferred


def _schema_drift(first, later):
    """Return the error message that says where ``later``, an inferred batch schema, differs from ``first``."""
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
    """Return a field's bitmap combined with its struct's bitmap, so that a row null as a whole is null in the field.

    Arrow need not clear a field's validity bit for a row that is null as a whole, so the field's bitmap alone can miss
    that row. A set bit means valid in both bitmaps, so the two combine with a bitwise and. Either argument may be
    ``None``, which means no nulls on that side.
    """
    if struct_bitmap is None:
        return field_bitmap
    if field_bitmap is None:
        # Each field gets its own copy, because a caller that writes into one field's bitmap would otherwise change
        # the bitmaps of the other fields too.
        return struct_bitmap.copy()
    return struct_bitmap & field_bitmap


def make_mapinarrow_func(
    main_func: Callable,
    input_columns: list[str] | None = None,
    broadcasts: dict | None = None,
    output_schema: pa.Schema | None = None
):
    """Return a function for ``DataFrame.mapInArrow`` that runs ``main_func`` on each batch's columns as numpy arrays.

    ``mapInArrow`` calls the returned function once per partition, with an iterator of :class:`pyarrow.RecordBatch`.
    For each batch, the function adapts the selected columns with :func:`~numbarrow.core.adapters.arrow_array_adapter`
    and calls ``main_func(data_dict, bitmap_dict, broadcasts)``. It builds an output batch from the dict that
    ``main_func`` returns, and yields it. If a column fails to adapt or to convert, the error names the column. An
    element of the iterator that is not a RecordBatch raises :class:`TypeError`. To call the function outside Spark,
    pass it ``[batch]`` or ``table.to_batches()``.

    Pass ``output_schema``, derived from the schema given to ``mapInArrow`` with
    ``pyspark.sql.pandas.types.to_arrow_schema``, so that the returned function takes each output column by name and
    converts it to its declared type. Without it, Spark binds the columns by their order in the returned dict, and
    pyarrow infers each type from the values.

    ``data_dict`` maps the name of each selected column to its data. A struct or list-of-struct column maps to a dict of
    its fields, ``data_dict[column][field]``, so a field never shares a namespace with another column or with the
    fields of another struct. Any other column maps to one array. ``bitmap_dict`` has the same keys and shape. Each of
    its arrays is a packed uint8 validity bitmap. A column without a validity buffer has ``None`` in its place, and so
    does a struct field if neither it nor its struct has one. Every key of ``data_dict`` is in ``bitmap_dict``, so a
    batch without nulls is indexed exactly like a batch with them.

    The bitmap of a struct field also marks the rows where the whole struct is null. One
    :func:`~numbarrow.core.is_null.is_null` call per field therefore sees both a null field and a null struct. For a
    list of structs, the field bitmaps cover the flattened struct elements, not the list rows. The adapter returns
    those elements without the list offsets, so a null list row could not be reported anywhere, and it would shift the
    mapping from elements to rows. A list column whose ``null_count`` is not zero therefore raises
    :class:`NotImplementedError`.

    ``main_func`` returns a dict that maps the name of each output column to an ndarray, a list or a
    :class:`pyarrow.Array`. A numpy record array becomes a struct column, with one child per field. A null comes out of
    a list, a tuple or an object array holding ``None``, a :class:`pyarrow.Array`, a numpy masked array, and a
    :class:`Nullable`. A plain ndarray carries no nulls out, except a ``NaT``, which pyarrow reads as null. A row that
    came in null then goes out valid, with whatever ``main_func`` computed from the value stored under the null, which
    is ``0``, ``0.0`` or ``''`` in a batch from Spark.

    ``Nullable(data, bitmap)`` takes a bitmap in the layout of ``bitmap_dict``, which
    :func:`~numbarrow.core.is_null.is_null` also reads: ``(rows + 7) // 8`` bytes of uint8, one bit per row, least
    significant bit first, and a set bit for a valid row. A result that is null exactly where one input column is null
    can pass that column's bitmap through, as in ``{"out": Nullable(result, bitmap_dict["value"])}``, or with
    ``bitmap_dict["column"]["field"]`` for a struct field. Any other result builds its own bitmap, for instance with
    ``np.packbits(valid, bitorder="little")`` from a boolean array ``valid``.

    A bitmap that is not a one-dimensional uint8 ndarray raises :class:`TypeError`, and one of another length raises
    :class:`ValueError`. Both errors name the column. A packed bitmap cannot tell apart row counts within one byte. A
    bitmap from ``bitmap_dict`` therefore raises :class:`ValueError` on a column of any row count but the one it
    covers. That count is the batch's rows for a column's bitmap or a struct field's, and the flattened elements for a
    field of a list of structs. A ``main_func`` that changes the length of a column needs a bitmap of its own.

    Spark binds the columns of the output batch to the schema given to ``mapInArrow`` by position, not by name, and it
    checks nothing about their names. It reads each column with the getter of its declared type, such as ``getLong``
    for ``LongType`` and ``TimestampType``. The column's Arrow type decides which getter works, because Spark wraps
    each Arrow type in an accessor that provides only one getter. The Arrow types whose accessors provide the same
    getter form an accessor family: int64 and timestamps share ``getLong``, and int32 and date32 share ``getInt``. If a
    column's Arrow type is in the family of its declared type, Spark reads it without any check. An int64 column
    declared as a timestamp therefore reads as a timestamp, and two columns of one family swap without an error if the
    dict is built in the other order.

    A column read through the accessor of another family fails in the JVM with
    ``java.lang.UnsupportedOperationException``, whatever its width. Both float64 under ``LongType`` and int64 under
    ``DoubleType`` fail, though each is 64 bits wide. An int32 column under ``LongType`` fails too, because int32
    provides ``getInt`` and not ``getLong``.

    The fields of a struct column are bound by position too. Without ``output_schema``, the field order of a record
    array's dtype, or the key order of the dicts, decides where each field lands: ``{"lat": .., "lon": ..}`` under
    ``struct<lon, lat>`` swaps the two fields without an error. Build the returned dict, and any record dtype, in the
    order of the declared schema, or pass ``output_schema``, and the returned function binds the columns, and the
    fields of dicts and record arrays, by name.

    :param main_func: the function to run on each batch, called as ``main_func(data_dict, bitmap_dict, broadcasts)``.
        It returns a dict of output columns, as described above.
    :param input_columns: optional names of the columns that ``main_func`` needs in ``data_dict``. ``None`` selects
        every column of the batch. A name must match exactly, case included. A name that the batch lacks raises
        :class:`KeyError` listing the batch's columns. A name that the batch holds more than once raises
        :class:`ValueError`. The returned function raises both errors on the first batch. Only a batch built outside
        Spark can repeat a name, because ``mapInArrow`` raises an ``AnalysisException`` for a DataFrame with a repeated
        column name. The names are read once, when this function is called, so a one-shot iterable such as a generator
        works as well as a list. A name given twice is passed once. A single string raises :class:`TypeError`, and an
        empty selection, such as a generator already used up, raises :class:`ValueError` rather than passing
        ``main_func`` no columns. This function raises both of these when it is called, before any batch.
    :param broadcasts: optional dict that every call of ``main_func`` receives as its third argument, such as
        parameters or lookup tables. ``None`` passes an empty dict.
    :param output_schema: optional :class:`pyarrow.Schema` of the output batches. A string column declared
        ``large_string`` is built as one, and a list of dicts declared as a struct is built field by field. A list of
        dicts or of pairs declared as a map becomes a map, which no inferred type can be cast to.

        A column that the schema names and the dict lacks raises :class:`KeyError`, and a key that the schema does not
        name raises :class:`ValueError`. A struct field that the declared type lacks raises :class:`ValueError` too,
        whether it comes as a dict key, a namedtuple field, or a field of a record array or a :class:`pyarrow.Array`.
        Arrow matches a dict's keys to struct fields by exact name, so without this check it would fill the declared
        field with nulls. A namedtuple or pyspark Row that names the declared fields in another order raises
        :class:`ValueError` as well, because pyarrow binds the items of a tuple by position.

        A schema that is not a :class:`pyarrow.Schema`, such as a PySpark ``StructType``, raises :class:`TypeError`. A
        schema that repeats a name among its columns, or among the fields of one struct, raises :class:`ValueError`.
        This function raises both of these when it is called, before any batch.

        Beyond these checks, a value is refused only if pyarrow refuses it, and pyarrow converts each shape of column
        its own way. A :class:`pyarrow.Array` is cast to the declared type, and an ndarray of a numeric or datetime
        dtype goes through ``pa.array``'s typed converter. Both raise :class:`pyarrow.ArrowInvalid` for an integer out
        of the declared type's range, a float with a fraction into an integer type, and a timestamp unit change that
        drops digits.

        A Python list, and any other sequence of Python objects such as an object-dtype ndarray, goes through
        ``pa.array``'s sequence converter. It raises :class:`pyarrow.ArrowInvalid` for an integer out of the declared
        type's range, but :class:`OverflowError` for a negative value under an unsigned type and for a value beyond the
        64-bit range. ``except pa.ArrowInvalid`` does not catch the :class:`OverflowError`. The sequence converter
        refuses a timestamp under ``time32`` or ``time64`` and a number under ``bool``.

        These conversions lose data without an error:

        - a timestamp into ``date32`` or ``date64``;
        - a float into a narrower float, which overflows to ``inf``;
        - from a datetime ndarray or a :class:`pyarrow.Array`, a timestamp into ``time32`` or ``time64``, which drops
          the date;
        - from a numeric ndarray or a :class:`pyarrow.Array`, a number into ``bool``, which is true for anything but
          zero;
        - and, from a sequence of Python objects alone, a fraction into an integer type and a timestamp unit change that
          drops digits, which the sequence converter truncates.

        A float under a ``decimal`` type is refused on both routes of ``pa.array``: the sequence converter takes an int
        or a :class:`decimal.Decimal`, and the typed converter refuses a float dtype. A float :class:`pyarrow.Array` is
        cast instead, and the cast rounds it to the declared scale.

        A ``datetime64`` or ``timedelta64`` ndarray is taken to a unit that pyarrow models before pyarrow reads it. A
        multiplier is applied, so ``datetime64[5s]`` becomes ``datetime64[s]`` with the same instants. A
        ``datetime64`` hour or minute becomes seconds, and a week, month or year becomes days. A ``timedelta64`` hour,
        minute, day or week becomes seconds. A unit finer than a nanosecond raises :class:`TypeError`, and so does a
        ``timedelta64`` month or year, which has no fixed length. If the declared type is not temporal, a change of
        unit raises :class:`TypeError` too.

        If ``output_schema`` is ``None``, the batch is built from the dict alone. The dict's order decides the column
        order, and each type is inferred from the values. A unicode array comes back ``string`` and a bytes array
        ``binary``, so a ``large_string`` input column comes back ``string``. A ``datetime64`` array comes back a naive
        ``timestamp`` of its unit after the conversion above, so ``datetime64[5s]`` comes back ``timestamp[s]`` and an
        hour or minute unit comes back in seconds. A day, week, month or year unit comes back ``date32``. Spark reads a
        timestamp column back only in microseconds, so a ``datetime64`` array of another unit fails in the JVM unless
        ``output_schema`` declares the column, which converts the unit. An object array that holds only ``None`` comes
        back ``null``. The first batch's inferred schema is held for the rest of
        the partition. A later batch whose inferred types differ, such as an all-``None`` list after a list with
        values, or ints after floats, raises :class:`ValueError` naming the column. Spark's writer would refuse that
        batch without naming anything. Declare ``output_schema`` if a batch may be empty or all null.
    """
    broadcasts = broadcasts if broadcasts is not None else {}
    if isinstance(input_columns, str):
        # A str is an iterable of its characters, so "value" would select the columns v, a, l, u and e.
        raise TypeError(
            f"input_columns must be a list of column names, not the string {input_columns!r}"
        )
    # The names are read once, here. If the batch loop read them, the first batch would use up a generator, map() or
    # filter(), and every later batch would see no columns. dict.fromkeys drops a repeated name and keeps the
    # first-seen order; a name given twice would only produce the same arrays twice.
    named = None if input_columns is None else list(dict.fromkeys(input_columns))
    _refuse_empty_selection(named)
    if output_schema is not None and not isinstance(output_schema, pa.Schema):
        # A PySpark StructType is the schema that mapInArrow itself takes. Without this check, the call below would fail
        # on a field without .type, with an AttributeError that names no fix.
        raise TypeError(
            f"output_schema must be a pyarrow.Schema, not a {type(output_schema).__name__}"
        )
    _refuse_repeated_names(output_schema)

    def _(iterator):
        inferred = None
        for batch in iterator:
            if not isinstance(batch, pa.RecordBatch):
                # A RecordBatch or a Table passed in place of an iterator yields its columns here, and mapInPandas
                # passes pandas DataFrames. Each would fail on a missing attribute with an error that names no fix.
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
                    # Spark matches column names without regard to case, so a caller can spell a column in another
                    # case than the batch does. The message lists the batch's own names.
                    raise MissingKeyError(f"column {col!r} is not in this batch, whose columns are {names}")
                if names.count(col) > 1:
                    # Only a batch built outside Spark can hold two columns of one name, because mapInArrow refuses
                    # such a DataFrame. For such a batch, batch.column would raise pyarrow's own KeyError, which names
                    # no fix.
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
                    # The adapter returns a struct or list-of-struct column as a 3-tuple. The struct's bitmap is
                    # combined into each field's bitmap rather than published on its own, so that one is_null call
                    # per field sees both a null field and a row that is null as a whole. For a list-of-struct
                    # column, a bitmap of its own would also suggest that it covers the list rows, and it does not.
                    data_dict[col] = field_datas
                    bitmap_dict[col] = {
                        name: _fold_struct_validity(struct_bitmap, field_bitmap)
                        for name, field_bitmap in field_bitmaps.items()
                    }
                else:
                    # A column's bitmap can be an array in one batch and None in the next. pyspark reads each batch
                    # with pyarrow's IPC reader, which drops the validity buffer of each column that holds no null.
                    bitmap_dict[col], data_dict[col] = adapted
            handed = _handed_bitmaps(data_dict, bitmap_dict)
            built = _build_batch(main_func(data_dict, bitmap_dict, broadcasts), output_schema, handed)
            inferred = _held_schema(output_schema, inferred, built)
            # The frame drops every reference to this batch's data before the yield. While the generator waits at
            # the yield, the consumer writes the batch out, and then the next batch is adapted. A reference kept in
            # the frame would keep the adapted arrays, the views and the input bitmaps alive through both. The |U
            # copies of a string column from two batches would then be alive at once, and an input bitmap would outlive
            # its batch.
            data_dict = bitmap_dict = handed = col_pa = adapted = None
            struct_bitmap = field_bitmaps = field_datas = None
            yield built
            built = None
    return _
