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


def _struct_fields(struct_type):
    return [struct_type[i] for i in range(struct_type.num_fields)]


def _is_list_like(arrow_type):
    return pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type) or pa.types.is_fixed_size_list(arrow_type)


def _carries_keys(arrow_type):
    """Whether a value of this type is built from dicts somewhere inside it."""
    if pa.types.is_struct(arrow_type):
        return True
    if _is_list_like(arrow_type):
        return _carries_keys(arrow_type.value_type)
    if pa.types.is_map(arrow_type):
        return _carries_keys(arrow_type.key_type) or _carries_keys(arrow_type.item_type)
    return False


def _unexpected_fields(source_type, declared_type):
    """Field names the source type carries, at any depth, that the declared type does not."""
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
    it can and names the column when it cannot.
    """
    return [row for row in rows if hasattr(row, "__iter__")]


def _check_keys(rows, arrow_type):
    """Refuse, at any depth, a dict key that no declared struct field has.

    Arrow matches struct fields by exact name and fills a missing one with
    null, so a list of dicts keyed ``Amount`` against a field called
    ``amount`` builds a whole column of nulls under an identical schema,
    without a word. The same typo on a top-level key raises; this makes the
    nested one raise too, however deep the struct sits inside a list, a map or
    another struct.
    """
    if pa.types.is_struct(arrow_type):
        fields = {field.name: field.type for field in _struct_fields(arrow_type)}
        dicts = [row for row in rows if isinstance(row, Mapping)]
        seen = set()
        for row in dicts:
            seen.update(row)
        unexpected_keys = sorted(str(key) for key in seen - set(fields))
        if unexpected_keys:
            raise ValueError(
                f"declared {type_repr(arrow_type)} but the dicts carry keys {unexpected_keys} "
                f"that no declared field has; Arrow matches struct fields by exact name and "
                f"fills a missing one with null"
            )
        for name, child_type in fields.items():
            if _carries_keys(child_type):
                _check_keys([row[name] for row in dicts if name in row], child_type)
    elif _is_list_like(arrow_type):
        _check_keys([item for row in _iterable_rows(rows) for item in row], arrow_type.value_type)
    elif pa.types.is_map(arrow_type):
        keys, items = _map_entries(rows)
        if _carries_keys(arrow_type.key_type):
            _check_keys(keys, arrow_type.key_type)
        if _carries_keys(arrow_type.item_type):
            _check_keys(items, arrow_type.item_type)


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
                if isinstance(pair, (tuple, list)) and len(pair) == 2:
                    keys.append(pair[0])
                    items.append(pair[1])
    return keys, items


def _record_to_struct(value, arrow_type):
    """A numpy record array as a struct column, one child per field.

    A record array is what an ``@njit`` function returns for a numba record
    type, and the one ndarray shape that means struct, but ``pa.array``
    refuses it with "Unsupported numpy type". Each field goes through the
    same conversion as a column of its own, so a unicode field keeps its NULs
    and a declared child type is honoured.
    """
    names = list(value.dtype.names)
    if arrow_type is None:
        children = [_convert(value[name], None) for name in names]
        return pa.StructArray.from_arrays(children, names=names)
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
        try:
            children.append(_convert(value[field.name], field.type))
        except (pa.ArrowException, TypeError, ValueError, OverflowError) as exc:
            raise renamed(exc, f"field {field.name!r}") from exc
    return pa.StructArray.from_arrays(children, fields=fields)


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
    trailing NUL is already gone before this point, dropped by numpy when the
    array was built, which matches the adapter refusing one on the way in.

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
    if arrow_type is not None and _carries_keys(arrow_type):
        if not hasattr(value, "__len__"):
            # A generator would be consumed by the check, so it is read once.
            value = list(value)
        _check_keys(value, arrow_type)
    return pa.array(value, type=arrow_type)


def _ndarray_to_arrow(value, arrow_type):
    """An ndarray of a non-object dtype as an Arrow array; see ``_convert``."""
    if value.dtype.names is not None:
        return _record_to_struct(value, arrow_type)
    kind = value.dtype.kind
    if kind == "U":
        return pa.array(value.tolist(), type=arrow_type or pa.string())
    if kind == "S":
        return pa.array(value.tolist(), type=arrow_type or pa.binary())
    return pa.array(value, type=arrow_type)


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
    string column with no nulls of its own takes the bitmap as its validity
    buffer and keeps its data buffer, so neither side is copied when the
    bitmap is contiguous, which every bitmap ``bitmap_dict`` hands out is.
    Any other column, one that already carries nulls, a sliced Arrow array or
    a nested type, is masked through ``if_else``, which keeps the nulls it
    had. The bitmap carries no row count of its own: the length check here is
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
    row count and nothing raised; it is refused outright. Every other failure
    on the output side named no column at all.
    """
    value, bitmap = _split_pair(value)
    if isinstance(value, Mapping):
        raise TypeError(
            f"output column {name!r} is a {type(value).__name__}, which pa.array would read as "
            f"its keys; return an ndarray, a list or a pyarrow Array per column, and for a "
            f"struct column a list of dicts or a record array"
        )
    try:
        array = _convert(value, arrow_type)
        if bitmap is None:
            return array
        covers = handed.get(id(bitmap))
        if covers is not None and len(array) != covers:
            raise ValueError(
                f"the bitmap is one this batch handed out for {covers} rows, but the column has "
                f"{len(array)} rows; a resized column needs a bitmap of its own"
            )
        return _with_validity(array, bitmap)
    except (pa.ArrowException, TypeError, ValueError, OverflowError) as exc:
        raise renamed(exc, f"output column {name!r}") from exc


def _handed_bitmaps(data_dict, bitmap_dict):
    """Each bitmap the batch hands out, by id, with the count of rows it covers.

    A column's bitmap covers the batch's rows, a struct field's covers the
    field's own elements, and for a list of structs those are the flattened
    elements rather than the outer rows. The count therefore comes from the
    data handed out beside the bitmap, never from the batch.
    """
    handed = {}
    for name, bitmaps in bitmap_dict.items():
        datas = data_dict[name]
        leaves = bitmaps.items() if isinstance(bitmaps, dict) else [(None, bitmaps)]
        for field, bitmap in leaves:
            if bitmap is not None:
                handed[id(bitmap)] = len(datas if field is None else datas[field])
    return handed


def _build_batch(outputs, output_schema, handed=MappingProxyType({})):
    """The RecordBatch a UDF's result becomes, bound to ``output_schema`` when there is one.

    ``handed`` maps the id of each bitmap the batch handed the UDF to the row
    count that bitmap covers; see ``_to_arrow``.
    """
    if not isinstance(outputs, Mapping):
        raise TypeError(
            f"main_func must return a dict of column name to array, not "
            f"{type(outputs).__name__}"
        )
    if output_schema is None:
        names = list(outputs)
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
        Four shapes carry a null out: a list holding ``None``, a
        :class:`pyarrow.Array`, a numpy masked array, and a
        :class:`Nullable`, ``Nullable(data, bitmap)``, whose ``bitmap`` is a
        packed uint8 validity bitmap in the layout ``bitmap_dict`` hands out,
        ``(rows + 7) // 8`` bytes with a set bit for a valid row, or ``None``.
        A bare array carries no nulls out: every null the UDF received comes
        back as whatever sat under it.  Passing the input's validity through
        is ``{"out": Nullable(result, bitmap_dict["value"])}``, or
        ``bitmap_dict["column"]["field"]`` for a struct field, and a UDF
        that decides its own nulls hands back a bitmap of that layout, which
        is the one :func:`~numbarrow.core.is_null.is_null` reads.  A bitmap
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
        unaliased join produces, raises :class:`ValueError`.
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
        unit, except a day-unit one, which comes back ``date32``, and an
        object array holding only ``None`` comes back ``null``.
    """
    broadcasts = broadcasts if broadcasts is not None else {}
    if isinstance(input_columns, str):
        # A str is iterable, so "value" was read as the columns v, a, l, u, e.
        raise TypeError(
            f"input_columns must be a list of column names, not the string {input_columns!r}"
        )
    if output_schema is not None and not isinstance(output_schema, pa.Schema):
        # A PySpark StructType is the schema mapInArrow itself takes, and it
        # carries .names too, so one handed here got as far as the first batch
        # and died on a field's missing .type.
        raise TypeError(
            f"output_schema must be a pyarrow.Schema, not a {type(output_schema).__name__}"
        )

    def _(iterator):
        for batch in iterator:
            data_dict: dict[str, np.ndarray | dict[str, np.ndarray]] = {}
            bitmap_dict: dict[str, np.ndarray | None | dict[str, np.ndarray | None]] = {}
            requested = input_columns if input_columns is not None else batch.schema.names
            # dict.fromkeys keeps first-seen order. Naming a column twice
            # produces the same arrays twice, so it stays harmless.
            input_columns_ = list(dict.fromkeys(requested))
            names = batch.schema.names
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
            yield _build_batch(main_func(data_dict, bitmap_dict, broadcasts), output_schema, handed)
    return _
