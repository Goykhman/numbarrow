"""
Factory for PySpark ``mapInArrow`` UDF functions.

Bridges PySpark's Arrow-based batch processing with Numba JIT-compiled functions
by converting each :class:`pyarrow.RecordBatch` column through
:func:`~numbarrow.core.adapters.arrow_array_adapter` before passing the data
to a user-supplied computation function.
"""

import numpy as np
import pyarrow as pa

from collections.abc import Mapping
from typing import Callable

from numbarrow.core.adapters import arrow_array_adapter
from numbarrow.utils.arrow_array_utils import renamed, type_repr


def _struct_fields(struct_type):
    return [struct_type[i] for i in range(struct_type.num_fields)]


def _check_struct_keys(rows, struct_type):
    """Refuse a dict key that no declared field has.

    Arrow matches struct fields by exact name and fills a missing one with
    null, so a list of dicts keyed ``Amount`` against a field called
    ``amount`` builds a whole column of nulls under an identical schema,
    without a word. The same typo on a top-level key raises; this makes the
    nested one raise too.
    """
    declared = {field.name for field in _struct_fields(struct_type)}
    seen = set()
    for row in rows:
        if isinstance(row, Mapping):
            seen.update(row)
    unexpected_keys = sorted(str(key) for key in seen - declared)
    if unexpected_keys:
        raise ValueError(
            f"declared {type_repr(struct_type)} but the dicts carry keys {unexpected_keys} "
            f"that no declared field has; Arrow matches struct fields by exact name and "
            f"fills a missing one with null"
        )


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
        except (pa.ArrowException, TypeError, ValueError) as exc:
            raise renamed(exc, f"field {field.name!r}") from exc
    return pa.StructArray.from_arrays(children, fields=fields)


def _convert(value, arrow_type):
    """Build one Arrow array from a UDF output value.

    With a declared type the array is built as that type from the start
    rather than inferred and cast afterwards: a string column declared
    ``large_string`` is built as one, a list of dicts declared ``struct`` is
    built field by field and its keys are checked, and a list of dicts or of
    pairs declared ``map`` becomes a map, which no inferred struct can be cast
    to.

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
        return value.cast(arrow_type)
    if isinstance(value, np.ndarray):
        if value.dtype.names is not None:
            return _record_to_struct(value, arrow_type)
        kind = value.dtype.kind
        if kind == "U":
            return pa.array(value.tolist(), type=arrow_type or pa.string())
        if kind == "S":
            return pa.array(value.tolist(), type=arrow_type or pa.binary())
        return pa.array(value, type=arrow_type)
    if arrow_type is not None and pa.types.is_struct(arrow_type) and isinstance(value, (list, tuple)):
        _check_struct_keys(value, arrow_type)
    return pa.array(value, type=arrow_type)


def _to_arrow(value, name, arrow_type=None):
    """Convert one UDF output column to an Arrow array, naming the column on any failure.

    ``pa.array`` iterates a Mapping, so a dict of arrays returned under one
    key silently became a string column of the dict's keys, with a different
    row count and nothing raised; it is refused outright. Every other failure
    on the output side named no column at all.
    """
    if isinstance(value, Mapping):
        raise TypeError(
            f"output column {name!r} is a {type(value).__name__}, which pa.array would read as "
            f"its keys; return an ndarray, a list or a pyarrow Array per column, and for a "
            f"struct column a list of dicts or a record array"
        )
    try:
        return _convert(value, arrow_type)
    except (pa.ArrowException, TypeError, ValueError) as exc:
        raise renamed(exc, f"output column {name!r}") from exc


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
        Three shapes carry a null out: a list holding ``None``, a
        :class:`pyarrow.Array`, and a numpy masked array.

        Spark binds the columns of that batch to the declared output schema by
        POSITION, not by name, and compares only their types.  The dict's
        insertion order is therefore what decides, and returning two same-typed
        columns in the other order swaps their values with no error.  Build the
        returned dict in the order the output schema declares, or pass
        ``output_schema`` and let Arrow bind it by name instead.

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
        differently.
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

        What the declared type refuses is what ``pa.array`` refuses: an
        integer out of range, a float with a fraction into an integer type
        and a timestamp unit change that drops digits all raise
        :class:`pyarrow.ArrowInvalid`.  Not every lossy conversion is refused:
        a timestamp into ``date32`` or ``date64`` floors to the day, and
        ``float64`` into ``float32`` overflows to ``inf``, both silently.

        Left as ``None`` the batch is built from the dict alone: insertion
        order decides, and every type is inferred from the value, so a unicode
        or bytes array comes back ``string`` or ``binary`` whatever type went
        in, a ``datetime64`` array comes back a naive ``timestamp`` of its
        unit, and an object array holding only ``None`` comes back ``null``.
    """
    broadcasts = broadcasts if broadcasts is not None else {}
    if isinstance(input_columns, str):
        # A str is iterable, so "value" was read as the columns v, a, l, u, e.
        raise TypeError(
            f"input_columns must be a list of column names, not the string {input_columns!r}"
        )

    def _(iterator):
        for batch in iterator:
            data_dict: dict[str, np.ndarray | dict[str, np.ndarray]] = {}
            bitmap_dict: dict[str, np.ndarray | None | dict[str, np.ndarray | None]] = {}
            requested = input_columns if input_columns is not None else batch.schema.names
            # dict.fromkeys keeps first-seen order. Naming a column twice
            # produces the same arrays twice, so it stays harmless.
            input_columns_ = list(dict.fromkeys(requested))
            for col in input_columns_:
                if col not in batch.schema.names:
                    # Spark's projection is case-insensitive and may have
                    # rewritten the name it was given; the batch's own names
                    # make that visible.
                    raise KeyError(
                        f"column {col!r} is not in this batch, whose columns are {batch.schema.names}"
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
            outputs = main_func(data_dict, bitmap_dict, broadcasts)
            if not isinstance(outputs, Mapping):
                raise TypeError(
                    f"main_func must return a dict of column name to array, not "
                    f"{type(outputs).__name__}"
                )
            if output_schema is None:
                yield pa.RecordBatch.from_pydict(
                    {name: _to_arrow(value, name) for name, value in outputs.items()}
                )
                continue
            extra = [name for name in outputs if name not in output_schema.names]
            if extra:
                raise ValueError(
                    f"main_func returned columns {extra} that output_schema does not name; "
                    f"it names {output_schema.names}"
                )
            arrays = []
            for field in output_schema:
                if field.name not in outputs:
                    raise KeyError(
                        f"output_schema names column {field.name!r}, which main_func did not "
                        f"return; it returned {list(outputs)}"
                    )
                arrays.append(_to_arrow(outputs[field.name], field.name, field.type))
            yield pa.RecordBatch.from_arrays(arrays, schema=output_schema)
    return _
