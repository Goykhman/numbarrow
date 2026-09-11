"""
Factory for PySpark ``mapInArrow`` UDF functions.

Bridges PySpark's Arrow-based batch processing with Numba JIT-compiled functions
by converting each :class:`pyarrow.RecordBatch` column through
:func:`~numbarrow.core.adapters.arrow_array_adapter` before passing the data
to a user-supplied computation function.
"""

import numpy as np
import pyarrow as pa

from typing import Callable

from numbarrow.core.adapters import arrow_array_adapter


def _to_arrow(output):
    """Convert one UDF output column to an Arrow array.

    A numpy fixed-width unicode or bytes array handed straight to ``pa.array``
    is read with C string semantics, so a value is cut at its first NUL:
    ``"a\x00b"`` arrives as ``"a"`` and a leading NUL empties the value
    outright. Going via ``tolist()`` hands Arrow real Python strings and
    bytes, which carry NULs, at about 18% more time on a 200k-row column. A
    trailing NUL is already gone before this point, dropped by numpy when the
    array was built, which matches the adapter refusing one on the way in.

    The type is named rather than inferred because ``pa.array([])`` infers
    ``null`` where the same column with rows infers ``string``: a UDF that
    filters a whole batch away would yield a schema its other batches do not
    share, and Spark's writer refuses the second schema it sees.
    """
    kind = getattr(getattr(output, "dtype", None), "kind", None)
    if kind == "U":
        return pa.array(output.tolist(), type=pa.string())
    if kind == "S":
        return pa.array(output.tolist(), type=pa.binary())
    return pa.array(output)


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
        ``main_func(data_dict, bitmap_dict, broadcasts)``, returning a
        ``dict[str, np.ndarray]`` that is used to create a PyArrow
        ``RecordBatch``.

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
        DataFrame will be used.
    :param broadcasts: optional dictionary of broadcast values
    :param output_schema: optional :class:`pyarrow.Schema` for the batch that is
        yielded.  When given, the dict returned by ``main_func`` is bound to it
        BY NAME, so insertion order stops deciding: a column supplied under the
        wrong key is no longer able to land in another column's position.  A
        name the schema declares but the dict omits raises :class:`KeyError`,
        and a value that cannot be converted to the declared type without loss
        raises :class:`pyarrow.ArrowInvalid`.  A key the schema does not name is
        dropped silently, which is the one mismatch this does not catch.  Left
        as ``None`` the batch is built from the dict alone and insertion order
        decides, which is the behaviour described above.
    """
    broadcasts = broadcasts if broadcasts is not None else {}

    def _(iterator):
        for batch in iterator:
            data_dict: dict[str, np.ndarray | dict[str, np.ndarray]] = {}
            bitmap_dict: dict[str, np.ndarray | None | dict[str, np.ndarray | None]] = {}
            requested = input_columns if input_columns is not None else batch.schema.names
            # dict.fromkeys keeps first-seen order. Naming a column twice
            # produces the same arrays twice, so it stays harmless.
            input_columns_ = list(dict.fromkeys(requested))
            for col in input_columns_:
                col_pa: pa.Array = batch.column(col)
                adapted = arrow_array_adapter(col_pa)
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
            yield pa.RecordBatch.from_pydict(
                {col: _to_arrow(output) for col, output in outputs.items()},
                schema=output_schema
            )
    return _
