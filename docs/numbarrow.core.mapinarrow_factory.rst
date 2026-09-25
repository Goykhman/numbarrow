numbarrow.core.mapinarrow_factory
=================================

Overview
++++++++

Factory for PySpark ``mapInArrow`` UDF functions. Bridges PySpark's
Arrow-based batch processing with Numba JIT-compiled functions by
converting each ``pyarrow.RecordBatch`` column through the adapter
layer before passing data to a user-supplied computation function.

Usage::

    from numbarrow.core.mapinarrow_factory import Nullable, make_mapinarrow_func

    def my_func(data_dict, bitmap_dict, broadcasts):
        # data_dict:   {name: np.ndarray} for a uniform column, or
        #              {name: {field: np.ndarray}} for a struct column
        # bitmap_dict: the same shape, each leaf a uint8 bitmap or None where
        #              the column carries no validity buffer; for a struct
        #              column the struct-level validity is folded into each
        #              field's bitmap
        # broadcasts:  {key: value}
        result = data_dict["input_col"] * broadcasts["scale"]
        # result is null wherever input_col is, so input_col's bitmap goes out
        # with it; a bare array would carry no nulls out, and a result with
        # nulls of its own packs them: np.packbits(valid, bitorder="little")
        return {"output_col": Nullable(result, bitmap_dict["input_col"])}

    udf = make_mapinarrow_func(my_func, broadcasts={"scale": 1.5})
    df_out = df_in.mapInArrow(udf, output_schema)

Every name in ``data_dict`` is also a key of ``bitmap_dict``, so a batch that
happens to contain no nulls is indexable exactly like one that does.

On the way out a bare array carries no nulls: a row that came in null goes out
valid, holding whatever the UDF computed from the placeholder under the null,
which is ``0``, ``0.0`` or ``''`` in a batch from Spark.
``Nullable(data, bitmap)`` carries nulls out, the bitmap being in the layout
``bitmap_dict`` hands out. A result that is null exactly where one input
column is, as in the example, passes that column's bitmap through,
``Nullable(result, bitmap_dict[column])``, or ``bitmap_dict[column][field]``
for a struct field; any other result builds its own, for instance
``np.packbits(valid, bitorder="little")`` from a boolean array ``valid``. A
packed bitmap carries no row count, so a bitmap the batch handed out is
accepted only on a column of the length it covers, the batch's rows for a
column's own bitmap and the flattened elements for a struct field's; one that
resizes the column needs a bitmap of its own.

For a ``StructArray`` column the struct-level validity is folded into each
field's bitmap, so a row that is null as a whole is visible to one ``is_null``
call per field. For a ``ListArray`` of structs the fold covers the flattened
struct elements, NOT the outer list rows: a null outer row can be reported
nowhere and also shifts the element-to-row mapping, so a list column whose
``null_count`` is non-zero raises ``NotImplementedError``.

A struct or list-of-struct column's fields sit under the column's own name,
``data_dict[column][field]``, so a field never shares a namespace with another
column or with another struct's fields.

See ``test_mapinarrow_spark.py`` in the `test suite
<https://github.com/Goykhman/numbarrow/tree/main/test>`_ for a complete runnable example.

Module
++++++

.. automodule:: numbarrow.core.mapinarrow_factory
   :members:
   :show-inheritance:
   :undoc-members:
