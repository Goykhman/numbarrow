numbarrow.core.mapinarrow_factory
=================================

Overview
++++++++

PySpark's `mapInArrow
<https://spark.apache.org/docs/3.5.7/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.mapInArrow.html>`_
runs a Python function on an iterator of Arrow record batches and expects an iterator of batches back.
:func:`~numbarrow.core.mapinarrow_factory.make_mapinarrow_func` builds that function from a simpler one of yours,
``main_func(data_dict, bitmap_dict, broadcasts)``. ``main_func`` receives the columns of each batch as numpy arrays in
dicts, and it returns a dict of output columns:

.. code-block:: python

    import pyarrow as pa
    from numbarrow.core.mapinarrow_factory import Nullable, make_mapinarrow_func


    def main_func(data_dict, bitmap_dict, broadcasts):
        doubled = data_dict["value"] * broadcasts["factor"]
        return {"doubled": Nullable(doubled, bitmap_dict["value"])}


    udf = make_mapinarrow_func(main_func, broadcasts={"factor": 2.0})
    batch = pa.RecordBatch.from_pydict({"value": [1.5, None, 4.0]})
    print(next(udf(iter([batch]))).column("doubled").to_pylist())  # [3.0, None, 8.0]

:class:`~numbarrow.core.mapinarrow_factory.Nullable` pairs an output column with a validity bitmap, here the bitmap of
the input column ``value``, so that ``doubled`` is null where ``value`` is null. A plain numpy array in the returned
dict becomes a column without nulls, except for a ``NaT``, which pyarrow reads as null. The example passes no
``output_schema``, so numbarrow infers the type of each output column from its values. In Spark, pass the same
``udf`` to ``df.mapInArrow(udf, "doubled double")``. The built function takes any iterator of record batches, so a
test can call it without Spark, as the example does.

For each batch, the built function converts every column of the batch with
:func:`~numbarrow.core.adapters.arrow_array_adapter`, or only the columns that the ``input_columns`` parameter names.
It then calls ``main_func`` once, with three dicts:

- ``data_dict`` maps each column's name to its numpy data. A struct column, or a list of structs, maps to a dict of
  its fields: ``data_dict["point"]["x"]`` is the field ``x`` of the column ``point``.
- ``bitmap_dict`` has the same keys in every batch and holds the validity bitmaps, or ``None`` where a column has no
  validity buffer.
- ``broadcasts`` is the dict given to ``make_mapinarrow_func``.

For a struct column, ``bitmap_dict`` differs from what ``arrow_array_adapter`` returns. The adapter returns the
struct's own bitmap apart from the bitmaps of its fields, for :func:`~numbarrow.core.is_null.is_null_struct` to read
both. ``bitmap_dict`` holds no bitmap for the struct itself. numbarrow combines it into the bitmap of each field, which
then also marks the rows where the whole struct is null. One :func:`~numbarrow.core.is_null.is_null` call per field is
therefore enough.

The README's `PySpark mapInArrow`_ section shows a complete Spark job. Its subsections explain how to
`return nulls`_, why columns `bind by position`_ unless the ``output_schema`` parameter binds them by name, and how
numbarrow chooses the `types of the output columns`_. The reference below documents the parameters and the errors
that ``make_mapinarrow_func`` raises.

.. _PySpark mapInArrow: https://github.com/Goykhman/numbarrow#pyspark-mapinarrow
.. _return nulls: https://github.com/Goykhman/numbarrow#returning-nulls
.. _bind by position: https://github.com/Goykhman/numbarrow#columns-bind-by-position-not-by-name
.. _types of the output columns: https://github.com/Goykhman/numbarrow#output-column-types

Module
++++++

.. automodule:: numbarrow.core.mapinarrow_factory
   :members:
   :show-inheritance:
   :undoc-members:
