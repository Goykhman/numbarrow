numbarrow.core.mapinarrow_factory
=================================

Overview
++++++++

Factory for PySpark ``mapInArrow`` UDF functions. Bridges PySpark's
Arrow-based batch processing with Numba JIT-compiled functions by
converting each ``pyarrow.RecordBatch`` column through the adapter
layer before passing data to a user-supplied computation function.

Usage::

    from numbarrow.core.mapinarrow_factory import make_mapinarrow_func

    def my_func(data_dict, bitmap_dict, broadcasts):
        # data_dict:   {col_name: np.ndarray}
        # bitmap_dict: {col_name: np.ndarray (uint8 bitmap)}
        # broadcasts:  {key: value}
        result = ...
        return {"output_col": result}

    udf = make_mapinarrow_func(my_func, broadcasts={"scale": 1.5})
    df_out = df_in.mapInArrow(udf, output_schema)

See `test/demo_map_in_arrow.py <../test/demo_map_in_arrow.py>`_ for a complete runnable example.

Module
++++++

.. automodule:: numbarrow.core.mapinarrow_factory
   :members:
   :show-inheritance:
   :undoc-members:
