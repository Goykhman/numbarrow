API reference
=============

Most programs use three functions. :func:`~numbarrow.core.adapters.arrow_array_adapter` converts an Arrow array into
numpy arrays, :func:`~numbarrow.core.is_null.is_null` reads a validity bitmap in jitted code, and
:func:`~numbarrow.core.mapinarrow_factory.make_mapinarrow_func` builds a function for PySpark's ``mapInArrow``.
:doc:`numbarrow.core.configurations` holds the options that numbarrow compiles its own numba functions with, such as
``is_null``. :doc:`numbarrow.utils` holds the conversions that ``arrow_array_adapter`` calls for numeric, string,
struct and list arrays, and the functions that let compiled code read Arrow memory.

.. toctree::
   :maxdepth: 4

   numbarrow.core.adapters
   numbarrow.core.is_null
   numbarrow.core.mapinarrow_factory
   numbarrow.core.configurations
   numbarrow.utils
