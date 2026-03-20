numbarrow.core.adapters
=======================

Overview
++++++++

Type-dispatched adapters that convert PyArrow arrays into NumPy arrays
for use in Numba ``@njit`` compiled functions.

Uses ``functools.singledispatch`` to route each PyArrow array type to a
handler that extracts the underlying data buffer as a NumPy view (where
possible) and the validity bitmap as a uint8 array.

Supported types:

- ``BooleanArray`` (requires copy due to bit-packed layout)
- ``Int32Array``, ``Int64Array``, ``DoubleArray`` (zero-copy view)
- ``Date32Array`` (copy: int32 days → datetime64[D])
- ``Date64Array`` (zero-copy view as datetime64[ms])
- ``TimestampArray`` (zero-copy view as datetime64[unit])
- ``StringArray`` (copy into fixed-width NumPy Unicode array)
- ``StructArray`` (returns dicts of field name → array)
- ``ListArray`` (delegates to StructArray adapter for list-of-struct)

Module
++++++

.. automodule:: numbarrow.core.adapters
   :members:
   :show-inheritance:
   :undoc-members:
