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
- ``Int32Array``, ``Int64Array``, ``DoubleArray``, ``UInt8Array`` (zero-copy view)
- ``Date32Array`` (copy: int32 days → datetime64[D])
- ``Date64Array`` (zero-copy view as datetime64[ms])
- ``TimestampArray`` (zero-copy view as datetime64[unit])
- ``StringArray``, ``LargeStringArray`` (tuple of bitmap and a copy of data into a
  fixed-width NumPy Unicode array, whose width counts characters)
- ``StructArray`` (returns a 3-tuple: the struct-level validity bitmap, then bitmaps
  and data as two dicts keyed by field name)
- ``ListArray`` of structs (delegates to the StructArray adapter, honouring the list
  array's own offset; any other element type raises ``NotImplementedError``). The
  elements are flattened and no offsets are returned, so a null outer row can be
  neither reported nor placed in the element-to-row mapping; a list column whose
  ``null_count`` is non-zero raises ``NotImplementedError``.

A row that is null as a whole carries no validity bits in its fields, so the
struct-level bitmap is the only record of it. Pass both layers to
``numbarrow.core.is_null.is_null_struct``.

A string value whose last character is NUL raises ``ValueError``: numpy's
fixed-width ``|U`` dtype pads with NUL, so a trailing NUL cannot be told from
padding and would come back silently truncated. Leading and interior NULs are
preserved.

Returned data arrays are read-only and cannot be made writable. The views are
over Arrow buffers the caller does not own, which is also why pyarrow's own
``to_numpy(zero_copy_only=True)`` refuses to hand out a writable one; the
copies, booleans, ``date32`` and strings, are marked read-only as well, so the
contract does not depend on the type. Declare numba signatures that receive
them with ``readonly=True``, which accepts writable arrays too. Returned
bitmaps own their memory and are writable.

A bitmap is ``None`` when the array carries no validity buffer and a uint8
array otherwise, which is not the same as having no nulls: ``slice``, ``take``,
``filter`` and ``fill_null`` keep an all-valid buffer, and Arrow IPC, which is
what Spark's transport uses, drops one when a batch has no nulls. Declare a
bitmap parameter ``Optional`` in an eager numba signature, or it compiles on
one batch and raises ``No matching definition`` on the next.

A ``TimestampArray`` adapts by its unit alone, so a zoned and a naive timestamp
holding the same int64 adapt to the same ``datetime64``, as pyarrow's
``to_numpy`` does; the zone is not reported.

Module
++++++

.. automodule:: numbarrow.core.adapters
   :members:
   :show-inheritance:
   :undoc-members:
