numbarrow.core.is_null
======================

Overview
++++++++

Arrow marks the null rows of an array in a validity bitmap: one bit per row, packed into bytes with the least
significant bit first, set for a valid row and clear for a null one.
:func:`~numbarrow.core.adapters.arrow_array_adapter` returns that bitmap as a ``uint8`` array, and the functions of
this module read it inside ``@njit`` code. Python code can call them too.

- ``is_null(i, bitmap)`` returns ``True`` when row ``i`` is null. The function does not accept ``None`` for
  ``bitmap``, so check a bitmap for ``None`` before the call.
- ``is_null_struct(i, struct_bitmap, field_bitmap)`` returns ``True`` when the struct's bitmap or the field's bitmap
  marks row ``i`` null, and it accepts ``None`` for either. Use it with the 3-tuple that ``arrow_array_adapter``
  returns for a struct, because a row that is null as a whole is not always null in its fields. The field bitmaps that
  :func:`~numbarrow.core.mapinarrow_factory.make_mapinarrow_func` passes already include the struct's nulls, so
  ``is_null`` is enough for them.
- ``unpack_booleans(offset, length, packed_data)`` expands bit-packed booleans into a ``bool_`` array. Arrow packs the
  values of a ``BooleanArray`` in the same layout as a validity bitmap, and the adapter uses this function to unpack
  them.

In the example, the struct has no bitmap, so ``is_null_struct`` receives ``None`` for it and ``bitmap`` as the field's:

.. code-block:: python

    import numpy as np
    from numbarrow.core.is_null import is_null, is_null_struct

    bitmap = np.packbits([True, False, True], bitorder="little")  # rows 0 and 2 valid
    print(bitmap, is_null(1, bitmap), is_null_struct(1, None, bitmap))  # [5] True True

numba compiles the three functions when this module is first imported, each for one signature. A call converts an
index of another integer type to ``int64``, and it accepts a writable bitmap as well as a read-only one. ``None`` as
the bitmap of ``is_null`` raises ``TypeError`` in a call from Python, and a jitted function that passes it fails to
compile.

By default, the functions do not check an index against the length of the bitmap. An index past the end reads memory
outside the bitmap: the answer is meaningless, and the read can crash the process.
``NUMBARROW_JIT_OPTIONS='{"boundscheck": true}'`` makes such a read raise ``IndexError``. The README section on
`compilation and the cache`_ explains the variable, and how to keep numba from loading cached code that was compiled
without bounds checks.

.. _compilation and the cache: https://github.com/Goykhman/numbarrow#compilation-and-the-cache

Module
++++++

.. automodule:: numbarrow.core.is_null
   :members:
   :show-inheritance:
   :undoc-members:
