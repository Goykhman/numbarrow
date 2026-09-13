numbarrow.utils
===============

numbarrow.utils.utils
---------------------

Overview
''''''''

Low-level pointer utilities for zero-copy access to Arrow memory buffers.
Provides Numba-compatible functions that reinterpret a raw memory address
(from ``pyarrow.Buffer.address``) as a typed NumPy array, enabling ``@njit``
code to read Arrow buffer data directly without copying.

``arrays_viewers`` holds the three viewers the adapters are built on, uint8
for validity bitmaps and packed booleans, int32 and int64 for string offsets.
Each takes ``(address, length)`` and returns a bare view over the memory at
that address, with no owner and no read-only flag, valid only while that
memory is. The supported way in is
:func:`~numbarrow.core.adapters.arrow_array_adapter`, which returns read-only
arrays tied to the Arrow array they view.

.. automodule:: numbarrow.utils.utils
   :members:
   :show-inheritance:
   :undoc-members:

numbarrow.utils.arrow_array_utils
---------------------------------

Overview
''''''''

Higher-level utilities for extracting data from PyArrow array buffers as
NumPy arrays. Handles uniform arrays (fixed-width elements), string arrays
(variable-length with offset buffers), struct arrays, and list-of-struct
arrays.

.. automodule:: numbarrow.utils.arrow_array_utils
   :members:
   :show-inheritance:
   :undoc-members:
