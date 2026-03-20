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

The key abstraction is ``arrays_viewers`` — a dictionary mapping NumPy dtypes
to pre-compiled viewer functions. Each viewer takes ``(address, length)`` and
returns a NumPy array backed by the memory at that address.

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
