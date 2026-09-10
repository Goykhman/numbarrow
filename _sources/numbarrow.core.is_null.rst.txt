numbarrow.core.is_null
======================

Overview
++++++++

Arrow uses a packed validity bitmap to track which elements in an array
are non-null. Each bit corresponds to one element: bit=1 means valid,
bit=0 means null. Bits are packed LSB-first into uint8 bytes.

This module provides three Numba ``@njit`` compiled helpers that read
such bitmaps:

- :func:`~numbarrow.core.is_null.is_null` returns whether one element
  index is null according to a single bitmap.
- :func:`~numbarrow.core.is_null.is_null_struct` answers the same
  question across a struct's two validity layers, its own and the
  field's, since a value is null when either layer says so.
- :func:`~numbarrow.core.is_null.unpack_booleans` expands bit-packed
  boolean *data* into a boolean array; Arrow packs boolean values the
  same way it packs validity bits.

Module
++++++

.. automodule:: numbarrow.core.is_null
   :members:
   :show-inheritance:
   :undoc-members:
