numbarrow.core.is_null
======================

Overview
++++++++

Arrow uses a packed validity bitmap to track which elements in an array
are non-null. Each bit corresponds to one element: bit=1 means valid,
bit=0 means null. Bits are packed LSB-first into uint8 bytes.

This module provides a single Numba ``@njit`` compiled function that
reads the bitmap and returns whether a given element index is null.

Module
++++++

.. automodule:: numbarrow.core.is_null
   :members:
   :show-inheritance:
   :undoc-members:
