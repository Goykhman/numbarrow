"""numbarrow's njit options, and the decorator that compiles numbarrow's functions with them.

:func:`get_jit_options` reads the options from the ``NUMBARROW_JIT_OPTIONS`` environment variable, and ``jit_options``
holds what it returned when this module was first imported. Every function that numbarrow compiles goes through
:func:`jit_with_options`, which passes ``jit_options`` to numba's ``njit``. If numba cannot cache a function, the
decorator compiles it without a cache and warns.
"""

import inspect
import os
import json
import warnings

from numba import njit


#: The text that begins the ``ValueError`` message for a ``NUMBARROW_JIT_OPTIONS`` value that is not a JSON object.
invalid_jit_options_err = (
    """NUMBARROW_JIT_OPTIONS must be a JSON object, e.g., export NUMBARROW_JIT_OPTIONS='{"cache": false}'"""
)


def get_jit_options():
    """Return the njit options that ``NUMBARROW_JIT_OPTIONS`` sets, or ``{"cache": True}`` if it is unset or empty.

    The variable holds a JSON object, whose entries numbarrow passes to ``njit`` as keyword arguments, unchanged::

        export NUMBARROW_JIT_OPTIONS='{"cache": false}'

    The object replaces the default instead of adding to it, so ``'{}'`` turns caching off. A value that is not a JSON
    object raises ``ValueError``. So does a ``"cache"`` that is not ``true`` or ``false``, and a string for any option
    but ``"error_model"`` and ``"inline"``, the two options that take a string. numba reads a non-empty string as true
    for a boolean option, so ``{"boundscheck": "false"}`` would turn bounds checks on.

    The function reads the variable each time it is called. The decorators use ``jit_options`` instead, which holds
    what the function returned when this module was first imported.

    The key under which numba's on-disk cache stores compiled code does not include the options that the code was
    compiled with. A run that asks for other options therefore loads the code that an earlier run cached:
    ``{"cache": true, "boundscheck": true}`` over a cache filled under the default options loads code without bounds
    checks. Point ``NUMBA_CACHE_DIR`` at a new directory when an option changes.
    """
    as_str = os.environ.get("NUMBARROW_JIT_OPTIONS")
    if not as_str:
        return {"cache": True}
    try:
        as_json = json.loads(as_str)
    except json.JSONDecodeError as error:
        raise ValueError(f"{invalid_jit_options_err}; {as_str!r} is not valid JSON: {error}") from None
    if not isinstance(as_json, dict):
        raise ValueError(
            f"{invalid_jit_options_err}; {as_str!r} is valid JSON but a {type(as_json).__name__}, not an object"
        )
    if "cache" in as_json and not isinstance(as_json["cache"], bool):
        raise ValueError(
            f'NUMBARROW_JIT_OPTIONS "cache" must be true or false, not {as_json["cache"]!r}: numba reads any '
            f"non-empty string as true"
        )
    for name, value in as_json.items():
        if isinstance(value, str) and name not in ("error_model", "inline"):
            raise ValueError(
                f'NUMBARROW_JIT_OPTIONS "{name}" cannot be the string {value!r}: numba reads any non-empty string '
                f'as true, and only "error_model" and "inline" take a string'
            )
    return as_json


#: The njit options that :func:`jit_with_options` passes to numba: what :func:`get_jit_options` returned when this
#: module was first imported.
jit_options = get_jit_options()


def jit_with_options(signature):
    """Return a decorator that compiles a function for ``signature`` with numba's ``njit`` and ``jit_options``.

    numba compiles the function when the decorator is applied, because the signature is given. If caching is on and
    numba cannot cache the function, the decorator compiles it without a cache and issues a ``RuntimeWarning``. That
    happens for a read-only install, where numba can write neither in numbarrow's own directory nor in the user's
    cache directory, unless ``NUMBA_CACHE_DIR`` names a directory that it can write. It also happens for an import
    from an archive, such as an ``.egg`` that ``spark-submit --py-files`` puts on the workers' path, or a ``.whl`` or
    ``.pyz`` file placed on ``sys.path``. From version 0.61, numba caches a module imported from a ``.zip`` and writes
    the cache to the user's cache directory. The decorator then warns only if that directory cannot be written.

    The warning names a way to a cache. For a source file on disk, that is ``NUMBA_CACHE_DIR`` set to a writable
    directory. numba ignores ``NUMBA_CACHE_DIR`` for a module in an archive, so there the warning names an unpacked
    install or a ``.zip`` instead. In either case, ``NUMBARROW_JIT_OPTIONS='{"cache": false}'`` turns caching off and
    silences the warning. Any other error propagates, and so does every error with caching off.
    """
    def decorate(func):
        try:
            return njit(signature, **jit_options)(func)
        except (RuntimeError, OSError) as error:
            # numba looks for a cache directory that it can write when the function is decorated, and raises
            # RuntimeError with "no locator available" if it finds none. For a .zip, numba 0.61 and later take the
            # user's cache directory without that check, and the first save into it raises OSError instead. That save
            # happens during the decoration too, because numba compiles a function with an explicit signature at once.
            cache_failed = isinstance(error, OSError) or "no locator available" in str(error)
            # Another RuntimeError, or any error with caching off, is not the cache's and propagates. The decorator
            # treats every OSError as the cache's; if one has another cause, the compilation without a cache below
            # raises it again.
            if not cache_failed or not jit_options.get("cache"):
                raise
            silence = "NUMBARROW_JIT_OPTIONS='{\"cache\": false}' to turn caching off and silence this warning"
            if os.path.exists(inspect.getfile(func)):
                remedy = f"Set NUMBA_CACHE_DIR to a writable directory, or {silence}"
            else:
                # numba reads NUMBA_CACHE_DIR only for a function whose source file is on disk. For a module in an
                # archive, the warning names an unpacked install or a .zip instead. numba 0.61 and later cache a module
                # from an archive only if a part of its path ends in .zip, so a .whl, .pyz or .egg is not cached,
                # although each is a zip file.
                remedy = (
                    "NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: to cache, "
                    "install numbarrow unpacked or import it from a .zip, which numba 0.61 and later cache in the "
                    f"user's cache directory. Set {silence}"
                )
            warnings.warn(
                f"numba cannot cache {func.__qualname__} here ({error}); it compiles without a cache. {remedy}",
                RuntimeWarning, stacklevel=2,
            )
            return njit(signature, **{**jit_options, "cache": False})(func)
    return decorate
