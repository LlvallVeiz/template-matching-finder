"""Pointwise similarity / distance metrics between two equal-size patches.

All functions operate on 2-D grayscale arrays of identical shape and return a
single float.  They are the building blocks of the score maps in
:mod:`template_matching.correlation` and are deliberately written as plain
NumPy so that every value can be checked by hand (see ``tests/test_metrics.py``).

Conventions
-----------
* ``ssd`` / ``sad`` are *distances*: 0 means identical, larger means more
  different.  SSD is the squared Euclidean distance; SAD is the Manhattan
  (L1) distance.
* ``ncc`` / ``zncc`` are *similarities* in [-1, 1]: 1 means identical up to a
  positive scale factor (NCC) or up to an affine brightness change
  (ZNCC), -1 means perfectly anti-correlated.
* ``ncc`` here is OpenCV's ``TM_CCORR_NORMED`` and ``zncc`` is OpenCV's
  ``TM_CCOEFF_NORMED`` (see ``docs/RESEARCH.md``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .errors import EmptyImageError

Array = NDArray[np.float64]


def _check_pair(a: Array, b: Array) -> tuple[Array, Array]:
    """Validate a pair of patches and return them as float64 arrays."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        raise EmptyImageError("patches must not be empty")
    if a.shape != b.shape:
        raise ValueError(f"patch shapes differ: {a.shape} vs {b.shape}")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("patches must be 2-D grayscale arrays")
    return a, b


def ssd(a: Array, b: Array) -> float:
    """Sum of squared differences: ``sum((a - b) ** 2)``.

    >>> ssd([[1, 2], [3, 4]], [[1, 2], [3, 4]])
    0.0
    """
    a, b = _check_pair(a, b)
    diff = a - b
    return float(np.sum(diff * diff))


def sad(a: Array, b: Array) -> float:
    """Sum of absolute differences: ``sum(abs(a - b))``."""
    a, b = _check_pair(a, b)
    return float(np.sum(np.abs(a - b)))


def ncc(a: Array, b: Array) -> float:
    """Normalized cross-correlation (no mean subtraction).

    ``sum(a * b) / sqrt(sum(a**2) * sum(b**2))``.  Returns 0.0 when either
    patch is all zeros (the ratio is undefined there).
    """
    a, b = _check_pair(a, b)
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denom == 0.0:
        return 0.0
    return float(np.sum(a * b) / denom)


def zncc(a: Array, b: Array) -> float:
    """Zero-mean normalized cross-correlation (Pearson correlation of pixels).

    Both patches have their mean subtracted before the NCC formula is
    applied, which makes the score invariant to an added brightness offset
    *and* to a multiplicative contrast change.  Returns 0.0 when either patch
    is constant (zero variance makes the ratio undefined).
    """
    a, b = _check_pair(a, b)
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = float(np.sqrt(np.sum(a0 * a0) * np.sum(b0 * b0)))
    if denom == 0.0:
        return 0.0
    return float(np.sum(a0 * b0) / denom)


#: Methods whose score is a similarity (higher is better).
SIMILARITY_METHODS = ("ncc", "zncc")
#: Methods whose score is a distance (lower is better).
DISTANCE_METHODS = ("ssd", "sad")
#: All supported method names.
METHODS = SIMILARITY_METHODS + DISTANCE_METHODS


def is_similarity(method: str) -> bool:
    """Return True when ``method`` scores similarity (higher is better)."""
    if method in SIMILARITY_METHODS:
        return True
    if method in DISTANCE_METHODS:
        return False
    from .errors import InvalidMethodError

    raise InvalidMethodError(
        f"unknown method {method!r}; expected one of {METHODS}"
    )
