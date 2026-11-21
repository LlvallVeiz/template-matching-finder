"""Full-image score maps: SSD, SAD, NCC and ZNCC at every template position.

The naive way to score every position is a double loop over the image and the
template.  For NCC/ZNCC this package instead implements the fast algorithm of
Lewis (1995), "Fast Normalized Cross-Correlation":

* the plain cross-correlation ``sum(I * T)`` over all windows is computed in
  one shot with an FFT convolution (``scipy.signal.fftconvolve``);
* the per-window sums ``sum(I)`` and ``sum(I**2)`` come from summed-area
  tables (integral images, Crow 1984), so each window sum costs O(1).

That reduces the whole ZNCC map to O(W*H*log(W*H)) instead of
O(W*H*w*h).  SSD is expressed through the same two ingredients
(``SSD = sum(I**2) - 2*cross + sum(T**2)``); SAD has no such shortcut and is
computed by accumulating shifted absolute differences, one pass per template
pixel.

A slow brute-force implementation (:func:`brute_force_map`) is kept as an
independent reference: the test suite checks every fast map against it on
small random images, so a bug in the FFT/integral-image algebra cannot hide.

All maps have shape ``(H - h + 1, W - w + 1)``; entry ``[y, x]`` scores the
window whose top-left corner is ``(x, y)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve

from .errors import EmptyImageError, InvalidMethodError, TemplateTooLargeError
from .metrics import METHODS, is_similarity

Array = NDArray[np.float64]


def as_gray(image: Array) -> Array:
    """Return ``image`` as a 2-D float64 grayscale array.

    Accepts 2-D arrays directly, or 3-channel RGB(A) arrays, which are
    converted with the ITU-R BT.601 luma weights (0.299, 0.587, 0.114).
    """
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        rgb = arr[:, :, :3]
        return (
            0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        )
    raise ValueError(f"expected a 2-D grayscale or 3-channel image, got shape {arr.shape}")


def _check_sizes(image: Array, template: Array) -> tuple[Array, Array]:
    """Validate sizes and return float64 copies."""
    image = as_gray(image)
    template = as_gray(template)
    if image.size == 0 or template.size == 0:
        raise EmptyImageError("image and template must not be empty")
    if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
        raise TemplateTooLargeError(
            f"template {template.shape} is larger than image {image.shape}"
        )
    return image, template


def integral_image(image: Array) -> Array:
    """Summed-area table with a zero row/column prepended.

    ``ii[y + 1, x + 1]`` holds the sum of ``image[:y + 1, :x + 1]``, so the sum
    of any axis-aligned rectangle is four table lookups (Crow 1984).
    """
    image = np.asarray(image, dtype=np.float64)
    ii = np.zeros((image.shape[0] + 1, image.shape[1] + 1), dtype=np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(image, axis=0), axis=1)
    return ii


def _window_sums(ii: Array, height: int, width: int) -> Array:
    """Sum of every ``height`` x ``width`` window from an integral image."""
    return (
        ii[height:, width:]
        - ii[:-height, width:]
        - ii[height:, :-width]
        + ii[:-height, :-width]
    )


def _cross_correlation(image: Array, template: Array) -> Array:
    """``sum(I * T)`` for every window, via FFT convolution.

    Convolving with the template flipped on both axes turns convolution into
    cross-correlation; ``mode="valid"`` keeps exactly the window positions.
    """
    return fftconvolve(image, template[::-1, ::-1], mode="valid")


def zncc_map(image: Array, template: Array) -> Array:
    """Zero-mean normalized cross-correlation map (Lewis 1995).

    Values lie in [-1, 1]; 1 is a perfect match up to an affine brightness
    change.  Windows with zero variance score 0.0 (the ratio is undefined
    there) instead of producing NaN.
    """
    image, template = _check_sizes(image, template)
    th, tw = template.shape
    n = float(th * tw)
    mean_t = float(template.mean())
    t_centered_energy = float(np.sum((template - mean_t) ** 2))
    if t_centered_energy == 0.0:
        # A constant template has no texture to correlate against.
        return np.zeros(
            (image.shape[0] - th + 1, image.shape[1] - tw + 1), dtype=np.float64
        )
    cross = _cross_correlation(image, template)
    s1 = _window_sums(integral_image(image), th, tw)
    s2 = _window_sums(integral_image(image * image), th, tw)
    numerator = cross - mean_t * s1
    window_var = np.maximum(s2 - (s1 * s1) / n, 0.0)
    denom = np.sqrt(window_var * t_centered_energy)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(denom > 0, numerator / np.where(denom > 0, denom, 1.0), 0.0)
    return np.clip(result, -1.0, 1.0)


def ncc_map(image: Array, template: Array) -> Array:
    """Normalized cross-correlation map without mean subtraction.

    Values lie in [0, 1] for non-negative images; 1 is a perfect match up to
    a positive scale factor.  Unlike ZNCC it is *not* invariant to an added
    brightness offset.
    """
    image, template = _check_sizes(image, template)
    th, tw = template.shape
    cross = _cross_correlation(image, template)
    s2 = _window_sums(integral_image(image * image), th, tw)
    denom = np.sqrt(np.maximum(s2, 0.0) * float(np.sum(template * template)))
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(denom > 0, cross / np.where(denom > 0, denom, 1.0), 0.0)
    return np.clip(result, 0.0, 1.0)


def ssd_map(image: Array, template: Array) -> Array:
    """Sum-of-squared-differences map; 0 is a perfect match.

    Uses the identity ``SSD = sum(I**2) - 2*sum(I*T) + sum(T**2)`` with the
    FFT cross-correlation and the integral image of ``I**2``.
    """
    image, template = _check_sizes(image, template)
    th, tw = template.shape
    cross = _cross_correlation(image, template)
    s2 = _window_sums(integral_image(image * image), th, tw)
    result = s2 - 2.0 * cross + float(np.sum(template * template))
    return np.maximum(result, 0.0)


def sad_map(image: Array, template: Array) -> Array:
    """Sum-of-absolute-differences map; 0 is a perfect match.

    No FFT shortcut exists for the absolute value, so this accumulates one
    shifted absolute difference per template pixel.  It is exact but slower
    than the other maps for large templates.
    """
    image, template = _check_sizes(image, template)
    th, tw = template.shape
    out_h = image.shape[0] - th + 1
    out_w = image.shape[1] - tw + 1
    result = np.zeros((out_h, out_w), dtype=np.float64)
    for dy in range(th):
        for dx in range(tw):
            result += np.abs(
                image[dy : dy + out_h, dx : dx + out_w] - template[dy, dx]
            )
    return result


def score_map(image: Array, template: Array, method: str) -> Array:
    """Dispatch to the score map for ``method`` (``ssd``/``sad``/``ncc``/``zncc``)."""
    if method == "zncc":
        return zncc_map(image, template)
    if method == "ncc":
        return ncc_map(image, template)
    if method == "ssd":
        return ssd_map(image, template)
    if method == "sad":
        return sad_map(image, template)
    raise InvalidMethodError(f"unknown method {method!r}; expected one of {METHODS}")


def brute_force_map(image: Array, template: Array, method: str) -> Array:
    """Slow reference implementation: score every window with :mod:`metrics`.

    Used only by the test suite to verify the fast maps independently.
    """
    from . import metrics as _metrics

    image, template = _check_sizes(image, template)
    th, tw = template.shape
    out_h = image.shape[0] - th + 1
    out_w = image.shape[1] - tw + 1
    fn = getattr(_metrics, method)
    result = np.zeros((out_h, out_w), dtype=np.float64)
    for y in range(out_h):
        for x in range(out_w):
            result[y, x] = fn(image[y : y + th, x : x + tw], template)
    return result


def best_location(score: Array, method: str) -> tuple[int, int, float]:
    """Return ``(x, y, value)`` of the best entry of a score map.

    Similarity methods take the maximum, distance methods the minimum.
    """
    if is_similarity(method):
        flat = int(np.argmax(score))
    else:
        flat = int(np.argmin(score))
    y, x = np.unravel_index(flat, score.shape)
    return int(x), int(y), float(score[y, x])
