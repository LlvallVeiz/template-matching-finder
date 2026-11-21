"""Detection on top of the score maps: single match, multi-location + NMS,
multi-scale pyramids and discrete-rotation search.

Coordinate convention: a detection is the axis-aligned box ``(x, y, w, h)``
where ``(x, y)`` is the top-left corner in the *original* image and ``w, h``
are the template dimensions (after undoing any pyramid scaling).  Scores keep
the raw meaning of their method — similarity methods (``ncc``/``zncc``) score
in [-1, 1] with higher better; distance methods (``ssd``/``sad``) score >= 0
with lower better.  The helper :func:`is_similarity` (from ``metrics``)
decides the direction everywhere, so callers never have to special-case it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .correlation import as_gray, best_location, score_map
from .errors import InvalidMethodError, InvalidParameterError
from .metrics import METHODS, is_similarity

Array = NDArray[np.float64]


@dataclass
class MatchResult:
    """One detection: bounding box, score and how it was found."""

    x: int
    y: int
    w: int
    h: int
    score: float
    method: str
    template_name: str = ""
    scale: float = 1.0
    angle: float = 0.0

    def box(self) -> tuple[int, int, int, int]:
        """Return the bounding box as ``(x, y, w, h)``."""
        return (self.x, self.y, self.w, self.h)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dictionary."""
        return asdict(self)


def iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    """Intersection over Union of two ``(x, y, w, h)`` boxes, in [0, 1]."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter_w, inter_h = ix1 - ix0, iy1 - iy0
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def non_maximum_suppression(
    detections: list[MatchResult], iou_threshold: float = 0.5
) -> list[MatchResult]:
    """Greedy IoU-based NMS.

    Detections are visited best-first (highest similarity / lowest distance)
    and a detection is kept only when its box overlaps no already-kept box
    by more than ``iou_threshold``.  This is the standard duplicate-removal
    step of object detection (see ``docs/RESEARCH.md``).
    """
    if not 0.0 <= iou_threshold <= 1.0:
        raise InvalidParameterError("iou_threshold must be in [0, 1]")
    if not detections:
        return []
    similarity = is_similarity(detections[0].method)
    ordered = sorted(detections, key=lambda d: d.score, reverse=similarity)
    kept: list[MatchResult] = []
    for cand in ordered:
        if all(iou(cand.box(), other.box()) <= iou_threshold for other in kept):
            kept.append(cand)
    return kept


def _check_threshold(method: str, threshold: float | None) -> None:
    if threshold is None:
        return
    if is_similarity(method) and not -1.0 <= threshold <= 1.0:
        raise InvalidParameterError(
            f"similarity threshold {threshold} must be in [-1, 1]"
        )
    if not is_similarity(method) and threshold < 0:
        raise InvalidParameterError(f"distance threshold {threshold} must be >= 0")


def match_single(
    image: Array,
    template: Array,
    method: str = "zncc",
    template_name: str = "",
) -> MatchResult:
    """Return the single best match of ``template`` in ``image``.

    Raises :class:`TemplateTooLargeError` when the template does not fit.
    """
    if method not in METHODS:
        raise InvalidMethodError(f"unknown method {method!r}; expected one of {METHODS}")
    scores = score_map(image, template, method)
    x, y, value = best_location(scores, method)
    th, tw = as_gray(template).shape
    return MatchResult(
        x=x, y=y, w=int(tw), h=int(th), score=value,
        method=method, template_name=template_name,
    )


def match_all(
    image: Array,
    template: Array,
    method: str = "zncc",
    threshold: float | None = 0.8,
    max_results: int = 10,
    iou_threshold: float = 0.5,
    template_name: str = "",
) -> list[MatchResult]:
    """Find all locations scoring past ``threshold``, then apply NMS.

    Candidates are collected from the whole score map, sorted best-first and
    greedily suppressed by IoU, so one true instance yields one box even
    though its neighbours also score highly.  ``threshold=None`` means the
    default: 0.8 for similarity methods, no thresholding (rank only) for
    distance methods, whose raw SSD/SAD values are not scale-free — pass an
    explicit bound when you need one.
    """
    if method not in METHODS:
        raise InvalidMethodError(f"unknown method {method!r}; expected one of {METHODS}")
    if max_results < 1:
        raise InvalidParameterError("max_results must be >= 1")
    if threshold is None and not is_similarity(method):
        pass  # distance methods default to unthresholded ranking
    elif threshold is None:
        threshold = 0.8
    _check_threshold(method, threshold)
    scores = score_map(image, template, method)
    th, tw = as_gray(template).shape
    similarity = is_similarity(method)
    if threshold is not None:
        ys, xs = np.where(scores >= threshold if similarity else scores <= threshold)
    else:
        ys, xs = np.unravel_index(np.arange(scores.size), scores.shape)
    candidates = [
        MatchResult(
            x=int(x), y=int(y), w=int(tw), h=int(th),
            score=float(scores[y, x]), method=method,
            template_name=template_name,
        )
        for y, x in zip(ys, xs)
    ]
    candidates.sort(key=lambda d: d.score, reverse=similarity)
    return non_maximum_suppression(candidates, iou_threshold)[:max_results]


def _resize_gray(image: Array, scale: float) -> Array:
    """Resize a grayscale array by ``scale`` with PIL (LANCZOS)."""
    pil = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="L")
    new_w = max(1, round(pil.width * scale))
    new_h = max(1, round(pil.height * scale))
    return np.asarray(pil.resize((new_w, new_h), Image.LANCZOS), dtype=np.float64)


def match_multiscale(
    image: Array,
    template: Array,
    method: str = "zncc",
    scales: tuple[float, ...] = (1.0, 0.5),
    threshold: float | None = 0.8,
    max_results: int = 10,
    iou_threshold: float = 0.5,
    template_name: str = "",
) -> list[MatchResult]:
    """Search at several image scales and merge the results with NMS.

    ``scale`` is the factor applied to the *image*: at scale 0.5 the image is
    halved, so a template matches objects twice its size in the original.
    Detections are mapped back to original coordinates (``x / scale``) and
    their boxes are rescaled to ``(w / scale, h / scale)``.  Scales are tried
    coarse-to-fine; the merged list is deduplicated by NMS across scales.
    """
    if not scales:
        raise InvalidParameterError("scales must not be empty")
    if any(s <= 0 for s in scales):
        raise InvalidParameterError("scales must be positive")
    image = as_gray(image)
    template = as_gray(template)
    th, tw = template.shape
    all_detections: list[MatchResult] = []
    for scale in scales:
        scaled = _resize_gray(image, scale) if scale != 1.0 else image
        if scaled.shape[0] < th or scaled.shape[1] < tw:
            continue  # template cannot fit at this pyramid level
        found = match_all(
            scaled, template, method=method, threshold=threshold,
            max_results=max_results, iou_threshold=iou_threshold,
            template_name=template_name,
        )
        for det in found:
            det.x = int(round(det.x / scale))
            det.y = int(round(det.y / scale))
            det.w = int(round(tw / scale))
            det.h = int(round(th / scale))
            det.scale = float(scale)
            all_detections.append(det)
    similarity = is_similarity(method)
    all_detections.sort(key=lambda d: d.score, reverse=similarity)
    return non_maximum_suppression(all_detections, iou_threshold)[:max_results]


def rotate_template(template: Array, angle: float) -> Array:
    """Rotate a template by ``angle`` degrees counter-clockwise.

    Multiples of 90 use exact ``np.rot90`` (no resampling blur); other angles
    use PIL bilinear rotation with ``expand=True`` and the template's corner
    value as fill, so the rotated patch keeps its background.
    """
    template = as_gray(template)
    angle = float(angle) % 360.0
    if angle == 0.0:
        return template.copy()
    if angle % 90.0 == 0.0:
        return np.rot90(template, k=int(round(angle / 90.0))).copy()
    pil = Image.fromarray(np.clip(template, 0, 255).astype(np.uint8), mode="L")
    fill = int(template[0, 0])
    rotated = pil.rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor=fill)
    return np.asarray(rotated, dtype=np.float64)


def match_rotated(
    image: Array,
    template: Array,
    method: str = "zncc",
    angles: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0),
    threshold: float | None = 0.8,
    max_results: int = 10,
    iou_threshold: float = 0.5,
    template_name: str = "",
) -> list[MatchResult]:
    """Search the template at each discrete rotation in ``angles``.

    Plain template matching is not rotation invariant, so each rotated copy
    is matched separately and the results are merged with NMS.  Each result
    records the ``angle`` that produced it.
    """
    if not angles:
        raise InvalidParameterError("angles must not be empty")
    image = as_gray(image)
    template = as_gray(template)
    all_detections: list[MatchResult] = []
    for angle in angles:
        rotated = rotate_template(template, angle)
        if rotated.shape[0] > image.shape[0] or rotated.shape[1] > image.shape[1]:
            continue
        found = match_all(
            image, rotated, method=method, threshold=threshold,
            max_results=max_results, iou_threshold=iou_threshold,
            template_name=template_name,
        )
        for det in found:
            det.angle = float(angle) % 360.0
            all_detections.append(det)
    similarity = is_similarity(method)
    all_detections.sort(key=lambda d: d.score, reverse=similarity)
    return non_maximum_suppression(all_detections, iou_threshold)[:max_results]


def match_batch(
    image: Array,
    templates: dict[str, Array],
    method: str = "zncc",
    threshold: float | None = 0.8,
    max_results: int = 10,
    iou_threshold: float = 0.5,
    scales: tuple[float, ...] | None = None,
    angles: tuple[float, ...] | None = None,
) -> dict[str, list[MatchResult]]:
    """Match several named templates against one image.

    Returns ``{name: [MatchResult, ...]}``.  When ``scales`` is given the
    multi-scale search runs per template; when ``angles`` is given the
    rotation search runs per template.  Templates that cannot fit the image
    yield an empty list instead of raising, so one bad template does not
    abort the whole batch (the single-template functions still raise).
    """
    from .errors import TemplateTooLargeError

    image = as_gray(image)
    results: dict[str, list[MatchResult]] = {}
    for name, template in templates.items():
        try:
            if scales is not None:
                found = match_multiscale(
                    image, template, method=method, scales=scales,
                    threshold=threshold, max_results=max_results,
                    iou_threshold=iou_threshold, template_name=name,
                )
            elif angles is not None:
                found = match_rotated(
                    image, template, method=method, angles=angles,
                    threshold=threshold, max_results=max_results,
                    iou_threshold=iou_threshold, template_name=name,
                )
            else:
                found = match_all(
                    image, template, method=method, threshold=threshold,
                    max_results=max_results, iou_threshold=iou_threshold,
                    template_name=name,
                )
        except TemplateTooLargeError:
            found = []
        results[name] = found
    return results
