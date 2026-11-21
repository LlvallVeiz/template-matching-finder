"""template-matching-finder: template matching from scratch (NumPy + Pillow).

No OpenCV is used anywhere: the four methods (SSD, SAD, NCC, ZNCC) are
implemented directly, with ZNCC/NCC accelerated by FFT cross-correlation plus
integral images after Lewis (1995).  See ``docs/RESEARCH.md`` for the theory.
"""

from __future__ import annotations

from .correlation import (
    best_location,
    brute_force_map,
    integral_image,
    ncc_map,
    sad_map,
    score_map,
    ssd_map,
    zncc_map,
)
from .errors import (
    EmptyImageError,
    InvalidMethodError,
    InvalidParameterError,
    TemplateMatchingError,
    TemplateTooLargeError,
)
from .evaluate import evaluate_detections
from .matching import (
    MatchResult,
    iou,
    match_all,
    match_batch,
    match_multiscale,
    match_rotated,
    match_single,
    non_maximum_suppression,
    rotate_template,
)
from .metrics import METHODS, is_similarity, ncc, sad, ssd, zncc
from .synthetic import (
    SHAPES,
    Placement,
    add_gaussian_noise,
    add_occlusion,
    draw_shape,
    extract_template,
    generate_scene,
    make_template_image,
)
from .visualize import annotate, heatmap, save_heatmap

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "METHODS",
    "SHAPES",
    "MatchResult",
    "Placement",
    "TemplateMatchingError",
    "TemplateTooLargeError",
    "EmptyImageError",
    "InvalidMethodError",
    "InvalidParameterError",
    "ssd",
    "sad",
    "ncc",
    "zncc",
    "is_similarity",
    "integral_image",
    "zncc_map",
    "ncc_map",
    "ssd_map",
    "sad_map",
    "score_map",
    "brute_force_map",
    "best_location",
    "iou",
    "non_maximum_suppression",
    "match_single",
    "match_all",
    "match_multiscale",
    "match_rotated",
    "match_batch",
    "rotate_template",
    "evaluate_detections",
    "generate_scene",
    "make_template_image",
    "extract_template",
    "add_gaussian_noise",
    "add_occlusion",
    "draw_shape",
    "annotate",
    "heatmap",
    "save_heatmap",
]
