"""Visual output: annotated detection images and correlation heatmaps.

Annotated images draw detections as red boxes with their score and, when
ground truth is supplied, the true boxes in green — so a glance at the PNG
shows whether a detection is a hit or a false alarm.  Heatmaps render a
score map with a perceptually ordered colormap and a colorbar, which makes
the correlation peak (and its runner-up lobes) visible.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from .matching import MatchResult

Array = NDArray[np.float64]


def _to_pil(image: Array) -> Image.Image:
    arr = np.asarray(image)
    if arr.ndim == 3:
        return Image.fromarray(arr.astype(np.uint8), mode="RGB")
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L").convert("RGB")


def annotate(
    image: Array,
    detections: list[MatchResult],
    ground_truth: list[dict] | None = None,
) -> Image.Image:
    """Draw red detection boxes (with scores) and green ground-truth boxes."""
    canvas = _to_pil(image)
    draw = ImageDraw.Draw(canvas)
    for g in ground_truth or []:
        x, y, w, h = g["x"], g["y"], g["w"], g["h"]
        draw.rectangle([x, y, x + w - 1, y + h - 1], outline=(0, 200, 0), width=2)
    for det in detections:
        x, y, w, h = det.x, det.y, det.w, det.h
        draw.rectangle([x, y, x + w - 1, y + h - 1], outline=(255, 0, 0), width=2)
        label = f"{det.template_name or det.method} {det.score:.3f}"
        if det.angle:
            label += f" {det.angle:.0f}d"
        if det.scale != 1.0:
            label += f" x{det.scale:g}"
        draw.text((x + 2, max(0, y - 12)), label, fill=(255, 0, 0))
    return canvas


def heatmap(score: Array, method: str, title: str = "") -> "matplotlib.figure.Figure":
    """Render a score map as a matplotlib figure with a colorbar.

    Similarity maps use ``viridis`` (bright = good match); distance maps use
    ``viridis_r`` so bright still means "good match".  The caller saves the
    returned figure (``fig.savefig(path, dpi=...)``).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .metrics import is_similarity

    cmap = "viridis" if is_similarity(method) else "viridis_r"
    fig, ax = plt.subplots(figsize=(7, 5))
    shown = ax.imshow(score, cmap=cmap, aspect="auto")
    ax.set_xlabel("x (template top-left)")
    ax.set_ylabel("y (template top-left)")
    ax.set_title(title or f"{method} score map")
    fig.colorbar(shown, ax=ax, label="score")
    fig.tight_layout()
    return fig


def save_heatmap(score: Array, method: str, path: str, title: str = "") -> str:
    """Render :func:`heatmap` and save it to ``path`` (PNG)."""
    fig = heatmap(score, method, title)
    fig.savefig(path, dpi=120)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path
