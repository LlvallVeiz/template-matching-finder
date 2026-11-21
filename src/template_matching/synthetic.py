"""Synthetic scene generator: geometric shapes, noise and partial occlusion.

Real labelled data is expensive; synthetic scenes with *known* ground truth
are the standard way to develop and evaluate a template matcher (Brunelli
2009 calls them "synthetic oracles").  Every scene this module builds comes
with exact bounding boxes, so detection quality can be measured with IoU /
precision / recall instead of eyeballed.

Scenes are grayscale ``uint8`` arrays: a dark background, bright shapes, and
optional Gaussian noise plus a rectangular occluder.  Five shape kinds are
available; each has a distinct silhouette so that templates do not
accidentally match the wrong shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from .errors import InvalidParameterError

Array = NDArray[np.float64]

#: Shape kinds the generator can draw.
SHAPES = ("cross", "circle", "ring", "triangle", "square", "diamond")

BACKGROUND = 40
SHAPE_VALUE = 220
DISTRACTOR_VALUE = 160


@dataclass
class Placement:
    """One shape placed in a scene: kind, top-left corner and size."""

    shape: str
    x: int
    y: int
    size: int

    def box(self) -> tuple[int, int, int, int]:
        """Return the bounding box as ``(x, y, w, h)``."""
        return (self.x, self.y, self.size, self.size)


def draw_shape(draw: ImageDraw.ImageDraw, shape: str, box: tuple[int, int, int, int], value: int) -> None:
    """Draw one filled shape inside ``box`` = ``(x0, y0, x1, y1)`` inclusive."""
    x0, y0, x1, y1 = box
    if shape == "square":
        draw.rectangle([x0, y0, x1, y1], fill=value)
    elif shape == "circle":
        draw.ellipse([x0, y0, x1, y1], fill=value)
    elif shape == "ring":
        draw.ellipse([x0, y0, x1, y1], fill=value)
        inner = (
            x0 + (x1 - x0) // 3,
            y0 + (y1 - y0) // 3,
            x1 - (x1 - x0) // 3,
            y1 - (y1 - y0) // 3,
        )
        draw.ellipse(inner, fill=BACKGROUND)
    elif shape == "triangle":
        draw.polygon(
            [(x0 + (x1 - x0) // 2, y0), (x0, y1), (x1, y1)], fill=value
        )
    elif shape == "diamond":
        draw.polygon(
            [
                (x0 + (x1 - x0) // 2, y0),
                (x1, y0 + (y1 - y0) // 2),
                (x0 + (x1 - x0) // 2, y1),
                (x0, y0 + (y1 - y0) // 2),
            ],
            fill=value,
        )
    elif shape == "cross":
        w = x1 - x0 + 1
        arm = max(2, w // 3)
        cx, cy = x0 + w // 2, y0 + w // 2
        draw.rectangle([cx - arm // 2, y0, cx + arm // 2, y1], fill=value)
        draw.rectangle([x0, cy - arm // 2, x1, cy + arm // 2], fill=value)
    else:
        raise InvalidParameterError(f"unknown shape {shape!r}; expected one of {SHAPES}")


def make_template_image(shape: str, size: int = 48, value: int = SHAPE_VALUE) -> Array:
    """Render one shape alone on the standard background (float64 array)."""
    if shape not in SHAPES:
        raise InvalidParameterError(f"unknown shape {shape!r}; expected one of {SHAPES}")
    if size < 8:
        raise InvalidParameterError("template size must be >= 8")
    img = Image.new("L", (size, size), BACKGROUND)
    draw_shape(ImageDraw.Draw(img), shape, (0, 0, size - 1, size - 1), value)
    return np.asarray(img, dtype=np.float64)


def add_gaussian_noise(image: Array, sigma: float, rng: np.random.Generator) -> Array:
    """Add zero-mean Gaussian noise with standard deviation ``sigma``."""
    if sigma < 0:
        raise InvalidParameterError("noise sigma must be >= 0")
    if sigma == 0:
        return image.copy()
    noisy = image + rng.normal(0.0, sigma, size=image.shape)
    return np.clip(noisy, 0, 255)


def add_occlusion(
    image: Array, box: tuple[int, int, int, int], fraction: float = 0.3
) -> Array:
    """Cover the right ``fraction`` of ``box`` with a background-coloured bar.

    The occluder has the background value, i.e. it erases part of the shape
    instead of adding a new object — the hard case for correlation matching.
    """
    if not 0.0 <= fraction <= 1.0:
        raise InvalidParameterError("occlusion fraction must be in [0, 1]")
    out = image.copy()
    x, y, w, h = box
    cut = int(round(w * fraction))
    if cut > 0:
        out[y : y + h, x + w - cut : x + w] = BACKGROUND
    return out


def generate_scene(
    width: int = 400,
    height: int = 300,
    placements: list[Placement] | None = None,
    distractors: list[Placement] | None = None,
    seed: int = 0,
    noise_sigma: float = 0.0,
    occlude: tuple[int, int, int, int] | None = None,
    occlusion_fraction: float = 0.3,
) -> tuple[Array, list[dict]]:
    """Build a synthetic scene and its ground truth.

    Parameters
    ----------
    placements:
        Shapes that count as ground-truth instances (their exact template is
        the crop of this scene).  Defaults to one cross at (120, 80).
    distractors:
        Extra shapes drawn in a dimmer value; they must *not* be detected as
        the target shapes.
    seed:
        Random seed for the noise, so scenes are reproducible.
    noise_sigma:
        Standard deviation of the added Gaussian noise (0 = clean).
    occlude:
        Box ``(x, y, w, h)`` whose right part is erased; use together with
        ``occlusion_fraction``.

    Returns
    -------
    (image, ground_truth): the scene as a float64 array and a list of
    ``{"shape": ..., "x": ..., "y": ..., "w": ..., "h": ...}`` dicts.
    """
    if width < 32 or height < 32:
        raise InvalidParameterError("scene must be at least 32x32")
    placements = placements if placements is not None else [Placement("cross", 120, 80, 48)]
    rng = np.random.default_rng(seed)
    img = Image.new("L", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(img)
    for p in distractors or []:
        draw_shape(draw, p.shape, (p.x, p.y, p.x + p.size - 1, p.y + p.size - 1), DISTRACTOR_VALUE)
    for p in placements:
        draw_shape(draw, p.shape, (p.x, p.y, p.x + p.size - 1, p.y + p.size - 1), SHAPE_VALUE)
    scene = np.asarray(img, dtype=np.float64)
    if occlude is not None:
        scene = add_occlusion(scene, occlude, occlusion_fraction)
    scene = add_gaussian_noise(scene, noise_sigma, rng)
    ground_truth = [
        {"shape": p.shape, "x": p.x, "y": p.y, "w": p.size, "h": p.size}
        for p in placements
    ]
    return scene, ground_truth


def extract_template(image: Array, box: tuple[int, int, int, int]) -> Array:
    """Crop the ``(x, y, w, h)`` window out of ``image``."""
    x, y, w, h = box
    return np.asarray(image, dtype=np.float64)[y : y + h, x : x + w].copy()
