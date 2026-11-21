"""Tests for the fast score maps (FFT + integral image).

Strategy: every fast map is checked against the independent brute-force
reference (:func:`template_matching.correlation.brute_force_map`) on small
seeded random images, plus targeted checks (exact template extraction must
score 1.0 for ZNCC, flat regions must not produce NaN).
"""

import numpy as np
import pytest

from template_matching.correlation import (
    brute_force_map,
    integral_image,
    ncc_map,
    sad_map,
    score_map,
    ssd_map,
    zncc_map,
)
from template_matching.errors import InvalidMethodError, TemplateTooLargeError

rng = np.random.default_rng(42)


def random_image(h=24, w=20):
    return rng.integers(0, 256, size=(h, w)).astype(float)


def test_integral_image_matches_naive_sums():
    img = random_image(10, 12)
    ii = integral_image(img)
    assert ii.shape == (11, 13)
    assert ii[0, 0] == 0.0
    assert ii[-1, -1] == pytest.approx(img.sum())
    # Rectangle (2, 3) .. (6, 8) exclusive of the far edge.
    expected = img[2:6, 3:8].sum()
    got = ii[6, 8] - ii[2, 8] - ii[6, 3] + ii[2, 3]
    assert got == pytest.approx(expected)


@pytest.mark.parametrize("method", ["zncc", "ncc", "ssd", "sad"])
def test_fast_map_matches_brute_force(method):
    img = random_image(24, 20)
    templ = random_image(5, 4)
    fast = score_map(img, templ, method)
    slow = brute_force_map(img, templ, method)
    assert fast.shape == slow.shape == (20, 17)
    assert np.allclose(fast, slow, atol=1e-8), method


def test_zncc_map_exact_template_scores_one():
    img = random_image(30, 28)
    templ = img[7:12, 9:15].copy()
    result = zncc_map(img, templ)
    assert result.shape == (26, 23)
    assert result[7, 9] == pytest.approx(1.0, abs=1e-9)
    assert result.max() == pytest.approx(1.0, abs=1e-9)
    y, x = np.unravel_index(int(np.argmax(result)), result.shape)
    assert (int(x), int(y)) == (9, 7)


def test_ncc_map_exact_template_scores_one():
    img = random_image(30, 28)
    templ = img[3:9, 4:10].copy()
    result = ncc_map(img, templ)
    assert result[3, 4] == pytest.approx(1.0, abs=1e-9)


def test_ssd_map_exact_template_scores_zero():
    img = random_image(30, 28)
    templ = img[3:9, 4:10].copy()
    result = ssd_map(img, templ)
    assert result[3, 4] == pytest.approx(0.0, abs=1e-6)
    assert result.min() == pytest.approx(0.0, abs=1e-6)


def test_sad_map_exact_template_scores_zero():
    img = random_image(30, 28)
    templ = img[3:9, 4:10].copy()
    result = sad_map(img, templ)
    assert result[3, 4] == pytest.approx(0.0)
    assert result.min() == pytest.approx(0.0)


def test_zncc_map_brightness_offset_invariant():
    # Keep pixel values low enough that +60 never clips at 255, so the
    # offset is a pure constant shift and ZNCC stays exactly invariant.
    img = rng.integers(0, 150, size=(30, 28)).astype(float)
    templ = img[7:12, 9:15].copy()
    bright = img + 60
    result = zncc_map(bright, templ)
    # Same geometry, only a constant offset: peak stays at the same place.
    y, x = np.unravel_index(int(np.argmax(result)), result.shape)
    assert (int(x), int(y)) == (9, 7)
    assert result[7, 9] == pytest.approx(1.0, abs=1e-9)


def test_zncc_map_flat_region_has_no_nan():
    img = np.full((20, 20), 128.0)
    img[5:10, 5:10] = 200.0
    templ = np.full((4, 4), 200.0)
    result = zncc_map(img, templ)
    assert not np.isnan(result).any()
    assert np.isfinite(result).all()


def test_score_map_range():
    img = random_image(24, 20)
    templ = random_image(5, 4)
    assert zncc_map(img, templ).min() >= -1.0 - 1e-9
    assert zncc_map(img, templ).max() <= 1.0 + 1e-9
    assert ncc_map(img, templ).min() >= -1e-9
    assert ssd_map(img, templ).min() >= 0.0
    assert sad_map(img, templ).min() >= 0.0


def test_template_larger_than_image_raises():
    with pytest.raises(TemplateTooLargeError):
        zncc_map(np.zeros((5, 5)), np.zeros((6, 6)))
    with pytest.raises(TemplateTooLargeError):
        ssd_map(np.zeros((5, 5)), np.zeros((5, 6)))


def test_unknown_method_raises():
    with pytest.raises(InvalidMethodError):
        score_map(np.zeros((8, 8)), np.zeros((3, 3)), "bogus")
