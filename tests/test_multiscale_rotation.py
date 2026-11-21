"""Tests for multi-scale pyramid search and discrete-rotation search."""

import numpy as np
import pytest
from PIL import Image

from template_matching.matching import match_multiscale, match_rotated, rotate_template
from template_matching.synthetic import (
    Placement,
    extract_template,
    generate_scene,
    make_template_image,
)


def test_rotate_template_multiples_of_90_are_exact():
    templ = make_template_image("triangle", 32)
    assert np.array_equal(rotate_template(templ, 0), templ)
    assert rotate_template(templ, 90).shape == (32, 32)
    # Four quarter-turns return the original.
    back = rotate_template(rotate_template(rotate_template(rotate_template(templ, 90), 90), 90), 90)
    assert np.array_equal(back, templ)
    # 180 twice returns the original.
    assert np.array_equal(rotate_template(rotate_template(templ, 180), 180), templ)


def test_rotate_template_arbitrary_angle_grows_canvas():
    templ = make_template_image("cross", 48)
    rotated = rotate_template(templ, 45)
    assert rotated.shape[0] > 48 and rotated.shape[1] > 48


def test_match_rotated_finds_90_degree_copy():
    # Scene contains a triangle rotated 90 degrees; the template is upright.
    upright = make_template_image("triangle", 48)
    rotated = rotate_template(upright, 90)
    scene = np.full((300, 400), 40.0)
    scene[100:148, 150:198] = rotated
    found = match_rotated(scene, upright, method="zncc", angles=(0, 90, 180, 270))
    assert len(found) >= 1
    best = found[0]
    assert (best.x, best.y) == (150, 100)
    assert best.angle == pytest.approx(90.0)
    assert best.score == pytest.approx(1.0, abs=1e-9)


def test_match_rotated_upright_needs_no_rotation():
    # Note: a solid square has zero variance, so ZNCC cannot match it;
    # use a cross, whose template carries real texture.
    scene, gt = generate_scene(
        400, 300,
        placements=[Placement("cross", 90, 70, 48)],
        seed=11,
    )
    template = extract_template(scene, (90, 70, 48, 48))
    found = match_rotated(scene, template, method="zncc", angles=(0, 90, 180, 270))
    assert len(found) == 1
    assert (found[0].x, found[0].y) == (90, 70)
    assert found[0].angle == pytest.approx(0.0)


def test_match_multiscale_finds_double_size_object():
    # Template is 48 px; the object in the scene is the same shape at 96 px.
    template = make_template_image("cross", 48)
    big = np.asarray(
        Image.fromarray(np.clip(template, 0, 255).astype(np.uint8)).resize(
            (96, 96), Image.LANCZOS
        ),
        dtype=float,
    )
    scene = np.full((300, 400), 40.0)
    scene[80:176, 120:216] = big
    found = match_multiscale(
        scene, template, method="zncc", scales=(1.0, 0.5), threshold=0.8
    )
    assert len(found) >= 1
    best = found[0]
    # Box should cover the 96x96 object at (120, 80) within a few pixels.
    assert abs(best.x - 120) <= 2
    assert abs(best.y - 80) <= 2
    assert abs(best.w - 96) <= 2
    assert best.scale == pytest.approx(0.5)
    assert best.score > 0.9


def test_match_multiscale_scale_one_behaves_like_plain_match():
    scene, gt = generate_scene(
        400, 300, placements=[Placement("diamond", 140, 90, 48)], seed=13
    )
    template = extract_template(scene, (140, 90, 48, 48))
    found = match_multiscale(scene, template, method="zncc", scales=(1.0,))
    assert len(found) == 1
    assert (found[0].x, found[0].y) == (140, 90)
    assert found[0].scale == pytest.approx(1.0)


def test_match_multiscale_invalid_scales_raise():
    scene, _ = generate_scene(200, 200, seed=1)
    template = extract_template(scene, (10, 10, 20, 20))
    with pytest.raises(Exception):
        match_multiscale(scene, template, scales=())
    with pytest.raises(Exception):
        match_multiscale(scene, template, scales=(0.0, 1.0))
