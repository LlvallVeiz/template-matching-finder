"""Tests for the synthetic scene generator and the evaluator."""

import numpy as np
import pytest

from template_matching.evaluate import evaluate_detections
from template_matching.matching import MatchResult
from template_matching.synthetic import (
    SHAPES,
    Placement,
    add_gaussian_noise,
    add_occlusion,
    extract_template,
    generate_scene,
    make_template_image,
)


def test_make_template_image_shapes_are_distinct():
    renders = {shape: make_template_image(shape, 32) for shape in SHAPES}
    assert set(renders) == set(SHAPES)
    for shape, img in renders.items():
        assert img.shape == (32, 32)
        assert img.max() == 220.0  # shape value
        # Every shape except the solid square leaves background visible.
        if shape != "square":
            assert img.min() == 40.0  # background value
    # No two shapes render identically.
    keys = list(renders)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            assert not np.array_equal(renders[a], renders[b])


def test_make_template_image_unknown_shape_raises():
    with pytest.raises(Exception):
        make_template_image("hexagon", 32)


def test_generate_scene_is_deterministic_with_seed():
    kwargs = dict(
        placements=[Placement("cross", 120, 80, 48)],
        distractors=[Placement("circle", 300, 60, 40)],
        noise_sigma=15.0,
    )
    scene_a, gt_a = generate_scene(400, 300, seed=7, **kwargs)
    scene_b, gt_b = generate_scene(400, 300, seed=7, **kwargs)
    assert np.array_equal(scene_a, scene_b)
    assert gt_a == gt_b
    scene_c, _ = generate_scene(400, 300, seed=8, **kwargs)
    assert not np.array_equal(scene_a, scene_c)


def test_generate_scene_ground_truth_matches_placement():
    scene, gt = generate_scene(
        400, 300,
        placements=[Placement("ring", 50, 60, 56), Placement("cross", 200, 150, 48)],
        seed=2,
    )
    assert gt == [
        {"shape": "ring", "x": 50, "y": 60, "w": 56, "h": 56},
        {"shape": "cross", "x": 200, "y": 150, "w": 48, "h": 48},
    ]
    assert scene.shape == (300, 400)


def test_noise_increases_pixel_std():
    clean, _ = generate_scene(200, 200, seed=1, noise_sigma=0.0)
    noisy, _ = generate_scene(200, 200, seed=1, noise_sigma=25.0)
    assert noisy.std() > clean.std()


def test_occlusion_erases_part_of_shape():
    scene, gt = generate_scene(400, 300, placements=[Placement("square", 100, 100, 48)], seed=1)
    occluded = add_occlusion(scene, (100, 100, 48, 48), fraction=0.5)
    # Right half of the square (value 220) becomes background (40).
    assert occluded[110, 130] == 40.0
    assert occluded[110, 110] == 220.0
    with pytest.raises(Exception):
        add_occlusion(scene, (100, 100, 48, 48), fraction=1.5)


def test_extract_template_round_trip():
    scene, _ = generate_scene(400, 300, placements=[Placement("cross", 120, 80, 48)], seed=1)
    template = extract_template(scene, (120, 80, 48, 48))
    assert template.shape == (48, 48)
    assert np.array_equal(template, scene[80:128, 120:168])


def test_add_gaussian_noise_zero_sigma_is_identity():
    img = np.full((10, 10), 100.0)
    rng = np.random.default_rng(0)
    assert np.array_equal(add_gaussian_noise(img, 0.0, rng), img)


def _det(x, y, score=0.99, method="zncc"):
    return MatchResult(x, y, 48, 48, score, method)


def test_evaluate_perfect_detections():
    gt = [{"shape": "cross", "x": 120, "y": 80, "w": 48, "h": 48}]
    report = evaluate_detections([_det(120, 80)], gt)
    assert (report["tp"], report["fp"], report["fn"]) == (1, 0, 0)
    assert (report["precision"], report["recall"], report["f1"]) == (1.0, 1.0, 1.0)


def test_evaluate_miss_and_false_positive():
    gt = [
        {"shape": "cross", "x": 120, "y": 80, "w": 48, "h": 48},
        {"shape": "cross", "x": 250, "y": 180, "w": 48, "h": 48},
    ]
    report = evaluate_detections([_det(120, 80), _det(10, 10)], gt)
    assert (report["tp"], report["fp"], report["fn"]) == (1, 1, 1)
    assert report["precision"] == pytest.approx(0.5)
    assert report["recall"] == pytest.approx(0.5)
    assert report["f1"] == pytest.approx(0.5)


def test_evaluate_iou_threshold_matters():
    # Detection shifted by 24 px: IoU = (24*48)/(48*48+48*48-24*48) = 1152/3456 = 1/3.
    gt = [{"shape": "cross", "x": 120, "y": 80, "w": 48, "h": 48}]
    det = _det(144, 80)
    assert evaluate_detections([det], gt, iou_threshold=0.5)["tp"] == 0
    assert evaluate_detections([det], gt, iou_threshold=0.25)["tp"] == 1


def test_evaluate_distance_method_greedy_order():
    # For SSD the lower score is the better detection; it must claim the box.
    gt = [{"shape": "cross", "x": 120, "y": 80, "w": 48, "h": 48}]
    dets = [_det(300, 250, score=900.0, method="ssd"), _det(120, 80, score=10.0, method="ssd")]
    report = evaluate_detections(dets, gt)
    assert report["tp"] == 1
    assert report["verdicts"][0]["verdict"] == "tp"


def test_evaluate_empty_detections():
    gt = [{"shape": "cross", "x": 120, "y": 80, "w": 48, "h": 48}]
    report = evaluate_detections([], gt)
    assert (report["tp"], report["fp"], report["fn"]) == (0, 0, 1)
    assert report["precision"] == 0.0 and report["recall"] == 0.0
