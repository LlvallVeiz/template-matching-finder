"""Tests for single/multi/batch matching, IoU and NMS."""

import numpy as np
import pytest

from template_matching.errors import InvalidMethodError, TemplateTooLargeError
from template_matching.matching import (
    MatchResult,
    iou,
    match_all,
    match_batch,
    match_single,
    non_maximum_suppression,
)
from template_matching.synthetic import Placement, extract_template, generate_scene


def scene_with_crosses():
    """400x300 scene: crosses at (120, 80), (250, 180) and (60, 200)."""
    placements = [
        Placement("cross", 120, 80, 48),
        Placement("cross", 250, 180, 48),
        Placement("cross", 60, 200, 48),
    ]
    distractors = [Placement("circle", 300, 60, 40), Placement("triangle", 180, 220, 44)]
    scene, gt = generate_scene(
        400, 300, placements=placements, distractors=distractors, seed=3
    )
    template = extract_template(scene, (120, 80, 48, 48))
    return scene, template, gt


def test_iou_identical_is_one():
    assert iou((10, 10, 20, 20), (10, 10, 20, 20)) == pytest.approx(1.0)


def test_iou_disjoint_is_zero():
    assert iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0


def test_iou_half_overlap_hand_computed():
    # Intersection 5x10=50, union 100+100-50=150 -> 1/3.
    assert iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(1 / 3)


def test_nms_keeps_best_and_suppresses_overlap():
    dets = [
        MatchResult(10, 10, 20, 20, 0.99, "zncc"),
        MatchResult(12, 11, 20, 20, 0.95, "zncc"),
        MatchResult(100, 100, 20, 20, 0.90, "zncc"),
    ]
    kept = non_maximum_suppression(dets, iou_threshold=0.5)
    assert len(kept) == 2
    assert kept[0].score == pytest.approx(0.99)
    assert kept[1].x == 100


def test_nms_distance_method_keeps_lowest():
    dets = [
        MatchResult(10, 10, 20, 20, 500.0, "ssd"),
        MatchResult(12, 11, 20, 20, 900.0, "ssd"),
        MatchResult(100, 100, 20, 20, 100.0, "ssd"),
    ]
    kept = non_maximum_suppression(dets, iou_threshold=0.5)
    assert len(kept) == 2
    assert kept[0].score == pytest.approx(100.0)
    assert kept[1].score == pytest.approx(500.0)


def test_nms_empty_input():
    assert non_maximum_suppression([]) == []


def test_nms_invalid_iou_threshold():
    with pytest.raises(Exception):
        non_maximum_suppression([MatchResult(0, 0, 5, 5, 1.0, "zncc")], iou_threshold=1.5)


@pytest.mark.parametrize("method", ["zncc", "ncc", "ssd", "sad"])
def test_match_single_finds_exact_location(method):
    scene, template, _ = scene_with_crosses()
    result = match_single(scene, template, method=method)
    # Three identical crosses exist; any of them is a perfect match.
    assert (result.x, result.y) in {(120, 80), (250, 180), (60, 200)}
    assert (result.w, result.h) == (48, 48)
    if method in ("zncc", "ncc"):
        assert result.score == pytest.approx(1.0, abs=1e-9)
    else:
        assert result.score == pytest.approx(0.0, abs=1e-6)


def test_match_all_finds_three_copies():
    scene, template, _ = scene_with_crosses()
    found = match_all(scene, template, method="zncc", threshold=0.8)
    assert len(found) == 3
    assert {(d.x, d.y) for d in found} == {(120, 80), (250, 180), (60, 200)}
    assert all(d.score == pytest.approx(1.0, abs=1e-9) for d in found)


def test_match_all_threshold_filters_weak_matches():
    scene, template, _ = scene_with_crosses()
    found = match_all(scene, template, method="zncc", threshold=0.8)
    assert len(found) == 3
    # A similarity threshold outside [-1, 1] is rejected.
    with pytest.raises(Exception):
        match_all(scene, template, method="zncc", threshold=1.5)


def test_match_all_max_results_limits_output():
    scene, template, _ = scene_with_crosses()
    found = match_all(scene, template, method="zncc", threshold=0.8, max_results=2)
    assert len(found) == 2


def test_match_single_template_equals_image():
    img = np.random.default_rng(1).integers(0, 256, size=(40, 50)).astype(float)
    result = match_single(img, img.copy(), method="zncc")
    assert (result.x, result.y) == (0, 0)
    assert result.score == pytest.approx(1.0, abs=1e-9)


def test_match_single_template_too_large_raises():
    with pytest.raises(TemplateTooLargeError):
        match_single(np.zeros((10, 10)), np.zeros((11, 10)))


def test_match_single_unknown_method_raises():
    with pytest.raises(InvalidMethodError):
        match_single(np.zeros((10, 10)), np.zeros((3, 3)), method="bogus")


def test_match_batch_two_templates():
    scene, gt = generate_scene(
        400, 300,
        placements=[Placement("cross", 120, 80, 48), Placement("ring", 260, 170, 56)],
        distractors=[Placement("triangle", 60, 220, 40)],
        seed=5,
    )
    templates = {
        "cross": extract_template(scene, (120, 80, 48, 48)),
        "ring": extract_template(scene, (260, 170, 56, 56)),
    }
    results = match_batch(scene, templates, method="zncc", threshold=0.8)
    assert set(results) == {"cross", "ring"}
    assert len(results["cross"]) == 1
    assert (results["cross"][0].x, results["cross"][0].y) == (120, 80)
    assert len(results["ring"]) == 1
    assert (results["ring"][0].x, results["ring"][0].y) == (260, 170)


def test_match_batch_oversized_template_yields_empty_list():
    scene, _ = generate_scene(
        100, 100, placements=[Placement("cross", 20, 20, 48)], seed=1
    )
    results = match_batch(
        scene,
        {"ok": extract_template(scene, (20, 20, 48, 48)), "huge": np.zeros((200, 200))},
        method="zncc",
    )
    assert results["huge"] == []
    assert len(results["ok"]) >= 1
