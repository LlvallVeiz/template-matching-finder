"""Tests for the pointwise metrics — every value below is hand-computed.

Reference matrices used throughout (checked by hand):

A = [[1, 2], [3, 4]]          B = A + 1 = [[2, 3], [4, 5]]
  SSD(A, B) = 1+1+1+1 = 4,    SAD(A, B) = 4
  NCC(A, B) = 40 / sqrt(30 * 54) = 40 / sqrt(1620) = 0.993808...
  ZNCC(A, B) = 1.0 (B is A shifted by a constant)

C = [[0, 1], [2, 3]]          D = [[3, 2], [1, 0]]
  SSD(C, D) = 9+1+1+9 = 20,   SAD(C, D) = 3+1+1+3 = 8
  NCC(C, D) = 4 / sqrt(14 * 14) = 4/14 = 0.285714...
  ZNCC(C, D) = -5 / 5 = -1.0 (D is C with both rows reversed in value)
"""

import numpy as np
import pytest

from template_matching.errors import EmptyImageError
from template_matching.metrics import is_similarity, ncc, sad, ssd, zncc

A = np.array([[1, 2], [3, 4]], dtype=float)
B = np.array([[2, 3], [4, 5]], dtype=float)
C = np.array([[0, 1], [2, 3]], dtype=float)
D = np.array([[3, 2], [1, 0]], dtype=float)


def test_ssd_identical_is_zero():
    assert ssd(A, A) == 0.0


def test_ssd_hand_computed():
    assert ssd(A, B) == 4.0
    assert ssd(C, D) == 20.0


def test_sad_hand_computed():
    assert sad(A, A) == 0.0
    assert sad(A, B) == 4.0
    assert sad(C, D) == 8.0


def test_ncc_identical_is_one():
    assert ncc(A, A) == pytest.approx(1.0)


def test_ncc_hand_computed():
    assert ncc(A, B) == pytest.approx(40 / np.sqrt(1620))
    assert ncc(C, D) == pytest.approx(4 / 14)


def test_zncc_identical_is_one():
    assert zncc(A, A) == pytest.approx(1.0)


def test_zncc_offset_invariant():
    # Adding a constant brightness offset must not change ZNCC.
    assert zncc(A, B) == pytest.approx(1.0)


def test_zncc_scale_invariant():
    # A positive affine change (2*A + 10) must not change ZNCC.
    assert zncc(A, 2 * A + 10) == pytest.approx(1.0)


def test_zncc_anticorrelated_is_minus_one():
    assert zncc(C, D) == pytest.approx(-1.0)


def test_ncc_zero_patch_returns_zero_not_nan():
    zeros = np.zeros((2, 2))
    assert ncc(zeros, A) == 0.0
    assert not np.isnan(ncc(zeros, A))


def test_zncc_constant_patch_returns_zero_not_nan():
    const = np.full((2, 2), 7.0)
    assert zncc(const, A) == 0.0
    assert zncc(A, const) == 0.0


def test_ssd_is_symmetric():
    assert ssd(A, B) == ssd(B, A)
    assert sad(A, B) == sad(B, A)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        ssd(A, np.ones((3, 3)))


def test_empty_patch_raises():
    with pytest.raises(EmptyImageError):
        ssd(np.array([[]]), np.array([[]]))


def test_is_similarity_direction():
    assert is_similarity("zncc") is True
    assert is_similarity("ncc") is True
    assert is_similarity("ssd") is False
    assert is_similarity("sad") is False
    with pytest.raises(Exception):
        is_similarity("bogus")
