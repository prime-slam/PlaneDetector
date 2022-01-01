import numpy as np

from src.utils.colors import normalize_color, denormalize_color


def test_normalize_color():
    normalized_color = [0.047, 0.176, 0.494]
    color = [12, 45, 126]
    estimated_color = normalize_color(color)
    np.testing.assert_almost_equal(normalized_color, estimated_color, decimal=3)


def test_denormalize_color():
    normalized_color = [0.047, 0.176, 0.494]
    color = [12, 45, 126]
    estimated_color = denormalize_color(normalized_color)
    np.testing.assert_almost_equal(color, estimated_color)


def test_normalize_denormalize_color():
    color = [12, 45, 126]
    normalized_color = normalize_color(color)
    estimated_color = denormalize_color(normalized_color)
    np.testing.assert_almost_equal(color, estimated_color)
