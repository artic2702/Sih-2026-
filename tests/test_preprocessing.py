"""
Tests for preprocessing utilities. Owner: Person 1, extend as functions are
implemented (many are still NotImplementedError stubs by design).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from src.preprocessing.normalize import select_bands


def test_select_bands_shape():
    arr = np.random.rand(4, 32, 32)  # 4-band image
    out = select_bands(arr, [0, 1, 2])
    assert out.shape == (3, 32, 32)


def test_select_bands_single_band():
    arr = np.random.rand(4, 32, 32)
    out = select_bands(arr, [3])
    assert out.shape == (1, 32, 32)
