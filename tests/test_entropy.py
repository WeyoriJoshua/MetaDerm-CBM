import numpy as np
from metaderm.uncertainty.entropy import normalized_categorical_entropy, binary_entropy

def test_categorical_entropy_bounds():
    low=normalized_categorical_entropy([[1,0,0]])[0]
    high=normalized_categorical_entropy([[1/3,1/3,1/3]])[0]
    assert low < 1e-6
    assert 0.999 < high <= 1.001

def test_binary_entropy():
    assert binary_entropy([0.5])[0] > 0.999
    assert binary_entropy([0.999999])[0] < 0.001
