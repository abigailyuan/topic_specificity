import numpy as np
from topic_specificity import normalize_vector

def test_normalize_vector_sum_to_one():
    vec = np.array([1,2,3])
    assert np.isclose(normalize_vector(vec).sum(), 1.0)

def test_normalize_vector_all_zeros():
    vec = np.zeros(3)
    assert (normalize_vector(vec) == 0).all()