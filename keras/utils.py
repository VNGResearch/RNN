import numpy as np

SENTENCE_END_TOKEN = '$SENTENCE_END'
UNKNOWN_TOKEN = '$UNKNOWN'


def nearest_vector(array, value):
    idx = np.array([np.linalg.norm(x+y) for (x, y) in array-value]).argmin()
    return array[idx]


def nearest_vector_index(array, value):
    return np.array([np.linalg.norm(x+y) for (x, y) in array-value]).argmin()
