import numpy as np


def invert(value):
    if isinstance(value, np.ndarray):
        return np.linalg.inv(value)
    elif value != 0:
        return 1.0 / value
    else:
        raise ValueError('Illegal division by zero in ivnersion!')


def rot_mat(al):
    return np.array([
        [np.cos(al), -np.sin(al)],
        [np.sin(al), np.cos(al)],
    ])
