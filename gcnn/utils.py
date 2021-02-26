import numpy as np
import tensorflow as tf

from keras.utils import to_categorical
from typing import List


def str_is_float(s: str) -> bool:
    """Check string can be converted to float"""
    try:
        float(s)
        return True

    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True

    except (TypeError, ValueError):
        pass

    return False


def symmetrize(matrix):
    """Symmetrize multidimensional matrix"""
    return matrix \
        + np.transpose(matrix, (1, 0, 2)) \
        - np.diag(matrix.diagonal())


def onehot_encoding(x: np.array, categories: List[str]) -> np.ndarray:
    """Generate one-hot encoded vector from categorical data"""
    maps = dict([(k, v) for k, v in zip(categories, enumerate(categories))])
    return to_categorical(maps[x], num_classes=len(categories))


def sigma(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Returns variance of error distribution"""
    return tf.abs(y_pred[-1])
