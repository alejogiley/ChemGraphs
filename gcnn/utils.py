import random
import numpy as np
import tensorflow as tf

from tensorflow.keras.utils import to_categorical


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
    return matrix + np.transpose(matrix, (1, 0, 2)) - np.diag(matrix.diagonal())


def onehot_encoding(index: int, num_classes: int) -> np.ndarray:
    """Generate one-hot encoded vector from categorical data"""
    return to_categorical(index, num_classes=num_classes)


def set_random_seed(seed):
    """Control reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
