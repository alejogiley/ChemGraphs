import numpy as np
import tensorflow as tf

from keras.utils import to_categorical


def str_is_float(s):
    """Check whether string
    can be converted to a float
    """
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
    return matrix \
        + np.transpose(matrix, (1, 0, 2)) \
        - np.diag(matrix.diagonal())


def onehot_encoding(x, keys):
    maps = dict([(k, v) for k, v in zip(keys, range(len(keys)))])
    return to_categorical(maps[x], num_classes=len(keys))


def sigma(y_true, y_pred):
    """Returns variance of error distribution"""
    return tf.abs(y_pred[-1])
