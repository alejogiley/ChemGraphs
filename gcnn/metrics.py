import tensorflow as tf


def rsquared(y_true, y_pred):
    """Metric of degree of any linear correlation

    R_square is the proportion of the variance in the dependent variable
    that is predictable from the independent variable(s). Details here
    https://en.wikipedia.org/wiki/Coefficient_of_determination

    """
    # Estimators∏
    true = y_true[:, 2]
    pred = y_pred[:-1, 0]

    ##################################
    # Censored values
    ##################################

    # Get indicators censoring
    lefts_indexes = y_true[:, 0]
    right_indexes = y_true[:, 1]
    inner_indexes = (1 - right_indexes) * (1 - lefts_indexes)

    ##################################
    # R square
    ##################################

    mean = tf.reduce_mean(true * inner_indexes)
    sres = tf.reduce_sum(tf.square(true - pred) * inner_indexes)
    stot = tf.reduce_sum(tf.square(true - mean) * inner_indexes)

    return tf.subtract(1.0, tf.math.divide_no_nan(sres, stot))


def pearson(y_true, y_pred):
    """Measure of linear correlation

    It is the covariance of two variables, divided by
    the product of their standard deviations. The result
    always has a value between -1 and 1. Details here
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

    """
    # Estimators
    true = y_true[:, 2]
    pred = y_pred[:-1, 0]

    ##################################
    # Censored values
    ##################################

    lefts_indexes = y_true[:, 0]
    right_indexes = y_true[:, 1]
    inner_indexes = (1 - right_indexes) * (1 - lefts_indexes)

    ##################################
    # Pearson
    ##################################

    mean_x = tf.reduce_mean(true * inner_indexes)
    mean_y = tf.reduce_mean(pred * inner_indexes)

    mean_x2 = tf.reduce_mean(true * true * inner_indexes)
    mean_y2 = tf.reduce_mean(pred * pred * inner_indexes)
    mean_xy = tf.reduce_mean(true * pred * inner_indexes)

    num = (mean_xy - mean_x * mean_y) * (mean_xy - mean_x * mean_y)
    den = (mean_x2 - mean_x * mean_x) * (mean_y2 - mean_y * mean_y)

    return tf.math.divide_no_nan(num, den)
