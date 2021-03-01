import tensorflow as tf


def r_squared(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Estimate R-square metric

    R_square explains to what extent the variance of he predicted
    variable explains the variance of the target variable. Details here
    https://en.wikipedia.org/wiki/Coefficient_of_determination

    Args:
        y_true: target values
        y_pred: predicted values

    Returns:
        correlation metric

    """
    # Predictions
    pred = y_pred[:-1, 0]

    ##################################
    # Censored values
    ##################################
    
    # Target affinities
    true = y_true[:, 2]

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


def pearson(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Estimate Pearson cefficient

    The Pearson metric indicates the linear relationship between
    the predicted and target sets of data. Details here
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

    Args:
        y_true: target values
        y_pred: predicted values

    Returns:
        relationship metric

    """
    # Predictions
    print(y_pred)
    pred = y_pred[:-1, 0]

    ##################################
    # Censored values
    ##################################
    
    # Target affinities
    print(y_true)
    true = y_true[:, 2]

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
