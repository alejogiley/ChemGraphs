import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def mse_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Mean Squared Error (MSE) for censored data

    In this implementation although the input predictions contain the
    estimated target affinities and the estimated errors variance,
    the predicted variance is ignored. This is a standard MSE where
    censored values are ignored.

    Details in https://en.wikipedia.org/wiki/Mean_squared_error

    Args:
        y_true: censored target
        y_pred: predictions & variance

    Returns:
        mean squared error

    """
    # Predictions
    pred = y_pred[:-1, 0]

    # Variance of error distribution
    sigma = tf.square(y_pred[-1:])

    # clip sigma values
    sigma = tf.clip_by_value(sigma, clip_value_min=1e-9, clip_value_max=1e9)

    ##################################
    # Censored values
    ##################################

    # Target affinities
    true = y_true[:, 2]

    # Get indicators censoring
    lefts_indexes = y_true[:, 0]
    right_indexes = y_true[:, 1]

    # Get indexes of not-, left- and right-censored values
    lefts_indexes = lefts_indexes * (1 - right_indexes)
    right_indexes = right_indexes * (1 - lefts_indexes)
    inner_indexes = (1 - right_indexes) * (1 - lefts_indexes)

    ##################################
    # full MSE loss
    ##################################

    # calculate loss
    loss = tf.reduce_sum(tf.square(true - pred) * inner_indexes)

    return loss


def maxlike_mse_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Maximum-likelihood MSE for censored data

    In this case a maximum likelihood estimate of MSE is used which takes
    into account the predicted variance of the errors distribution. When the
    variance is 1 the final result is equivalent to the standard MSE. Censored
    values are ignored.

    Details here https://en.wikipedia.org/wiki/Maximum_likelihood_estimation

    Args:
        y_true: censored target
        y_pred: predictions & variance

    Returns:
        mean squared error

    """
    # Predictions
    pred = y_pred[:-1, 0]

    # Variance of error distribution
    sigma = tf.square(y_pred[-1:])

    # clip sigma values
    sigma = tf.clip_by_value(sigma, clip_value_min=1e-9, clip_value_max=1e9)

    ##################################
    # Censored values
    ##################################

    # Target affinities
    true = y_true[:, 2]

    # Get indicators censoring
    lefts_indexes = y_true[:, 0]
    right_indexes = y_true[:, 1]

    # Get indexes of not-, left- and right-censored values
    lefts_indexes = lefts_indexes * (1 - right_indexes)
    right_indexes = right_indexes * (1 - lefts_indexes)
    inner_indexes = (1 - right_indexes) * (1 - lefts_indexes)

    ##################################
    # full MSE loss
    ##################################

    # calculate loss
    loss = tf.reduce_sum(
        tf.math.log(2 * sigma * np.pi) + tf.square(true - pred) / sigma * inner_indexes
    )

    return loss


def maxlike_cse_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Censored maximum-likelihood MSE for censored data

    For lef-censored values a MSE error is computed only when predictions
    are larger than the target value, equivalently for right-censored
    values when predictions are smaller than targets.

    Args:
        y_true: censored target
        y_pred: predictions & variance

    Returns:
        mean squared error

    """

    ##################################
    # Split predictions
    ##################################

    # Predictions
    pred = y_pred[:-1, 0]

    # Variance of error distribution
    sigma = tf.square(y_pred[-1:])

    # Clip values, avoid overflow
    # or sigma = 0 and then log-nan
    sigma = tf.clip_by_value(sigma, clip_value_min=1e-9, clip_value_max=1e9)

    ##################################
    # Censored values
    ##################################

    # Target affinities
    true = y_true[:, 2]

    # Get indicators censoring
    lefts_indexes = y_true[:, 0]
    right_indexes = y_true[:, 1]

    # Get indexes of not-, left- and right-censored values
    lefts_indexes = lefts_indexes * (1 - right_indexes)
    right_indexes = right_indexes * (1 - lefts_indexes)
    inner_indexes = (1 - right_indexes) * (1 - lefts_indexes)

    delta_inner = true - pred
    delta_right = tf.nn.relu(true - pred)
    delta_lefts = tf.nn.relu(pred - true)

    ##################################
    # censored MSE loss
    ##################################

    # calculate loss
    loss = (
        (tf.square(delta_inner) / sigma) * inner_indexes
        + (tf.square(delta_lefts) / sigma) * lefts_indexes
        + (tf.square(delta_right) / sigma) * right_indexes
    )

    loss = tf.reduce_sum(loss + tf.math.log(2 * sigma * np.pi))

    return loss


def maxlike_tobit_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Maximum-likelihood error for a Tobit model

    Details here https://en.wikipedia.org/wiki/Tobit_model

    Args:
        y_true: censored target
        y_pred: predictions & variance

    Returns:
        mean squared error

    """

    ##################################
    # Split predictions
    ##################################

    # Predictions
    pred = y_pred[:-1, 0]

    # Variance of error distribution
    sigma = tf.square(y_pred[-1:])

    # Clip values, avoid overflow
    # or sigma = 0 and then log-nan
    sigma = tf.clip_by_value(sigma, clip_value_min=1e-9, clip_value_max=1e9)

    ##################################
    # Censored values
    ##################################

    # Target affinities
    true = y_true[:, 2]

    # Get indicators censoring
    lefts_indexes = y_true[:, 0]
    right_indexes = y_true[:, 1]

    # Get indexes of not-, left- and right-censored values
    lefts_indexes = lefts_indexes * (1 - right_indexes)
    right_indexes = right_indexes * (1 - lefts_indexes)
    inner_indexes = (1 - right_indexes) * (1 - lefts_indexes)

    # normal distribution
    normal = tfp.distributions.Normal(loc=0.0, scale=1.0)

    ##################################
    # not-censored values - inner MSE
    ##################################

    # probability function of normal distribution at point y_pred
    inner_prob = normal.log_prob((true - pred) / sigma) - tf.math.log(sigma)

    ##################################
    # "<"-censored values - left MSE
    ##################################

    # probability of random variable being < than y_pred
    lefts_prob = normal.log_cdf((true - pred) / sigma)

    ##################################
    # ">"-censored values - right MSE
    ##################################

    # probability of point random variable being > than y_pred
    right_prob = normal.log_cdf((pred - true) / sigma)

    ##################################
    # log-likelihood
    ##################################

    # logarithm of likelihood
    logp = tf.reduce_sum(inner_prob * inner_indexes) \
        + tf.reduce_sum(lefts_prob * lefts_indexes) \
        + tf.reduce_sum(right_prob * right_indexes)

    return -logp
