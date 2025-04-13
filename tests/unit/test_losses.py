import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.framework.errors_impl import InvalidArgumentError

from gcnn.losses import mse_loss, maxlike_mse_loss, maxlike_cse_loss, maxlike_tobit_loss


class TestMSELoss(tf.test.TestCase):

    def test_mse_loss_good_input(self):
        # shape = (batch_size, (mu, sigma))
        y_pred = tf.constant([[1.0, 0.0], [2.0, 0.0]])
        # shape = (batch_size, (left, right, label))
        y_true = tf.constant([[0, 0, 100], [0, 0, 3.5]])

        expect_loss = mse_loss(y_true, y_pred)
        target_loss = tf.reduce_mean(tf.square(y_true[:, 2] - y_pred[:, 0]))
        self.assertAllClose(target_loss, expect_loss)

    def test_mse_loss_wrong_input(self):
        y_true = tf.constant([1.0, 2.0, 3.0])
        y_pred = tf.constant([1.5, 2.5, 3.5, 4.5])
        with self.assertRaises(InvalidArgumentError):
            mse_loss(y_true, y_pred)

    def test_maxlike_mse_loss(self):
        # shape = (batch_size, (mu, sigma))
        y_pred = tf.constant([[1.0, 1.0], [2.0, 1.0]])
        # shape = (batch_size, (left, right, label))
        y_true = tf.constant([[0, 0, 100], [0, 0, 3.5]])

        expect_loss = maxlike_mse_loss(y_true, y_pred)
        target_loss = tf.reduce_mean(
            tf.square(y_true[:, 2] - y_pred[:, 0])) + tf.math.log(2.0 * np.pi)
        self.assertAllClose(target_loss, expect_loss)

    def test_maxlike_cse_loss(self):
        # shape = (batch_size, (mu, sigma))
        y_pred = tf.constant([[1.0, 1.0], [2.0, 1.0]])
        # shape = (batch_size, (left, right, label))
        y_true = tf.constant([[1, 0, 100], [1, 0, 3.5]])

        expect_loss = maxlike_cse_loss(y_true, y_pred)
        target_loss = tf.reduce_sum(tf.square(y_true - y_pred))

    # def test_maxlike_tobit_loss(self):
    #     y_true = tf.constant([1.0, 2.0, 3.0])
    #     y_pred = tf.constant([1.5, 2.5, 3.5])
    #     loss = maxlike_tobit_loss(y_true, y_pred)
    #     # Add your expected loss calculation here
    #     # expected_loss = ...
    #     # self.assertAllClose(loss, expected_loss)


if __name__ == '__main__':
    tf.test.main()
