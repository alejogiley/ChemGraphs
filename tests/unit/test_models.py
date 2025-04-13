import random
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import InputLayer
from tensorflow.python.keras.engine import training
from spektral.data.graph import Graph
from spektral.data.dataset import Dataset
from spektral.layers import ECCConv, GlobalAttnSumPool

from gcnn.models import train_model, create_gcnn


class TestTrainModel(tf.test.TestCase):
    """Test train_model routine"""

    def setUp(self):
        super(TestTrainModel, self).setUp()
        tf.random.set_seed(1)

        # simple dataset
        class TestDataset(Dataset):

            def read(self):
                return [
                    Graph(
                        x=np.ones((2, 1)),
                        a=np.identity(2),
                        e=np.ones((2, 2, 1)),
                        y=np.array([1]),
                    )
                ]

        # initialize
        dataset = TestDataset()

        # train model
        self.model, self.history = train_model(
            dataset,
            tf_loss="mse",
            metrics=["mape"],
            n_layers=1,
            channels=[1],
            batch_size=10,
            number_epochs=1,
            learning_rate=0.1,
        )

    def test_training_model(self):
        self.assertIsInstance(self.model, training.Model)

    def test_model_weights(self):
        target_label = np.array([0.09957985])
        target_sigma = np.array([[-0.25975102]])
        self.assertAllClose(self.model.weights[-2], target_label)
        self.assertAllClose(self.model.weights[-1], target_sigma)

    def test_training_history(self):
        target_history = {
            "loss": [1.4248809814453125],
            "mape": [118.00852966308594],
        }
        self.assertDictEqual(self.history.history, target_history)


class TestCreateGCNN(tf.test.TestCase):
    """Test create_gcnn routine"""

    def setUp(self):
        random.seed(1)
        np.random.seed(1)
        tf.random.set_seed(1)

    def test_create_graphconv_model(self):
        model = create_gcnn(nodes_shape=1,
                            edges_shape=1,
                            channels=[1],
                            n_layers=1)

        # check model layers
        self.assertIsInstance(model, training.Model)
        self.assertIsInstance(model.layers[0], InputLayer)
        self.assertIsInstance(model.layers[7], ECCConv)
        self.assertIsInstance(model.layers[9], GlobalAttnSumPool)

        # check model summary
        self.assertEqual(len(model.layers), 14)

    def test_model_call(self):
        model = create_gcnn(nodes_shape=1,
                            edges_shape=1,
                            channels=[1],
                            n_layers=1)

        y = np.array([[-0.03089559], [-0.35975078]])  # (mu, sigma)
        x = np.ones((1, 2, 1))  # (batch, n_nodes, n_node_features)
        a = np.identity(2)[None]  # (batch, n_nodes, n_nodes)
        e = np.ones((1, 2, 2, 1))  # (batch, n_nodes, n_nodes, n_edge_features)

        # check model call
        self.assertAllClose(model([x, a, e]), y)


if __name__ == "__main__":
    tf.test.main()
