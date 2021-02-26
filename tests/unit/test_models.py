import shutil
import tempfile

import numpy as np
import tensorflow as tf

from tensorflow.python.keras.engine import training
from spektral.data.dataset import Dataset
from spektral.data.graph import Graph

from gcnn.models import (
    train_model,
    MLEDense,
    create_gcnn
)


class TestTrainModel(tf.test.TestCase):
    """Test train_model routine"""
    
    def setUp(self):
        super(TestTrainModel, self).setUp()
        tf.random.set_seed(1)

        # simple dataset
        class TestDataset(Dataset):
            
            def read():
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
            learning_rate=0.1)

    def test_training_model(self):
        self.assertIsInstance(self.model, training.Model)
        
    def test_model_weights(self):
        target_label = np.array([0.09957985])
        target_sigma = np.array([[-0.25975102]])
        self.assertAllClose(self.model.weights[-2], target_label)
        self.assertAllClose(self.model.weights[-1], target_sigma)
        
    def test_training_history(self):
        target_history = {
            'loss': [1.4248809814453125], 
            'mape': [118.00852966308594]
        }
        self.assertDictEqual(
            self.history.history, target_history)


if __name__ == '__main__':
    tf.test.main()