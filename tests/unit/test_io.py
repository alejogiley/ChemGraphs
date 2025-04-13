import os
import shutil
import tempfile

import numpy as np
import pathlib as pl
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense
from spektral.data import BatchLoader
from spektral.data.dataset import Dataset
from spektral.data.graph import Graph
from spektral.layers import ECCConv, GlobalSumPool

from gcnn.io import save_gcnn, save_history


class TestSaveHistory(tf.test.TestCase):
    """Check save_history routine"""

    def setUp(self):
        super(TestSaveHistory, self).setUp()
        tf.random.set_seed(1)

        # temporary directory
        self.tmpdir = tempfile.mkdtemp()
        # temporary file
        self.tmpdat = os.path.join(self.tmpdir, "history.dat")

        # simple dataset
        class TestDataset(Dataset):
            def read(self):
                return [
                    Graph(
                        x=np.ones((2, 1)),
                        a=np.ones((2, 2)),
                        e=np.ones((2, 2, 1)),
                        y=np.array([1.0]),
                    )
                ]

        # initialize
        loader = BatchLoader(TestDataset(), batch_size=1, epochs=1, shuffle=False)

        # dummy model
        nodes = Input(shape=(None, 1))
        adjcs = Input(shape=(None, None))
        edges = Input(shape=(None, None, 1))
        messg = ECCConv(2)([nodes, adjcs, edges])
        messg = GlobalSumPool()(messg)
        final = Dense(1, "tanh")(messg)

        # compile model
        model = Model(inputs=[nodes, adjcs, edges], outputs=final)
        model.compile(optimizer=Adam(lr=0.1), loss="mse")

        # train model
        self.history = model.fit(loader.load(), verbose=0, steps_per_epoch=1)

    def tearDown(self):
        # delete temp directory
        shutil.rmtree(self.tmpdir)

    def test_good_file_path(self):
        """check history is saved properly"""
        save_history(self.history, self.tmpdat)
        self.assertTrue(pl.Path(self.tmpdat).is_file())

    def test_wrong_file_path(self):
        """check error wrong output path"""
        with self.assertRaises(IOError):
            save_history(self.history, pl.Path(r"/path/to/file"))

    def test_file_content(self):
        """check saved history content"""
        save_history(self.history, self.tmpdat)
        with open(self.tmpdat, "r") as file:
            lastline = file.readlines()[-1]
        self.assertEqual(lastline, "3.158142328262329\n")


class TestSaveModel(tf.test.TestCase):
    """Check save_model routine"""

    def setUp(self):
        super(TestSaveModel, self).setUp()

        # temporary directory
        self.tmpdir = tempfile.mkdtemp()
        # SavedModel model path
        self.tmpdat = os.path.join(self.tmpdir, "model")

    def tearDown(self):
        # delete temp directory
        shutil.rmtree(self.tmpdir)

    def test_good_model(self):
        """check model is saved properly"""
        # dummy model
        nodes = Input(shape=(1, 1))
        adjcs = Input(shape=(1, 1))
        edges = Input(shape=(1, 1, 1))
        messg = ECCConv(2)([nodes, adjcs, edges])
        messg = GlobalSumPool()(messg)
        final = Dense(1, "tanh")(messg)
        # create model
        model = Model(inputs=[nodes, adjcs, edges], outputs=final)

        save_gcnn(model, self.tmpdat)
        self.assertTrue(pl.Path(self.tmpdat).is_dir())

    def test_wrong_model(self):
        """check error for empty model"""
        model = Model()
        with self.assertRaises(IOError):
            save_gcnn(model, self.tmpdat)


if __name__ == "__main__":
    tf.test.main()
