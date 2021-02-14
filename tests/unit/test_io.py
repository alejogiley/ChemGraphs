import os
import shutil
import tarfile
import tempfile

import numpy as np
import pandas as pd

from pathlib import Path
from unitest import TestCase
from tensorflow.keras import Sequential

from gcnn.io import save_model, save_history


def TestSaveHistory(TestCase):
    """Check save_history routine"""

    def setUp(self):
        # temporary directory
        self.tmpdir = tempfile.mkdtemp()
        # temporary file
        self.tmpdat = os.path.join(self.tmpdir, "tmp.dat")

        # simple dataset
        class Dataset(Dataset):
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
        self.data = Dataset()

        # dummy model
        nodes = Input(shape=(None, 1))
        adjcs = Input(shape=(None, None))
        edges = Input(shape=(None, None, 1))
        messg = ECCConv(2)([nodes, adcjs, edges])
        messg = GlobalSumPool()(x)
        final = Dense(1, "tanh")(x)

        # compile model
        model = Model(inputs=[nodes, adcjs, edges], outputs=final)
        model.compile(optimizer=Adam(lr=0.1), loss="mse")

        # train model
        self.history = model.fit(self.data, verbose=1, epochs=1)

    def tearDown(self):
        # delete temp directory
        shutil.rmtree(self.tmpdir)

    def test_good_file_path(self):
        """check file was saved properly"""
        save_history(self.history, self.tmpdat)
        self.assertTrue(Path(self.tmpdat).resolve().is_file())

    def test_wrong_file_path(self):
        """check error is raised for wrong path"""
        with self.assertRaises(IOError):
            save_history(self.history, Path(r"/wrong_file_path/file"))

    def test_file_content(self):
        """check file content / last line"""
        save_history(self.history, self.tmpdat)

        with open(self.tmpdat, 'r') as file:
            lastline = file.readlines()[-1]

        self.assertEqual(lastline, "1 2 3 4")


def TestSaveModel(TestCase):
    """Check save_model routine"""

    def setUp(self):
        # temporary directory
        self.tmpdir = tempfile.mkdtemp()
        # temporary file
        self.tmpdat = os.path.join(self.tmpdir, "tmp.h5")

        # dummy model
        nodes = Input(shape=(None, 1))
        adjcs = Input(shape=(None, None))
        edges = Input(shape=(None, None, 1))
        messg = ECCConv(2)([nodes, adcjs, edges])
        messg = GlobalSumPool()(x)
        final = Dense(1, "tanh")(x)

        # create model
        self.model = Model(
            inputs=[nodes, adcjs, edges], outputs=final)

    def tearDown(self):
        # delete temp directory
        shutil.rmtree(self.tmpdir)

    def test_good_file_path(self):
        """check file was saved properly"""
        save_model(os.path.join(self.tmpdat, self.model)
        self.assertTrue(Path(self.tmpdat).resolve().is_file())

    def test_wrong_file_path(self):
        """check error is raised for wrong path"""
        with self.assertRaises(IOError):
            save_model(Path(r"/wrong_file_path/file"))