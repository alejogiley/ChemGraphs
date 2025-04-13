import numpy as np

from unittest import TestCase, mock
from spektral.data.graph import Graph
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays, array_shapes

from gcnn.datasets import GraphDB, split_dataset, LayerPreprocess


class TestGraphDB(TestCase):

    def setUp(self):
        self.nodes = np.random.rand(1, 10, 5)
        self.edges = np.random.rand(1, 11, 4)
        self.feats = np.random.rand(1, 1)
        self.adjcs = np.random.randint(2, size=(1, 10, 10))
        self.dataset = GraphDB(nodes=self.nodes,
                               edges=self.edges,
                               adjcs=self.adjcs,
                               feats=self.feats)

    def test_read(self):
        graphs = self.dataset.read()
        self.assertIsInstance(graphs, list)
        self.assertIsInstance(graphs[0], Graph)

    def test_apply(self):
        transform = mock.Mock()
        self.dataset.apply(transform)
        self.assertEqual(transform.call_count, len(self.dataset))

    def test_len(self):
        self.assertEqual(len(self.dataset), len(self.nodes))

    def test_getitem(self):
        graph = self.dataset[0]
        self.assertIsInstance(graph, Graph)

    def test_n_graphs(self):
        self.assertEqual(self.dataset.n_graphs, len(self.nodes))

    @given(
        node=arrays(
            dtype=np.float64,
            shape=array_shapes(min_dims=2,
                               max_dims=2,
                               min_side=1,
                               max_side=100),
            elements=st.floats(allow_nan=False, allow_infinity=False),
        ),
        edge=arrays(
            dtype=np.float64,
            shape=array_shapes(min_dims=2,
                               max_dims=2,
                               min_side=1,
                               max_side=100),
            elements=st.floats(allow_nan=False, allow_infinity=False),
        ),
        feat=arrays(
            dtype=np.float64,
            shape=array_shapes(min_dims=1,
                               max_dims=1,
                               min_side=1,
                               max_side=100),
            elements=st.floats(allow_nan=False,
                               allow_infinity=False,
                               min_value=0.0,
                               exclude_min=True),
        ),
    )
    @settings(max_examples=500)
    def test_make_graph(self, node, edge, feat):
        adjcs = np.random.randint(2, size=node.shape[0]**2).reshape(
            node.shape[0], node.shape[0])
        graph = self.dataset.make_graph(node, adjcs, edge, feat)
        self.assertIsInstance(graph, Graph)


class TestSplitDataset(TestCase):

    def setUp(self):
        self.nodes = np.random.rand(8, 10, 5)
        self.edges = np.random.rand(8, 11, 4)
        self.feats = np.random.rand(8, 1)
        self.adjcs = np.random.randint(2, size=(8, 10, 10))
        self.dataset = GraphDB(nodes=self.nodes,
                               edges=self.edges,
                               adjcs=self.adjcs,
                               feats=self.feats)

    def test_split_dataset(self):
        ratio = 0.9
        train_set, val_set = split_dataset(self.dataset, ratio)
        self.assertIsInstance(train_set, GraphDB)
        self.assertIsInstance(val_set, GraphDB)
        self.assertEqual(len(train_set), int(ratio * len(self.dataset)))


class TestLayerPreprocess(TestCase):

    def setUp(self):
        self.layer_class = mock.Mock(preprocess=mock.Mock())
        self.preprocessing = LayerPreprocess(self.layer_class)

    def test_call(self):
        graph = mock.Mock()
        self.preprocessing(graph)
        self.preprocessing.layer_class.preprocess.assert_called()
