# Unit tests for functions in datasets.py

import os
import sys

import unittest
import numpy as np

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from gcnn.datasets import Graph, GraphDB


class TestGraph(unittest.TestCase):
    """Test Graph class"""

    def setUp(self):
        super(TestGraph, self).setUp()
        pass

    @given(nodes=st.lists(st.floats(), min_size=10))
    def test_get_nodes(self, nodes):
        """Check function returns nodes"""
        graph = Graph(nodes=np.array(nodes))
        target_values = np.array(nodes)
        actual_values = graph.get('nodes')[0]
        np.testing.assert_array_equal(actual_values, target_values)

    @given(edges=st.lists(st.floats(), min_size=10))
    def test_get_edges(self, edges):
        """Check function returns edges"""
        graph = Graph(edges=np.array(edges))
        target_values = np.array(edges)
        actual_values = graph.get('edges')[0]
        np.testing.assert_array_equal(actual_values, target_values)

    @given(adjacency=st.lists(st.floats(), min_size=10))
    def test_get_adjacency(self, adjacency):
        """Check function returns adjacency"""
        graph = Graph(adjcs=np.array(adjacency))
        target_values = np.array(adjacency)
        actual_values = graph.get('adjcs')[0]
        np.testing.assert_array_equal(actual_values, target_values)

    @given(features=st.lists(st.floats(), min_size=10))
    def test_get_features(self, features):
        """Check function returns features"""
        graph = Graph(feats=np.array(features))
        target_values = np.array(features)
        actual_values = graph.get('feats')[0]
        np.testing.assert_array_equal(actual_values, target_values)

    @given(
        nodes=arrays(dtype=np.float32,
                     shape=st.tuples(st.integers(1, 10), st.integers(1, 10))),
        feats=arrays(dtype=np.float32,
                     shape=st.tuples(st.integers(1, 10), st.integers(1, 10))),
        edges=arrays(dtype=np.int32,
                     shape=st.tuples(st.integers(1, 10), st.integers(1, 10))),
        adjcs=arrays(dtype=np.int32,
                     shape=st.tuples(st.integers(1, 10), st.integers(1, 10))),
    )
    def test_graph_numpy(self, nodes, edges, adjcs, feats):
        """Check function returns a Tuple"""
        graph = Graph(
            nodes=nodes,
            edges=edges,
            adjcs=adjcs,
            feats=feats,
        )
        actual_values = graph.numpy()
        target_values = (nodes, adjcs, edges, feats)
        self.assertSequenceEqual(actual_values, target_values)


class TestGraphDB(unittest.TestCase):
    """Test GraphDB class"""

    size = 10

    def setUp(self):
        super(TestGraphDB, self).setUp()
        pass

    @given(
        nodes=st.lists(
            arrays(dtype=np.float32, shape=(2, 3)),
            min_size=size,
            max_size=size,
        ),
        edges=st.lists(
            arrays(dtype=np.int32, shape=(2, 2, 3)),
            min_size=size,
            max_size=size,
        ),
        adjcs=st.lists(
            arrays(dtype=np.int32, shape=(2, 2)),
            min_size=size,
            max_size=size,
        ),
        feats=st.lists(
            arrays(dtype=np.float32,
                   elements=st.floats(
                       allow_nan=False,
                       width=32,
                       min_value=1,
                       max_value=10,
                   ),
                   shape=(1)),
            min_size=size,
            max_size=size,
        ),
    )
    def test_graphdb_valid_input(self, nodes, edges, adjcs, feats):
        """Test loading of GraphDB with valid features"""
        graphdb = GraphDB(
            nodes=nodes,
            edges=edges,
            adjcs=adjcs,
            feats=feats,
        )
        self.assertEqual(graphdb.n_graphs, self.size)

    @given(
        nodes=st.lists(
            arrays(dtype=np.float32, shape=(2, 3)),
            min_size=size,
            max_size=size,
        ),
        edges=st.lists(
            arrays(dtype=np.int32, shape=(2, 2, 3)),
            min_size=size,
            max_size=size,
        ),
        adjcs=st.lists(
            arrays(dtype=np.int32, shape=(2, 2)),
            min_size=size,
            max_size=size,
        ),
        feats=st.lists(
            arrays(dtype=np.float32,
                   elements=st.floats(
                       allow_nan=False,
                       width=32,
                       min_value=0,
                       max_value=0,
                   ),
                   shape=(1)),
            min_size=size,
            max_size=size,
        ),
    )
    def test_graphdb_zero_input(self, nodes, edges, adjcs, feats):
        """Test loading of GraphDB with zero features"""
        graphdb = GraphDB(
            nodes=nodes,
            edges=edges,
            adjcs=adjcs,
            feats=feats,
        )
        self.assertEqual(graphdb.n_graphs, 0)


if __name__ == "__main__":
    unittest.main()
