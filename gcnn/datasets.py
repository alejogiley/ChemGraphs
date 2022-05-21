from __future__ import annotations

import itertools

import numpy as np
import scipy.sparse as sp

import tensorflow as tf
import tensorflow_datasets as tfds


class Graph:
    """A container to represent a graph structure
    
    Copied from the oficial Spektral implementation
    https://github.com/danielegrattarola/spektral/tree/master
    ommiting all data instance verification and defaults
    when data is ommited on the constructor 

    """
    def __init__(
        self,
        node=None,
        edge=None,
        adjc=None,
        feat=None,
        **kwargs,
    ):
        self.node = node
        self.edge = edge
        self.adjc = adjc
        self.feat = feat

    def numpy(self):
        return tuple(
            ret for ret in [
                self.node, 
                self.adjc, 
                self.edge, 
                self.feat
            ] if ret is not None)

    def get(self, *keys):
        return tuple(
            self[key] 
            for key in keys 
            if self[key] is not None
        )

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key, None)

    def __contains__(self, key):
        return key in self.keys

    def __repr__(self):
        out = "Graph(n_nodes={}, n_node_features={}, n_edge_features={}, n_labels={})"
        return out.format(
            self.n_nodes, 
            self.n_node_features, 
            self.n_edge_features, 
            self.n_labels
        )

    @property
    def n_nodes(self):
        if self.node is not None:
            return self.node.shape[-2]
        elif self.adjc is not None:
            return self.adjc.shape[-1]
        else:
            return None

    @property
    def n_edges(self):
        if isinstance(self.adjc, np.ndarray):
            return np.count_nonzero(self.adjc)
        else:
            return None

    @property
    def n_node_features(self):
        if self.node is not None:
            return self.node.shape[-1]
        else:
            return None

    @property
    def n_edge_features(self):
        if self.edges is not None:
            return self.edges.shape[-1]
        else:
            return None

    @property
    def n_labels(self):
        if self.feats is not None:
            shp = np.shape(self.feats)
            return 1 if len(shp) == 0 else shp[-1]
        else:
            return None


class GraphDB:
    """Database for Molecular Graphs"""
    
    def __init__(
        self,
        nodes=None,
        edges=None,
        adjcs=None,
        feats=None,
        **kwargs,
    ):
        self.graphs = self.read(nodes, edges, adjcs, feats)

    def read(self, nodes, edges, adjcs, feat):
        return [
            self.make_graph(
                node=nodes[i],
                adjc=adjcs[i],
                edge=edges[i],
                feat=feats[i],
            )
            for i, _ in enumerate(feats)
            # if binding affinity metrics
            # is not available ignore ligand
            if feats[i][-1] > 0.0
        ]
    def __len__(self):
        return len(self.graphs)

    @property
    def n_graphs(self):
        return self.__len__()

    @staticmethod
    def make_graph(node, adjc, edge, feat):
        """Create a Graph instance"""

        # The node features
        x = node.astype(float)

        # The adjacency matrix
        # convert to scipy.sparse matrix
        a = adjc.astype(np.int8)
        a = sp.csr_matrix(a)
        # check shape (n_nodes, n_nodes)
        assert a.shape[0] == len(node)
        assert a.shape[1] == len(node)

        # The labels
        y = feat.astype(float)
        # transform IC50 values into pIC50
        y[-1] = np.log10(y[-1])

        # The edge features
        e = edge.astype(np.int8)
        # check shape (n_nodes, n_nodes, ..)
        assert e.shape[0] == len(node)
        assert e.shape[1] == len(node)

        return Graph(node=x, adjc=a, edge=e, feat=y)


def split_dataset(dataset: GraphDB, ratio=0.9) -> Tuple:
    """Split Dataset into Train and Tests sets

    Args:
        dataset: graph dataset
        ratio: split ratio of train / tests

    Returns:
        train and tests subsets

    """
    # randomize indexes
    indexes = np.random.permutation(dataset.n_graphs)

    # size of training subset
    size_train = int(ratio * dataset.n_graphs)

    # train/tests subsets
    subsets = np.split(indexes, [size_train])

    # dataset partition
    train = dataset[subsets[0]]
    tests = dataset[subsets[1]]

    return train, tests