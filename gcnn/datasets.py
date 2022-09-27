from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from typing import Tuple, List


class Graph:
    """A container to represent a graph structure

    Copied from the oficial Spektral implementation
    https://github.com/danielegrattarola/spektral/tree/master
    ommiting all data instance verification and defaults
    when data is ommited on the constructor 

    """

    def __init__(
        self,
        nodes: np.ndarray = None,
        edges: np.ndarray = None,
        adjcs: np.ndarray = None,
        feats: np.ndarray = None,
    ):
        """"
        Initialize the Graph object

        Parameters
        ----------
        nodes: np.ndarray
            Node features of shape [n_nodes, n_node_features].
        edges: np.ndarray
            Edge features of shape [n_edges, n_edge_features].
        adjcs: np.ndarray
            Adjacency matrix of shape [n_nodes, n_nodes].
        feats: np.ndarray
            Node labels of shape [n_nodes, n_labels].

        Returns
        -------
        Graph
            A Graph object.

        """
        self.nodes = nodes
        self.edges = edges
        self.adjcs = adjcs
        self.feats = feats

    def numpy(self) -> Tuple[np.ndarray]:
        """Converts the graph to numpy arrays."""
        return tuple(
            x for x in [
                self.nodes,
                self.adjcs,
                self.edges,
                self.feats
            ] if x is not None)

    def get(self, *keys) -> Tuple:
        """Returns the values of the specified keys."""
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
    def n_nodes(self) -> int:
        """Number of nodes in the graph."""
        if self.nodes is not None:
            return self.nodes.shape[-2]
        elif self.adjcs is not None:
            return self.adjcs.shape[-1]
        else:
            return None

    @property
    def n_edges(self) -> int:
        """Number of edges in the graph."""
        if isinstance(self.adjcs, np.ndarray):
            return np.count_nonzero(self.adjc)
        else:
            return None

    @property
    def n_node_features(self) -> int:
        """Number of features per node."""
        if self.nodes is not None:
            return self.nodes.shape[-1]
        else:
            return None

    @property
    def n_edge_features(self) -> int:
        """Number of features per edge."""
        if self.edges is not None:
            return self.edges.shape[-1]
        else:
            return None

    @property
    def n_labels(self) -> int:
        """Number of labels per node."""
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
    ):
        """"
        Initialize the GraphDB object

        Parameters
        ----------
        nodes: np.ndarray
            Node features of shape [n_nodes, n_node_features].
        edges: np.ndarray
            Edge features of shape [n_edges, n_edge_features].
        adjcs: np.ndarray
            Adjacency matrix of shape [n_nodes, n_nodes].
        feats: np.ndarray
            Node labels of shape [n_nodes, n_labels].

        Returns
        -------
        GraphDB

        """
        self.nodes = nodes
        self.edges = edges
        self.adjcs = adjcs
        self.feats = feats

        self.graphs = self.read(nodes, edges, adjcs, feats)

    def read(self, nodes, edges, adjcs, feats) -> List[Graph]:
        """Reads the data and returns a list of Graph objects."""
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

    def __len__(self) -> int:
        return len(self.graphs)

    @property
    def n_graphs(self) -> int:
        return self.__len__()

    @staticmethod
    def make_graph(node, adjc, edge, feat) -> Graph:
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
        # e.g. transform IC50 values into pIC50
        y[-1] = np.log10(y[-1])

        # The edge features
        e = edge.astype(np.int8)
        # check shape (n_nodes, n_nodes, ..)
        assert e.shape[0] == len(node)
        assert e.shape[1] == len(node)

        return Graph(nodes=x, adjcs=a, edges=e, feats=y)


def split_dataset(dataset: GraphDB, ratio=0.9) -> Tuple:
    """Split Dataset into Train and Tests sets

    Args:
        dataset (GraphDB): Dataset to be split
        ratio (float, optional): Ratio of the train set. Defaults to 0.9.

    Returns:
        Tuple: Train and Test sets

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
