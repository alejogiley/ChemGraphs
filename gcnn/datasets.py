from __future__ import annotations

import copy
import numpy as np
import scipy.sparse as sp

from typing import Tuple, List
from spektral.data import Dataset
from spektral.data.graph import Graph


class GraphDB(Dataset):
    """Database for Molecular Graphs

    Modified from the oficial Spektral implementation
    https://github.com/danielegrattarola/spektral/blob/master/spektral/data/dataset.py

    """

    def __init__(
        self,
        nodes: List[np.ndarray] = None,
        edges: List[np.ndarray] = None,
        adjcs: List[np.ndarray] = None,
        feats: List[np.ndarray] = None,
        **kwargs,
    ):
        """ "
        Initialize the GraphDB object

        Parameters
        ----------
        nodes: list[np.ndarray]
            Node features of shape [n_nodes, n_node_features].
        edges: list[np.ndarray]
            Edge features of shape [n_edges, n_edge_features].
        adjcs: list[np.ndarray]
            Adjacency matrices each of shape [n_nodes, n_nodes].
        feats: list[np.ndarray]
            Target labels of shape [n_labels,].

        Returns
        -------
        GraphDB

        """
        self.nodes = nodes
        self.edges = edges
        self.adjcs = adjcs
        self.feats = feats

        super().__init__(**kwargs)

    def read(self) -> List[Graph]:
        """Reads the data and returns a list of Graph objects."""
        return [
            self.make_graph(
                node=nodes[i],
                adjc=adjcs[i],
                edge=edges[i],
                feat=feats[i],
            ) for i, _ in enumerate(feats)
                node=self.nodes[i],
                adjc=self.adjcs[i],
                edge=self.edges[i],
                feat=self.feats[i],
            )
            for i, _ in enumerate(self.feats)
            # if binding affinity metrics
            # is not available ignore ligand
            if self.feats[i][-1] > 0.0
        ]

    def apply(self, transform: callable):
        """Applies a transformation to the graphs in the database."""
        if not callable(transform):
            raise ValueError("`transform` must be callable")
        for i in range(self.n_graphs):
            self.graphs[i] = transform(self.graphs[i])

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, key: int | slice | list | np.ndarray | bool) -> GraphDB:
        """Return a graph or a subset of graphs from the dataset.

        This method supports indexing with integers, slices, lists, and arrays
        of integers (and booleans). In all cases, it returns a subset of the
        graphs in the dataset.

        Args:
            key (int, slice, list, array, bool): Index.

        Returns:
            :obj:`Dataset`: A new :obj:`Dataset` object with the selected graphs.

        """
        # Copy the dataset
        dataset = copy.copy(self)

        if not (np.issubdtype(type(key), np.integer) or isinstance(key, (slice, list, tuple, np.ndarray))):
            # If key is not an integer or a slice, raise an error
            raise ValueError("Unsupported key type: {}".format(type(key)))

        # Single graph selection ##########################################

        if np.issubdtype(type(key), np.integer):
            return self.graphs[int(key)]

        # Slice or list of graphs selection ###############################

        if isinstance(key, slice):
            # Get the start, stop, and step from the slice
            dataset.graphs = self.graphs[key]

        else:
            # If key is a list or array of integers, return the corresponding graphs
            dataset.graphs = [self.graphs[i] for i in key]

        return dataset

    @property
    def n_graphs(self) -> int:
        return self.__len__()

    @staticmethod
    def make_graph(node: np.ndarray, adjc: np.ndarray, edge: np.ndarray, feat: np.ndarray) -> Graph:
        """Create a Graph instance

        Args:
        node: np.ndarray
            Node features of shape [n_nodes, n_node_features].
        edge: np.ndarray
            Edge features of shape [n_edges, n_edge_features].
        adjc: np.ndarray
            Adjacency matrix of shape [n_nodes, n_nodes].
        feat: np.ndarray
            Target labels of shape [n_labels,].

        Returns:
        Graph instance

        """
        # The node features
        x = node.astype(float)

        # The adjacency matrix
        # convert to scipy.sparse matrix
        a = adjc.astype(np.int8)
        a = sp.csr_matrix(a)

        # The labels
        y = feat.astype(float)
        # e.g. transform IC50 values into pIC50
        y[-1] = np.log10(y[-1])

        # The edge features
        e = edge.astype(np.int8)

        return Graph(x=x, a=a, e=e, y=y)


def split_dataset(dataset: GraphDB, ratio: float = 0.9) -> Tuple:
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


class LayerPreprocess(object):
    """
    Applies the `preprocess` function of a convolutional Layer to the adjacency matrix.
    taken from https://github.com/danielegrattarola/spektral/blob/master/spektral/transforms/layer_preprocess.py#L1
    """

    def __init__(self, layer_class: callable):
        """ "
        Initialize the LayerPreprocess object

        Parameters
        ----------
        layer_class: Layer
            Layer class to be used for preprocessing the adjacency matrix.

        Returns
        -------
        LayerPreprocess

        """
        self.layer_class = layer_class

    def __call__(self, graph):
        if graph.adjcs is not None and hasattr(self.layer_class, "preprocess"):
            graph.adjcs = self.layer_class.preprocess(graph.adjcs)
        return graph
