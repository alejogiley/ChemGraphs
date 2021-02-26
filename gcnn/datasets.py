from __future__ import annotations

import os
import itertools

import numpy as np
import scipy.sparse as sp

from typing import Tuple, Type
from spektral.data import Dataset, Graph
from sklearn.preprocessing import QuantileTransformer


def split_dataset(dataset: Type[EstrogenDB], ratio=0.9) -> Tuple:
    """Split Dataset into Train and Tests sets
    
    Args:
        dataset: graph dataset
        ratio: split ratio of train / tests

    Returns:
        train and tests subsets

    """
    # randomize indexes
    indexes = np.random.permutation(len(dataset))

    # size of training subset
    size_train = int(ratio * len(dataset))

    # train/tests subsets
    subsets = np.split(indexes, [size_train])

    # dataset partition
    train = dataset[subsets[0]]
    tests = dataset[subsets[1]]

    return train, tests


def apply_transformation(reference: np.ndarray, dataset: np.ndarray) -> np.ndarray:
    """Feature transformation

    Transform features to follow a uniform or a normal distribution
    see: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html

    Args:
        reference: feature set to fit transformation
        dataset: feature set to apply transformation

    Returns:
         np.ndarray: transformed feature set

    """
    scaler = QuantileTransformer(output_distribution="normal")
    return scaler.fit(reference).transform(dataset)


def transform_datasets(train_set: Type[EstrogenDB], tests_set: EstrogenDB) -> Tuple:
    """Preprocessing of node features in Graph dataset

    Args:
        train_set: subset of graph dataset used for training
        tests_set: subset of graph dataset used for testing

    Returns:
        datasets with transformed node features

    """
    new_train_set = train_set
    new_tests_set = tests_set

    # number of node features
    n_features = train_set[0]["x"].shape[1]

    for index in range(n_features):
        # get train data for each feature
        # to fit transformation on
        train = [train_set[i]["x"][:, index] for i in enumerate(train_set)]
        train = list(itertools.chain(*train))
        train = np.array(train).reshape(-1, 1)

        for k in enumerate(train_set):
            # reshape node data for sklearn
            data = train_set[k]["x"][:, index]
            data = data.reshape(-1, 1)
            # apply transformation
            scaled_data = apply_transformation(train, data).reshape(-1)
            # replace node features in array
            new_train_set[k]["x"][:, index] = scaled_data

        for k in enumerate(tests_set):
            # reshape node data for sklearn
            data = tests_set[k]["x"][:, index]
            data = data.reshape(-1, 1)
            # apply transformation
            scaled_data = apply_transformation(train, data).reshape(-1)
            # replace node features in array
            new_tests_set[k]["x"][:, index] = scaled_data

    return new_train_set, new_tests_set


class EstrogenDB(Dataset):
    """Dataset from BindingDB"""

    def __init__(
        self,
        dpath=None,
        nodes=None,
        edges=None,
        adjcs=None,
        feats=None,
        **kwargs,
    ):
        self.nodes = nodes
        self.edges = edges
        self.adjcs = adjcs
        self.feats = feats
        # dataset to load
        self.dpath = dpath

        super().__init__(**kwargs)

    @Dataset.path.getter
    def path(self):
        path = os.path.join(self.dpath, "EstrogenDB.npz")
        return "" if not os.path.exists(path) else path

    def read(self):
        # load Graph objects
        data = np.load(
            os.path.join(self.dpath, "EstrogenDB.npz"),
            allow_pickle=True
        )

        # graphs
        output = [
            self.make_graph(
                node=data["x"][i],
                adjc=data["a"][i],
                edge=data["e"][i],
                feat=data["y"][i],
            )
            for i in enumerate(data["y"])
            if data["y"][i][-1] > 0.0
        ]

        return output

    def download(self):
        # save graph arrays into directory
        filename = os.path.join(self.dpath, "EstrogenDB")
        np.savez_compressed(
            filename,
            x=self.nodes,
            a=self.adjcs,
            e=self.edges,
            y=self.feats
        )

    @staticmethod
    def make_graph(node, adjc, edge, feat):
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
        # transform IC50 values
        # into pIC50
        y[-1] = np.log10(y[-1])

        # The edge features
        e = edge.astype(np.int8)
        # check shape (n_nodes, n_nodes, ..)
        assert e.shape[0] == len(node)
        assert e.shape[1] == len(node)

        return Graph(x=x, a=a, e=e, y=y)
