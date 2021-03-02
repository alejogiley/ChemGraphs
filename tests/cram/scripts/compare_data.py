#!/usr/bin/env python -W ignore

import sys
import joblib
import numpy as np


def cmp(file1: str, file2: str):
    data1 = joblib.load(file1)
    data2 = joblib.load(file2)

    def check_allclose(x, y):
        return np.testing.assert_allclose(x, y, rtol=1e-5, atol=0)

    if data1.n_graphs != data2.n_graphs:
        print("Error! Datasets do not have same number of graphs")
    if all([check_allclose(x, y) for x, y in zip(data1.nodes, data2.nodes)]):
        print("Error! Nodes features do not match")
    if all([check_allclose(x, y) for x, y in zip(data1.edges, data2.edges)]):
        print("Error! Edges features do not match")
    if all([check_allclose(x, y) for x, y in zip(data1.feats, data2.feats)]):
        print("Error! Label features do not match")
    if all([check_allclose(x, y) for x, y in zip(data1.adjcs, data2.adjcs)]):
        print("Error! Connectivities do not match")
    print("Success! Datasets have no differences")


if __name__ == "__main__":
    cmp(sys.argv[1], sys.argv[2])
