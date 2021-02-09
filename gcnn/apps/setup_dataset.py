import os
import argparse

from gcnn.data.datasets import (
    EstrogenDB, data_features)


def main(args):

    ##########################################################################
    # Prepare Graph Dataset
    ##########################################################################

    # read BindingDB and get graph features for each molecule
    nodes, adjcs, edges, labels = data_features(args.bindingdb)

    # create Graph dataset
    dataset = EstrogenDB(
        n_samples=len(nodes),
        nodes=nodes,
        edges=edges,
        adjcs=adjcs,
        feats=labels,
        dpath=args.data_path)

    # dataset name is hardcoded
    path = os.path.join(
        args.data_path, f'EstrogenGraphs.gz')
    
    # save Graph dataset
    joblib.dump(
        dataset, path, compress=('gzip', 6))

    return 0


def parse_arguments():

    ##########################################################################
    # Parse input arguments
    ##########################################################################

    parser = argparse.ArgumentParser(
        description="Epitope prediction app")

    # Set the default for the dataset argument
    parser.add_argument(
        "--bindingdb", required=True, help="BindingDB SDF file")
    parser.add_argument(
        "--data_path", required=True, help="Path to dataset directory")

    args = parse.parse_args()

    return args


if __name__ == "__main__":

    arguments = parse_arguments()
    main(arguments)
