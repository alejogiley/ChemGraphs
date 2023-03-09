#!/usr/bin/env python -W ignore

import argparse

from gcnn.io import save_dataset
from gcnn.datasets import GraphDB
from gcnn.features import data_features


def main(args):

    ##########################################################################
    # Prepare Graph Dataset
    ##########################################################################

    # read BindingDB and get graph features for each molecule
    nodes, adjcs, edges, labels = data_features(args.bindingdb,
                                                affinity=args.metric_type)

    # create Graph dataset
    dataset = GraphDB(
        nodes=nodes,
        edges=edges,
        adjcs=adjcs,
        feats=labels,
    )

    # Save Graph dataset
    save_dataset(dataset, args.data_path, args.file_name)

    return 0


def parse_arguments():

    ##########################################################################
    # Parse input arguments
    ##########################################################################
    parser = argparse.ArgumentParser(
        prog="setup_dataset",
        formatter_class=lambda prog: argparse.HelpFormatter(
            prog, width=120, max_help_position=40),
        description=
        "Setup a Spektral Dataset from BindingDB binding affinities",
        epilog='"Engage" (J.L. Picard)',
    )

    parser.add_argument("--bindingdb",
                        required=True,
                        help="BindingDB SDF file path")
    parser.add_argument("--data_path",
                        required=True,
                        help="Output datasets directory path")
    parser.add_argument("--file_name",
                        required=True,
                        help="Name output Graph database")

    parser.add_argument(
        "--metric_type",
        choices=["Ki", "IC50", "Kd", "EC50"],
        default="IC50",
        required=False,
        help="Binding metric type",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    arguments = parse_arguments()
    main(arguments)
