#!/usr/bin/env python -W ignore

import argparse
import joblib

from spektral.layers import ECCConv
from spektral.transforms import LayerPreprocess

from gcnn.io import save_history, save_gcnn, save_metrics
from gcnn.models import train_model, evaluate_model
from gcnn.datasets import split_dataset, transform_datasets
from gcnn.utils import sigma, set_random_seed
from gcnn.metrics import r_squared
from gcnn.losses import (
    mse_loss,
    maxlike_mse_loss,
    maxlike_cse_loss,
    maxlike_tobit_loss,
)


def main(args):

    set_random_seed(args.seed)

    ##########################################################################
    # Prepare Graph Dataset
    ##########################################################################

    # Load dataset
    dataset = joblib.load(args.data_path)

    # Transform the adjacency matrix
    # according to ECCConv
    dataset.apply(LayerPreprocess(ECCConv))

    # Splitting data into train/tests
    train_set, tests_set = split_dataset(dataset, ratio=0.8)

    # Dataset transformation
    train_set, tests_set = transform_datasets(train_set, tests_set)

    ##########################################################################
    # Training GCNN
    ##########################################################################

    loss_types = {
        "mse_loss": mse_loss,
        "maxlike_mse_loss": maxlike_mse_loss,
        "maxlike_cse_loss": maxlike_cse_loss,
        "maxlike_tobit_loss": maxlike_tobit_loss,
    }

    model, history = train_model(
        dataset=train_set,
        tf_loss=loss_types[args.loss_function],
        metrics=[r_squared, sigma],
        n_layers=args.n_layers,
        channels=args.channels,
        batch_size=args.batch_size,
        number_epochs=args.epochs,
        learning_rate=args.learning_rate,
    )

    ##########################################################################
    # Testing GCNN
    ##########################################################################

    metrics = evaluate_model(model, tests_set)
    save_metrics(metrics, args.metrics_path)

    ##########################################################################
    # Save GCNN
    ##########################################################################

    save_history(history, args.record_path)
    save_gcnn(model, args.model_path)

    return 0


def parse_arguments():

    ##########################################################################
    # Parse input arguments
    ##########################################################################

    def choices_descriptions():
        return """
Loss function for regression:
    mse_loss            - Mean squared error for NOT-censored data
    maxlike_mse_loss    - Mean squared maximum-likelihood error for NOT-censored data
    maxlike_cse_loss    - Mean squared maximum-likelihood error for CENSORED data
    maxlike_tobit_loss  - Tobit loss or CENSORED data
            
\"There's No Such Thing As The Unknown, Only Things Temporarily Hidden\" (James T. Kirk)
"""

    def get_choices():
        return ["mse_loss", "maxlike_mse_loss", "maxlike_cse_loss", "maxlike_tobit_loss"]

    parser = argparse.ArgumentParser(
        prog="train_gcnn",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(
            prog, width=120, max_help_position=40
        ),
        description="Train GCNN model for affinity prediction",
        epilog=choices_descriptions(),
    )

    # optional arguments
    parser.add_argument(
        "--data_path", 
        type=str, 
        required=True, 
        help="Input graph dataset file path"
    )
    parser.add_argument(
        "--record_path",
        type=str,
        required=True,
        help="Training history output path",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Trained model output path",
    )
    parser.add_argument(
        "--metrics_path",
        type=str,
        required=True,
        help="Model performance output path",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=2,
        required=False,
        help="Number of GCN hidden layers",
    )
    parser.add_argument(
        "--channels",
        nargs="*",
        type=int,
        default=[64, 32],
        help="List of channels per GCN layer",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        required=False,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        required=False,
        help="mini-batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-2,
        required=False,
        help="Optimizer learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        required=False,
        help="Random seed",
    )

    # positional arguments
    parser.add_argument(
        "loss_function",
        metavar="loss_function",
        choices=get_choices(),
        help="Loss function for regression",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
