import argparse
import joblib

from spektral.layers import ECCConv
from spektral.transforms import LayerPreprocess

from gcnn.io import save_history, save_model
from gcnn.models import train_model, evaluate_model
from gcnn.datasets import split_dataset, transform_datasets
from gcnn.losses import (
    mse_loss,
    maxlike_mse_loss,
    maxlike_cse_loss,
    maxlike_tobit_loss,
)


def main(args):

    ##########################################################################
    # Prepare Graph Dataset
    ##########################################################################

    # Load dataset
    dataset = joblib.load(args.data_path)

    # Transform the adjacency matrix
    # according to ECCConv
    dataset.apply(LayerPreprocess(ECCConv))

    # Splitting data into train/tests
    train_set, tests_set = split_dataset(dataset, ratio=0.9)

    # Dataset transformation
    train_set, tests_set = transform_datasets(train_set, tests_set)

    ##########################################################################
    # Training GCNN
    ##########################################################################

    loss_types = {
        "mse_loss": mse_loss,
        "maxlike_mse_loss": maxlike_mse_loss,
        "maxlike_cse_loss": maxlike_cse_loss,
        "maxlike_tobit_loss": maxlike_tobit_loss
    }

    model, history = train_model(
        dataset=train_set,
        tf_loss=loss_types[args.loss_type],
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

    ##########################################################################
    # Performance report
    ##########################################################################

    template = "MAE{0:<10} MSE{0:<10} rho{0:<10} r^2{0:<10} %%lefts{0:<10} %%right{0:<10}"

    with open(args.metrics_path) as file:
        file.write(template.format(metrics.values()))

    ##########################################################################
    # Save GCNN
    ##########################################################################

    save_history(history, args.record_path)
    save_model(model, args.model_path)

    return 0


def parse_arguments():

    ##########################################################################
    # Parse input arguments
    ##########################################################################

    parser = argparse.ArgumentParser(
        prog="train_gcnn",
        description="Train GCNN model for affinity prediction",
        usage="%(prog)s [options]",
    )

    # Set the default for the dataset argument
    parser.add_argument(
        "-data_path", type=str, required=True, help="Dataset file path"
    )
    parser.add_argument(
        "-record_path",
        type=str,
        required=True,
        help="Training history output path",
    )
    parser.add_argument(
        "-model_path",
        type=str,
        required=True,
        help="Trained model output path",
    )
    parser.add_argument(
        "-metrics_path",
        type=str,
        required=True,
        help="Model performance output path",
    )
    parser.add_argument(
        "loss_type",
        type="str",
        metavar="loss",
        choices=[
            "mse_loss",
            "maxlike_mse_loss",
            "maxlike_cse_loss",
            "maxlike_tobit_loss",
        ],
        help="Loss function for regression",
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

    group = parser.add_argument_group(title="Loss function for regression")
    group.add_argument("mse_loss", help="Mean-squared error for NOT-censored data")
    group.add_argument("maxlike_mse_loss", help="Mean-squared maximum-likelihood error for NOT-censored data")
    group.add_argument("maxlike_cse_loss", help="Mean-squared maximum-likelihood error for CENSORED data")
    group.add_argument("maxlike_tobit_loss", help="Tobit loss or CENSORED data")

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    arguments = parse_arguments()
    main(arguments)
