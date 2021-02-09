import argparse
import joblib

from spektral.data import BatchLoader
from spektral.layers import ECCConv
from spektral.transforms import LayerPreprocess

from gcnn.datasets import split_dataset, transform_datasets
from gcnn.losses import (
    mse_loss,
    maxlike_mse_loss,
    maxlike_cse_loss,
    maxlike_tobit_loss,
)
from gcnn.models import train_model


def main(args):

    ##########################################################################
    # Prepare Graph Dataset
    ##########################################################################

    # Load dataset
    dataset = joblib.load(args.dataset)

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

    model, history = train_model(
        dataset=train_set,
        tf_loss=args.loss,
        n_layers=args.n_layers,
        channels=args.channels,
        batch_size=args.batch_size,
        number_epochs=args.epochs,
        learning_rate=args.learning_rate,
    )

    ##########################################################################
    # Testing GCNN
    ##########################################################################

    metrics = evaluate_model(model, test_set)

    ##########################################################################
    # Performance report
    ##########################################################################

    template = "MAE{0:<10} MSE{0:<10} rho{0:<10} r^2{0:<10} \%lefts{0:<10} \%right{0:<10}"
    print(template.format(metrics.values()), file=args.outfile)

    ##########################################################################
    # Save GCNN
    ##########################################################################

    return 0


def parse_arguments():

    ##########################################################################
    # Parse input arguments
    ##########################################################################

    parser = argparse.ArgumentParser(description="Epitope prediction app")

    epochs = 100  # Number of training epochs
    batch_size = 32  # MiniBatch sizes
    learning_rate = 1e-2  # Optimizer learning rate

    n_layers = 2  # number of ECCConv layers
    channels = [64, 16]  # number of Hidden units

    # Set the default for the dataset argument
    parser.add_argument("--bindingdb", required=True, help="BindingDB SDF file")
    parser.add_argument(
        "--data_path", required=True, help="Path to dataset directory"
    )

    args = parse.parse_args()

    return args


if __name__ == "__main__":

    arguments = parse_arguments()
    main(arguments)