import logging
import os

import argparse
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from spektral.data import BatchLoader
from spektral.layers import ECCConv
from spektral.transforms import LayerPreprocess

from gcnn.models.models import gcn_model, train_model
from gcnn.data.datasets import (
    get_nodes, get_edges, get_labels,
    data_features, split_dataset,
    EstrogenDB
)

logger = logging.getLogger(__name__)


def main(args): 

    ################################################################################
    # Load data
    ################################################################################

    # create instance of sdf reader
    suppl = Chem.SDMolSupplier(args.data, sanitize=True, strictParsing=True)

    # read all molecules besides ones with errors into a list
    mols = [mol for mol in suppl if mol is not None]

    # Get nodes
    x = [get_nodes(mol) for mol in mols]
        
    # Adjacency matrices
    a = [Chem.rdmolops.GetAdjacencyMatrix(mol) for mol in mols]

    # Edge features: bond types
    e = [get_edges(mol) for mol in mols]

    # Labels: (rank, IC50s)
    # this metric is less reliable than e.g. Kd as 
    # it depends on the of the substrates used in 
    # the essay and it is cell type dependent.
    y = [get_labels(mol) for mol in mols]

    ################################################################################
    # Create dataset
    ################################################################################

    dataset = EstrogenDB(
        n_samples=len(y),
        nodes=x, edges=e, 
        adjcs=a, feats=y, 
        dpath=args.path)

    # Transform the adjacency matrix 
    # according to ECCConv
    dataset.apply(LayerPreprocess(ECCConv))

    ################################################################################
    # Partition dataset
    ################################################################################

    train, tests = split_dataset(dataset, ratio=0.9)

    ################################################################################
    # Parameters
    ################################################################################

    epochs = 20  # Number of training epochs
    batch_size = 32 # MiniBatch sizes
    learning_rate = 1e-3 # Optimizer learning rate

    n_layers = 3  # number of ECCConv layers
    n_neurons = 8  # number of Dense channels
    n_channels = [64, 32, 32]  # number of Hidden units

    ################################################################################
    # Train model
    ################################################################################

    model, history = train_model(dataset, epochs, learning_rate, n_channels, n_layers, n_neurons)

    ################################################################################
    # Evaluate model
    ################################################################################

    print("Testing model")
    loader = BatchLoader(tests, batch_size=batch_size)
    model_loss = model.evaluate(loader.load(), steps=loader.steps_per_epoch)
    print("Done. Test loss: {}".format(model_loss))


def parse_arguments():

    ################################################################################
    # Parse input arguments 
    ################################################################################

    parser = argparse.ArgumentParser(description="A Graph Convolutional Neural Network with Edge-Conditioned Filters")

    parser.add_argument("--data", required=True, help="BindingDB dataset")
    parser.add_argument("--path", required=True, help="Path to dataset directory")

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_arguments()
    main(args)
