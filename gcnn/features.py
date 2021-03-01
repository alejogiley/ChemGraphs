from __future__ import annotations

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
from typing import Tuple, Literal, List, Any

from gcnn.utils import onehot_encoding, str_is_float, symmetrize


METRICS = Literal["Ki (nM)", "IC50 (nM)", "Kd (nM)", "EC50 (nM)"]


def get_nodes(mol: Mol) -> np.ndarray:
    """Compute node features

    Node (atom) features are the van der waals radius, the atomic charge
    and the number of directly-bonded neighbors in each molecule.

    Args:
        mol: rdkit Mol object

    Returns:
        array of node feaures

    """
    AllChem.ComputeGasteigerCharges(mol)

    nodes = np.array(
        [
            (
                Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()),
                atom.GetDoubleProp("_GasteigerCharge"),
                atom.GetDegree(),
            )
            for atom in mol.GetAtoms()
        ]
    )

    return nodes


def get_edges(mol: Mol) -> np.ndarray:
    """Compute edge features

    Eged (bond) features are categorical, a one-hot encoded vector
    is used to represent one of these types: aromatic, "double", "single", or "triple".
    although these are not all the classes available in rdkit.

    Args:
        mol: rdkit Mol object

    Returns:
        array of edge feaures

    """
    keys = ["AROMATIC", "DOUBLE", "SINGLE", "TRIPLE"]

    num_atoms = mol.GetNumAtoms()
    edges = np.zeros((num_atoms, num_atoms, len(keys)))

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges[i, j] = onehot_encoding(keys.index(str(bond.GetBondType())), len(keys))

    return symmetrize(edges)


def get_labels(mol: Mol, metric: METRICS = "IC50 (nM)") -> np.ndarray:
    """Compute target variables

    The target variables are the binding affinities of the molecules.
    Experimental values are sometimes right and left censored, for ligands
    whose binding concentrations range is beyond detection limits.
    Those values are indicated by the symbols ">" and "<" for right and
    left censored respecively.

    Target variable is reported as a vector of 3 dimensions, the first 2 indicate
    whether the metric is censored and the last dimension reports the experimental
    binding affinity, or the censored boundary. When the input metric type, e.g. IC50
    are not reported for a molecule the target metric is assigned 0.

    Args:
        mol: rdkit Mol object
        metric: type of affinity metric

    Returns:
        array of target variables

    """
    # read potency metric
    sample = mol.GetPropsAsDict()[metric]
    # remove leading and trailing whitespaces
    sample = sample.strip()

    # below exp. range
    if "<" in sample:

        lefts = 1
        right = 0

        metrics = float(sample.replace("<", ""))

    # outside exp. range
    elif ">" in sample:

        lefts = 0
        right = 1

        metrics = float(sample.replace(">", ""))

    # inside exp. range
    elif str_is_float(sample):

        lefts = 0
        right = 0

        metrics = float(sample)

    # no data provided
    else:

        lefts = 0
        right = 0

        metrics = 0.0

    return np.array([lefts, right, metrics])


def data_features(
    path: str,
    affinity: str = "IC50",
) -> Tuple[List[Any], List[Any], List[Any], List[Any]]:
    """Calculate graph features from BindingDB Dataset

    Args:
        path: path to dataset
        affinity: target metric type

    Returns:
        x: node features
        a: adjacency matrices
        e: edges features
        y: labels features

    """
    # create instance of sdf reader
    suppl = Chem.SDMolSupplier(path, sanitize=True, strictParsing=True)

    # read all molecules besides ones with errors into a list
    molecules = [mol for mol in suppl if mol is not None]

    # Get nodes
    x = [get_nodes(mol) for mol in molecules]

    # Adjacency matrices
    a = [Chem.rdmolops.GetAdjacencyMatrix(mol) for mol in molecules]

    # Edge features: bond types
    e = [get_edges(mol) for mol in molecules]

    # Labels: (rank, IC50s)
    # this metric is less reliable than e.g. Kd as
    # it depends on the of the substrates used in
    # the essay and it is cell type dependent.
    y = [get_labels(mol, metric=affinity + " (nM)") for mol in molecules]

    return x, a, e, y
