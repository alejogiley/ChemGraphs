import numpy as np

# Generic List, supports indexing.
from typing import Tuple

# Cheminformatics package
from rdkit import Chem
from rdkit.Chem import AllChem

from gcnn.utils import onehot_encoding, str_is_float, symmetrize


def get_nodes(mol: object) -> np.ndarray:
    """Compute node features: vdw radius, charge, degree"""
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


def get_edges(mol: object) -> np.ndarray:
    """
    the bond types here are
    {'AROMATIC', 'DOUBLE', 'SINGLE', 'TRIPLE'}
    but the number of classes in rdkit is larger
    *temporary solution

    """
    keys = ["AROMATIC", "DOUBLE", "SINGLE", "TRIPLE"]

    num_atoms = mol.GetNumAtoms()
    edges = np.zeros((num_atoms, num_atoms, len(keys)))

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        edges[i, j] = onehot_encoding(str(bond.GetBondType()), keys)

    return symmetrize(edges)


def get_labels(mol: object, key="IC50 (nM)") -> np.ndarray:
    """Generate label data for each molecule

    indicators of right-censored ">" or lef-censored "<"
    which are reported for concentrations beyond detection limits.

    For reported concentrations, angle brackets are removed and
    boundary values are saved. When concentration value is 0,
    it means metric was not reported.

    """
    # read potency metric
    sample = mol.GetPropsAsDict()[key]
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


def data_features(data_path: str) -> Tuple[np.ndarray]:
    """Calculate graph objects features

    Args:
        data_path (str): path to Dataset from BindingDB

    Returns:
        x (ndarray): node features
        a (ndarray): adjacency matrices
        e (ndarray): edges features
        y (ndarray): labels features

    """
    # create instance of sdf reader
    suppl = Chem.SDMolSupplier(data_path, sanitize=True, strictParsing=True)

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
    y = [get_labels(mol) for mol in molecules]

    return x, a, e, y