import os

import numpy as np
import scipy.sparse as sp
from rdkit import Chem
from rdkit.Chem import AllChem
from spektral.data import Dataset, Graph

from gcnn.utils.misc import str_is_float, symmetrize


def get_nodes(mol):
    """Compute node features: 
    atm number, charge, position
    
    """
    AllChem.ComputeGasteigerCharges(mol)
    nodes = np.concatenate((
        np.array([(
            atom.GetAtomicNum(), 
            atom.GetDoubleProp("_GasteigerCharge")) 
        for atom in mol.GetAtoms()]),
        mol.GetConformer().GetPositions()[:,:2]),
        axis=1
    )
    return nodes


def get_edges(mol):
    
    natms = mol.GetNumAtoms()
    edges = np.zeros((natms, natms))
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges[i, j] = bond.GetBondTypeAsDouble()
    
    return symmetrize(edges)[:, :, None]


def get_labels(mol, key='IC50 (nM)'):
    """Generate label data for each molecule
    
    "rank" indicates precense or absence of angle brackets,
    which are reported for concentrations beyond detection limits.
    rank = 1 when "<", 2 when ">", and 3 when none
    
    "conc" containts the reported concentration values
    angle brackets are removed and boundary values are saved.
    when conc value is 0, it means metric was not reported.
    
    """
    # read potency metric
    sample = mol.GetPropsAsDict()[key]
    # remove leading and trailing whitespaces
    sample = sample.strip()
        
    # below exp. range
    if "<" in sample: 
        
        rank = 1
        conc = sample.replace('<', '')

    # outside exp. range
    elif ">" in sample:
        
        rank = 2
        conc = sample.replace('>', '')

    # inside exp. range
    elif str_is_float(sample):
        
        rank = 3
        conc = sample

    # no data provided
    else:
        rank = 3
        conc = 0.0
    
    return np.array([rank, float(conc)])


def data_features(data_path):
    """Calculate graph objects features
    for each molecule in the dataset

    Args:
        data_path (str): path to Dataset from BindingDB
    
    """
    # create instance of sdf reader
    suppl = Chem.SDMolSupplier(data_path, sanitize=True, strictParsing=True)

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

    return (x, a, e, y)


def split_dataset(dataset, ratio=0.9):
    """Partition Dataset into Train and Tests sets
    """
    # randomize indexes
    indxs = np.random.permutation(len(dataset))

    # split 90%/10%
    split = int(0.9 * len(dataset))

    # Train/test indexes
    trnxs, tesxs = np.split(indxs, [split])

    # Dataset partition
    train, tests = dataset[trnxs], dataset[tesxs]

    return train, tests

class EstrogenDB(Dataset):
    """Dataset from BindingDB
    """
    def __init__(self, 
                 n_samples,
                 dpath=None, 
                 nodes=None, 
                 edges=None, 
                 adjcs=None, 
                 feats=None,
                 **kwargs):
        self.n_samples = n_samples
        self.nodes = nodes
        self.edges = edges
        self.adjcs = adjcs
        self.feats = feats
        # dataset to load
        self.dpath = dpath
        
        super().__init__(**kwargs)
        
    @Dataset.path.getter
    def path(self):
        return self.dpath
        
    def read(self):
        # create Graph objects
        data = np.load(os.path.join(
            self.dpath, f'EstrogenDB.npz'), 
                       allow_pickle=True)
        
        output = [
            self.make_graph(
                node=data['x'][i],
                adjc=data['a'][i], 
                edge=data['e'][i],
                feat=data['y'][i])
            for i in range(self.n_samples)
            if data['y'][i][1] != 0
        ]
        
        self.n_samples = len(output)
        
        return output
    
    def download(self):
        # save graph arrays into directory
        filename = os.path.join(self.dpath, f'EstrogenDB')
        
        np.savez_compressed(
            filename, 
            x=self.nodes, 
            a=self.adjcs, 
            e=self.edges, 
            y=self.feats)
    
    @staticmethod
    def make_graph(node, adjc, edge, feat):
        # The node features
        x = node.astype(float)
        
        # The adjacency matrix
        # convert to scipy.sparse matrix
        a = adjc.astype(int)
        a = sp.csr_matrix(a)
        # check shape (n_nodes, n_nodes)
        assert a.shape[0] == len(node)
        assert a.shape[1] == len(node)
        
        # The labels
        y = feat.astype(float)
        
        # The edge features 
        e = edge.astype(float)
        # check shape (n_nodes, n_nodes, ..)
        assert e.shape[0] == len(node)
        assert e.shape[1] == len(node)
        
        return Graph(x=x, a=a, e=e, y=y)
