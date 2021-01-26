import numpy as np


def str_is_float(s):
    """Check whether string 
    can be converted to a float
    """
    try:
        float(s)
        return True
    
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    
    except (TypeError, ValueError):
        pass
 
    return False


def symmetrize(matrix):
    return matrix + matrix.T - np.diag(matrix.diagonal())