import numpy as np
from scipy.linalg import hadamard
import matplotlib.pyplot as plt


def generate_patterns(px):
    '''Function that generates Hadamard basis pattern for differential measurements 
    to display on a DMD
    px - size of the reconstructed image'''

    #Hadmard matrix
    hadamard_mat = hadamard(px**2)

    #reshape into 3d structure
    hadamard_patterns = hadamard_mat.reshape(px, px, px**2)

    #DMD patterns
    M_plus = 0.5*(hadamard_patterns + 1)
    M_minus = 1 - M_plus

    #collect patterns into one array
    DMD_patterns = np.empty((px, px, 2*px**2))
    DMD_patterns[:, :, 0::2] = M_plus
    DMD_patterns[:, :, 1::2] = M_minus

    return DMD_patterns

def enlarge_pattern(pattern, factor):
    '''Function that makes pattern larger, i.e. one entry in original matrix is
    represented by k x k 
    factor = k'''

    pattern = np.kron(pattern, np.ones((factor, factor)))

    return pattern
