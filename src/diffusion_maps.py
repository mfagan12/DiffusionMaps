import numpy as np
#from data import generate_toy_ellipse
from pdb import set_trace
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from numpy.linalg import matrix_power

array = np.ndarray

def rbf_kernel(epsilon: float = 1/2) -> callable:
    '''
    Implements the radial basis function kernel, aka the Gaussian kernel.

    Args:
        epsilon (float, optional): Bandwidth parameter. Defaults to 2.

    Returns:
        callable: quickly computes rbf similarity matrix for given data matrix X
    '''
    return lambda X: np.exp(-squareform(pdist(X.T), 'sqeuclidean') / epsilon)

def polynomial_kernel(d: int = 2) -> callable:
    '''
    Implements a polynomial kernel.

    Args:
        d (int, optional): Degree of polyomial. Defaults to 2.

    Returns:
        callable: quickly computes polynomial similarity matrix for given data
            matrix X
    '''
    return lambda x,y: (x.T @ y + 1)**d
    
def cknn_kernel(bandwidth=1/2):
    '''
    Implements the continuous k-nearest neighbors kernel function.
    '''
    raise NotImplementedError

def kernel_matrix(kernel: callable, X: array) -> array:
    '''
    Computes the pairwise similarity matrix for a matrix of vectors
    according to a given kernel function.
    '''        
    K = kernel(X)
    return K

def normalize_kernel(K: array, alpha: float = 1) -> array:
    '''
    Computes the graph Laplacian normalization of a similarity matrix.
    
    Arguments:
    K -- array -- pairwise similarity matrix generated from data via some kernel
    alpha -- float -- 
    
    Returns:
    array -- normalized version of similarity matrix
    '''
    # Diagonal matrix of row sums of K to the -alpha power
    D = np.diag(K.sum(axis=1)**(-alpha))
    K_hat = D @ K @ D
    
    D_hat = np.diag(K_hat.sum(axis=1)**(-1))
    K_bar = D_hat @ K_hat
    
    return (K_bar + K_bar.T) / 2

# def normalize_kernel2(K: array) -> array:
#     '''
#     Computes the graph Laplacian normalization of a similarity matrix.
#     '''
#     D = np.diag(K.sum(axis=1)**(-1))
#     return D @ K

def diffusion_maps(X: array, kernel: callable, alpha: float = 1) -> array:
    '''
    Compute the diffusion map embedding for given data and kernel.
    
    Arguments:
        X -- array -- data matrix, data assumed to be along columns, one row 
            per feature
        kernel -- function -- kernel function to compute similarity matrix from 
            data
        alpha -- float -- 
        t -- int -- 
    '''
    K1 = kernel_matrix(kernel, X)
    K2 = normalize_kernel(K1, alpha)
    D = np.linalg.eigh(K2)
    return D

def diffusion_embedding(D: array, k: int = 2, t: int = 1) -> array:
    eigvals, eigvecs = D
    eigvals, eigvecs = eigvals[::-1], eigvecs[:,::-1]
    Phi = eigvecs * eigvals**t
    return Phi[:,:k]
    
# class DiffusionMaps(BaseEstimator, TransformerMixin):
#     def __init__(data, kernel = rbf_kernel):
#         self.kernel = kernel
#         self.data = data
        
#     def fit(X, y = None):
#         return self
    
#     def transform(X):
#         return self

if __name__ == "__main__":
    X = generate_toy_ellipse()

    Y = diffusion_maps(X)

    plot_embedding(Y, dims=3)
