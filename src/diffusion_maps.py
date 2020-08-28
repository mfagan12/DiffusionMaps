import numpy as np
#from data import generate_toy_ellipse
from pdb import set_trace
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from numpy.linalg import matrix_power

def rbf_kernel(epsilon=2):
    '''
    Implements the radial basis function kernel, aka the Gaussian kernel.
    
    Arguments:
    '''
    return lambda X: np.exp(-squareform(pdist(X.T))**2 / epsilon)

def polynomial_kernel(d=2):
    '''
    Implements the polynomial kernel.
    
    Arguments:
    '''
    return lambda x,y: (x.T @ y + 1)**d
    raise NotImplementedError
    
def cknn_kernel(bandwidth=1/2):
    '''
    Implements the continuous k-nearest neighbors kernel function.
    '''
    raise NotImplementedError

def kernel_matrix(kernel, X):
    '''
    Computes the pairwise similarity matrix for a matrix of vectors
    according to a given kernel function.
    '''
    # This implementation can almost certainly be optimized,
    # at least by not computing the elements above and below
    # the diagonal.
#     N = X.shape[1]
#     K = np.zeros((N,N))
#     for i in tqdm(range(N)):
#         for j in range(N):
#             K[i,j] = kernel(X[:,i], X[:,j])
            
    K = kernel(X)
    return K

def normalize_kernel(K, alpha=1):
    '''
    Computes the graph Laplacian normalization of a similarity matrix.
    '''
    D = np.diag(K.sum(axis=1)**(-alpha))
    L_alpha = D @ K @ D
    D_alpha = np.diag(L_alpha.sum(axis=1)**(-1))
    M = D_alpha @ L_alpha
    return (M + M.T) / 2

def normalize_kernel2(K):
    '''
    Computes the graph Laplacian normalization of a similarity matrix.
    '''
    D = np.diag(K.sum(axis=1)**(-1))
    return D @ K

def diffusion_maps(X, kernel, alpha=1/2, t=1):
    '''
    Compute the diffusion map embedding for given data and kernel.
    
    Arguments:
        X -- 
        kernel -- 
        dims -- 
    '''
    K1 = kernel_matrix(kernel, X)
#     set_trace()
#     K2 = normalize_kernel(K1, alpha)
    K2 = normalize_kernel2(K1)
#     set_trace()
    D = np.linalg.eigh(matrix_power(K2, t))
#     set_trace()
    return D

def diffusion_embedding(D, k=2):
    eigvals, eigvecs = D
    eigvals, eigvecs = eigvals[::-1], eigvecs[:,::-1]
#     set_trace()
    Phi = eigvecs * eigvals
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
