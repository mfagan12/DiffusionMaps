import numpy as np

def generate_toy_ellipse(n_points=1000, noisiness=1/80, r1=1, r2=1):
    '''
    Generate data in the shape of a noisy ellipse.
    
    Arguments:
    n_points -- number of sample points
    noisiness -- strength of noise effect. 0 implies no noise.
    r1 -- radius of ellipse along x-axis
    r2 -- radius of ellipse along y-axis
    '''
    X = np.linspace(0, 2*np.pi , n_points)
    X = np.array((r1 * np.cos(X), r2 * np.sin(X)))
    epsilon = np.random.randn(2, n_points) * noisiness
    return X + epsilon

def generate_broken_ellipse(n_points=1000, noisiness=1/80, r1=1, r2=1):
    '''
    Generate data in the shape of a noisy ellipse.
    
    Arguments:
    n_points -- number of sample points
    noisiness -- strength of noise effect. 0 implies no noise.
    r1 -- radius of ellipse along x-axis
    r2 -- radius of ellipse along y-axis
    '''
    X = np.linspace(0, (3/2)*np.pi , n_points)
    X = np.array((r1 * np.cos(X), r2 * np.sin(X)))
    epsilon = np.random.randn(2, n_points) * noisiness
    return X + epsilon

def generate_toroidal_helix(n_points, noisiness, R, r, n):
    X = np.linspace(0, 2*np.pi , n_points)
    X = np.array(((R + r * np.cos(n*X)) * np.cos(X), 
                  (R + r * np.cos(n*X)) * np.sin(X), 
                   r*np.sin(n*X)))
    epsilon = np.random.randn(3, n_points) * noisiness
    return X + epsilon

if __name__ == "__main__":
    pass
