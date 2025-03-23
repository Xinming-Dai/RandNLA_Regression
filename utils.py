import numpy as np
import argparse

def generate_cotaminated_error(n:int, delta:float, dist, param, seed) -> np.ndarray:
    """
    Generate cotaminated_error from a mixture of two distributions (Normal and dist).
    
    Parameters:
        n: Number of samples.
        delta: Contamination level (probability of the heavy-tailed noise).
        dist: Distribution of the heavy-tailed noise.  
        param: Parameter for the heavy-tailed noise distribution.
        seed: Random seed.
    
    Returns:
        epsilon: Array of n errors from the mixture model.
    """
    # Choose which distribution to sample from
    np.random.seed(seed)
    choices = np.random.choice([0, 1], size=n, p=[1-delta, delta])
    
    # Generate samples based on the choices
    normal_errors = np.random.normal(0, 1, size=n)  # Normal noise
    if dist == "laplace":
        heavy_tailed_errors = np.random.laplace(0, param, size=n)
    elif dist == "t":
        heavy_tailed_errors = np.random.standard_t(df=param, size=n)
    elif dist == "cauchy":
        heavy_tailed_errors = np.random.standard_cauchy(size=n)
    elif dist == "point_mass":
        heavy_tailed_errors = np.repeat(np.max(np.abs(normal_errors)) * param * 100, n)
    elif dist == "gamma":
        heavy_tailed_errors = np.random.gamma(param, 1, size=n)
    elif dist == "normal":
        heavy_tailed_errors = normal_errors
    else:
        raise ValueError("Invalid distribution")

    # Combine the samples based on the choice of distribution
    epsilon = np.where(choices == 0, normal_errors, heavy_tailed_errors)
    
    return epsilon


def coefficients(p:int, seed) -> np.ndarray:
    """
    Generate coefficients.

    Args:
        p: Number of features.

    Returns:
        Generated coefficient vector.
    """
    rng = np.random.default_rng(seed=seed)
    return rng.uniform(low=1, high=10, size=p)


def data_generation(coefficients, n:int=100, p:int=1, delta:float=0.1, dist="Laplace", param=1, seed=53):
    """
    Data generating process. Generates data for a linear regression model with Huber's contamination model for the noise.
    
    Args:
        n: Number of data points.
        p: Number of features.
        delta: Contamination level (probability of the heavy-tailed noise).
        hubers_scale: Scale parameter for the contamination noise distribution (Huber distribution).
    
    Returns:
        X: Generated feature matrix.
        Y: Generated target values.
    """
    np.random.seed(seed)
    X = np.random.uniform(-5, 5, size=(n, p))
    epsilon = generate_cotaminated_error(n, delta, dist, param, seed)
    Y = X @ coefficients + epsilon
    return X, Y

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed' , default=0, type=int,  help='Seed for Numpy. Default: 0 (None)')
    parser.add_argument('--n', default=1000, type=int, help='Number of samples. Default: 100')
    parser.add_argument('--p', default=5, type=int, help='Number of features. Default: 5')
    parser.add_argument('--delta', default=0.1, type=float, help='Contamination level. Default: 0.1')
    parser.add_argument('--dist', default='laplace', type=str, help='Distribution of the heavy-tailed noise. Default: laplace')
    parser.add_argument('--param', default=1, type=int, help='Parameter for the heavy-tailed noise distribution. Default: 1')
    parser.add_argument('--n_trials', default=10, type=int, help='Number of trials. Default: 10')
    parser.add_argument('--sketch_fn', default='proposal1', type=str, help='Sketch function. Default: clarkson_woodruff')
    parser.add_argument('--tau', default=1.35, type=float, help='Threshold for the Huber loss. Default: 1.35')
    parser.add_argument('--k', default=100, type=int, help='Number of rows in the sketch matrix. Default: 100')
    
    return parser.parse_args()