import numpy as np
from typing import List

def coefficients(p:int) -> np.ndarray:
    """
    Generate coefficients.

    Args:
        p: Number of features.

    Returns:
        Generated coefficient vector.
    """
    rng = np.random.default_rng(seed=42)
    return rng.uniform(low=1, high=10, size=p+1)

def generate_cotaminated_error(n:int, delta:float, hubers_scale:float) -> np.ndarray:
    """
    Generate cotaminated_error from a mixture of two distributions (Normal and Laplace).
    
    Parameters:
        n: Number of samples.
        delta: Contamination level (probability of the heavy-tailed noise).
        hubers_scale: Scale parameter for the contamination noise distribution (Huber distribution).
    
    Returns:
        epsilon: Array of n errors from the mixture model.
    """
    # Choose which distribution to sample from
    rng = np.random.default_rng(seed=42)
    choices = rng.choice([0, 1], size=n, p=[1-delta, delta])  # 0 for normal, 1 for Laplace
    
    # Generate samples based on the choices
    normal_errors = np.random.normal(0, 1, size=n)  # Normal noise
    heavy_tailed_errors = np.random.laplace(0, hubers_scale, size=n)  # Heavy-tailed noise (Laplace)
    
    # Combine the samples based on the choice of distribution
    epsilon = np.where(choices == 0, normal_errors, heavy_tailed_errors)
    
    return epsilon

def dgp(coefficients:List[float], n:int=100, p:int=1, delta:float=0.1, hubers_scale:float=1.0):
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
    rng = np.random.default_rng(seed=42)
    X = rng.uniform(-5, 5, size=(n, p))
    epsilon = generate_cotaminated_error(n, delta, hubers_scale)
    Y = coefficients[0] + X @ coefficients[1:] + epsilon
    
    return X, Y

if __name__ == "__main__":
    p = 3
    coef = coefficients(p)
    X, Y = dgp(coef, 10, p)
    print(X.shape, Y.shape)