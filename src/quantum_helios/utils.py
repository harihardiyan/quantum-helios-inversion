import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from typing import Tuple, List
from .physics import forward_observation_1AU, R_SUN

class StandardScalerJAX:
    """
    JAX-compatible standard scaler for feature normalization.
    Ensures input features have mean=0 and std=1 for stable VQC training.
    """
    mu: jnp.ndarray
    sigma: jnp.ndarray

    def __init__(self, mu: jnp.ndarray, sigma: jnp.ndarray):
        self.mu = mu
        self.sigma = sigma

    @staticmethod
    def fit(X: jnp.ndarray):
        mu = jnp.mean(X, axis=0)
        sigma = jnp.std(X, axis=0) + 1e-12
        return StandardScalerJAX(mu, sigma)

    def transform(self, X: jnp.ndarray) -> jnp.ndarray:
        return (X - self.mu) / self.sigma

    def inverse_transform(self, X_scaled: jnp.ndarray) -> jnp.ndarray:
        return X_scaled * self.sigma + self.mu

def train_test_split_jax(X: jnp.ndarray, Y: jnp.ndarray, test_ratio: float = 0.25, key: jr.PRNGKey = jr.PRNGKey(0)):
    """
    Splits the dataset into training and testing sets using JAX PRNG.
    """
    n_samples = X.shape[0]
    indices = jnp.arange(n_samples)
    shuffled_indices = jr.permutation(key, indices)
    
    n_test = int(test_ratio * n_samples)
    test_idx = shuffled_indices[:n_test]
    train_idx = shuffled_indices[n_test:]
    
    return X[train_idx], Y[train_idx], X[test_idx], Y[test_idx]

def synthesize_dataset_np(seed: int = 42, N: int = 128, noise_sigma: float = 0.03):
    """
    Generates a synthetic heliospheric dataset based on Parker Spiral & Lambert W solutions.
    Used for benchmarking the PIQNN model.
    """
    rng = np.random.default_rng(seed)
    
    # Astrophysical Parameter Ranges
    Br_r0_range = (1e-5, 6e-5)         # Tesla (Magnetic field at source)
    r0_range    = (2.0*R_SUN, 3.0*R_SUN) # Meters (Critical radius)
    cs_range    = (3e4, 8e4)           # m/s (Sound speed)
    vr_range    = (3.0e5, 7.0e5)       # m/s (Wind speed)
    n_p_range   = (2e6, 10e6)          # m^-3 (Proton density)

    X_list, Y_list = [], []
    
    from dataclasses import dataclass
    @dataclass
    class LocalParams:
        Br_r0: float; r0: float; cs: float; vr: float; n_p: float

    for _ in range(N):
        p = LocalParams(
            Br_r0 = rng.uniform(*Br_r0_range),
            r0    = rng.uniform(*r0_range),
            cs    = rng.uniform(*cs_range),
            vr    = rng.uniform(*vr_range),
            n_p   = rng.uniform(*n_p_range)
        )
        
        feats, targs = forward_observation_1AU(p)
        
        # Add observational noise
        noise = rng.normal(0.0, noise_sigma, size=feats.shape)
        feats_noisy = feats * (1.0 + noise)
        
        X_list.append(feats_noisy)
        Y_list.append(targs)
    
    return (jnp.array(np.stack(X_list), dtype=jnp.float64), 
            jnp.array(np.stack(Y_list), dtype=jnp.float64),
            Br_r0_range, r0_range, cs_range)

def calculate_metrics(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> dict:
    """
    Calculates scientific regression metrics: RMSE and Relative MAE.
    """
    rmse = jnp.sqrt(jnp.mean((y_true - y_pred)**2, axis=0))
    # Relative MAE to handle scaling differences between Tesla and Meters
    rel_mae = jnp.mean(jnp.abs(y_true - y_pred) / (jnp.abs(y_true) + 1e-12), axis=0)
    
    return {
        "rmse": rmse,
        "rel_mae": rel_mae
    }
