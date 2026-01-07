import jax.numpy as jnp
import numpy as np
import scipy.special as sp

# Physical Constants (SI Units)
R_SUN = 6.957e8
AU    = 1.496e11
G     = 6.67430e-11
M_SUN = 1.98847e30
OMEGA = 2.9e-6
MU0   = 4 * np.pi * 1e-7
M_P   = 1.6726219e-27

def parker_u_at_r_supersonic(cs, r):
    """Analytical solution for supersonic Parker wind via Lambert W function."""
    rc = (G * M_SUN) / (2.0 * cs**2)
    RHS = (rc / r)**2 * jnp.exp(2.0 * rc / r - 1.0)
    arg = -jnp.clip(RHS**2, -1.0/np.e, 0.0)
    # Branch k=-1 for supersonic solutions
    y = -sp.lambertw(np.array(arg), k=-1).real
    u = cs * jnp.sqrt(jnp.maximum(y, 0.0))
    return jnp.array(u, dtype=jnp.float64)

def physics_residuals(features_unscaled, y_pred_phys):
    """Calculates Differentiable Parker Wind Residuals."""
    Br, Bphi = features_unscaled[:, 0], features_unscaled[:, 1]
    psi_deg, u_1AU, Pmag_meas = features_unscaled[:, 3], features_unscaled[:, 4], features_unscaled[:, 6]
    cs_hat = y_pred_phys[:, 2]
    
    # ODE Consistency Check
    eps = 1e-30
    denom = (u_1AU**2 - cs_hat**2) + eps
    num = u_1AU * ((2.0 * cs_hat**2) / AU - (G * M_SUN) / (AU**2))
    du_rhs = num / denom
    wind_pen = jnp.mean(du_rhs**2)
    
    # Magnetic Consistency
    psi_rad = jnp.deg2rad(psi_deg)
    tan_psi_model = jnp.abs(Bphi) / (jnp.abs(Br) + eps)
    pitch_pen = jnp.mean((jnp.tan(psi_rad) - tan_psi_model)**2)
    
    return wind_pen + pitch_pen
