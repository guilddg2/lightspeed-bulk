import numpy as np
from scipy.optimize import curve_fit

def dispersion_model(r, a, c):
    """
    Power-law model for radial velocity dispersion:
        σ(r) = c * r^(-a)
    Parameters:
        r : array — radial distance (kpc)
        a : float — decay exponent
        c : float — normalization factor
    Returns:
        σ(r) : array — predicted dispersion
    """
    return c * r**(-a)

def fit_dispersion(r_vals, sigma_vals):
    """
    Fits the dispersion model to galaxy kinematics data.
    Parameters:
        r_vals : array — radii (e.g., from Gaia)
        sigma_vals : array — observed velocity dispersion
    Returns:
        popt : list — best-fit [a, c]
        pcov : 2x2 matrix — parameter covariance
    """
    popt, pcov = curve_fit(dispersion_model, r_vals, sigma_vals, p0=[0.5, 200])
    return popt, pcov

def log_residuals(r_vals, sigma_vals, params):
    """
    Computes residuals in log space for visualizing deviations.
    Parameters:
        r_vals : array — radii
        sigma_vals : array — observations
        params : list — fitted [a, c]
    Returns:
        residuals : array — log10 ratio between observed and predicted
    """
    model_vals = dispersion_model(r_vals, *params)
    return np.log10(sigma_vals / model_vals)
