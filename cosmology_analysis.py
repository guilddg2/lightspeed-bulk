import numpy as np
from scipy.integrate import odeint

def friedmann_equations(y, t, params):
    """
    Solves modified Friedmann equations with bulk-induced corrections.
    Parameters:
        y : list — [a(t), H(t)]
        t : float — cosmic time
        params : dict — cosmological parameters
    Returns:
        dydt : list — time derivatives [da/dt, dH/dt]
    """
    a, H = y
    Ω_m, Ω_Λ, ε = params['Ω_m'], params['Ω_Λ'], params['ε']
    dadt = a * H
    dHdt = -0.5 * H**2 + ε * (a**(-3)) + Ω_Λ
    return [dadt, dHdt]

def solve_cosmology(a0, H0, t_span, params):
    """
    Evolves the scale factor and Hubble rate over cosmic time.
    Parameters:
        a0 : initial scale factor
        H0 : initial Hubble rate
        t_span : numpy array — cosmic time values
        params : dict — model parameters
    Returns:
        solution : array of [a(t), H(t)]
    """
    y0 = [a0, H0]
    return odeint(friedmann_equations, y0, t_span, args=(params,))

def luminosity_distance(a_vals):
    """
    Estimates luminosity distance vs redshift from scale factor evolution.
    Parameters:
        a_vals : array — scale factor values
    Returns:
        d_L : array — luminosity distances
    """
    z = 1.0 / np.array(a_vals) - 1.0
    return (1 + z) * np.cumsum(1.0 / a_vals)  # Simplified comoving integral
