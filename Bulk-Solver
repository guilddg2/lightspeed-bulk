import numpy as np

def potential(phi, R5, params):
    """
    Computes scalar potential with geometry coupling.
    Parameters:
        phi : numpy array — scalar field values
        R5  : float — 5D Ricci scalar
        params : dict — model parameters
    Returns:
        V(phi) : numpy array
    """
    v, λ1, λ2, λ3, k = params['v'], params['λ1'], params['λ2'], params['λ3'], params['k']
    return λ1 * (phi**2 - v**2)**2 + λ2 * phi * R5 + λ3 * np.exp(-2 * k * phi**4)

def evolve_scalar(phi0, dt, dx, steps, R5, params):
    """
    Time evolution of scalar field using finite difference method.
    Parameters:
        phi0 : initial field values
        dt   : time step
        dx   : spatial resolution
        steps : total evolution steps
        R5   : background curvature
        params : potential parameters
    Returns:
        phi : evolved field array
    """
    phi = np.copy(phi0)
    for t in range(steps):
        laplacian = (np.roll(phi, -1) - 2 * phi + np.roll(phi, 1)) / dx**2
        dphi_dt = -laplacian - potential(phi, R5, params)
        phi += dphi_dt * dt
    return phi
