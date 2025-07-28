import numpy as np

def bulk_field_profile(y, Δ, φ0=1.0):
    """
    Simulates scalar field near AdS boundary:
        φ(y) ≈ y^Δ * φ0 + higher terms
    Parameters:
        y   : array — extra-dimensional coordinate
        Δ   : float — scaling dimension
        φ0  : float — boundary amplitude
    Returns:
        φ : array — bulk field values
    """
    return φ0 * y**Δ + 0.5 * φ0 * y**(Δ + 2)  # Truncated expansion

def extract_boundary_wavefunction(φ_y, Δ, y_vals):
    """
    Projects bulk field to boundary wavefunction.
    Assumes φ ~ y^Δ * ψ(x, t)
    Parameters:
        φ_y : array — bulk field sampled over y
        Δ   : float — scaling dimension
        y_vals : array — corresponding y values
    Returns:
        ψ : float — approximate boundary wavefunction value
    """
    return φ_y / y_vals**Δ

def schrodinger_rhs(ψ, dx, mass, potential=None):
    """
    Computes right-hand side of time-dependent Schrödinger equation.
    Parameters:
        ψ : array — wavefunction over space
        dx : float — spatial resolution
        mass : float — effective particle mass
        potential : array — optional potential V(x)
    Returns:
        dψ_dt : array — time derivative of ψ
    """
    laplacian = (np.roll(ψ, -1) - 2 * ψ + np.roll(ψ, 1)) / dx**2
    V = potential if potential is not None else np.zeros_like(ψ)
    return -1j * (-0.5 * laplacian / mass + V * ψ)

def evolve_wavefunction(ψ0, steps, dt, dx, mass, potential=None):
    """
    Evolves boundary wavefunction in time using Schrödinger dynamics.
    Parameters:
        ψ0 : initial wavefunction
        steps : number of time steps
        dt : time step size
        dx : spatial resolution
        mass : effective mass
        potential : potential array (optional)
    Returns:
        ψ : final wavefunction
    """
    ψ = np.copy(ψ0)
    for _ in range(steps):
        dψ = schrodinger_rhs(ψ, dx, mass, potential)
        ψ += dψ * dt
    return ψ
