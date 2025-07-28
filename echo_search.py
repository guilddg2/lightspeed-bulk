import numpy as np

def echo_waveform(t, freqs, Q, phi0=0.0):
    """
    Generates a gravitational wave echo template.
    Parameters:
        t     : array — time vector
        freqs : list — central frequencies of echoes
        Q     : float — damping factor (quality)
        phi0  : float — initial phase
    Returns:
        h(t) : array — echo signal
    """
    h = np.zeros_like(t)
    for n, fn in enumerate(freqs, start=1):
        damping = np.exp(-t / Q)
        phase_shift = phi0 + n * np.pi
        h += damping * np.cos(2 * np.pi * fn * t + phase_shift)
    return h

def snr(signal, noise_psd):
    """
    Computes simplified Signal-to-Noise Ratio.
    Parameters:
        signal : array — GW strain signal
        noise_psd : array — noise power spectral density
    Returns:
        SNR : float
    """
    return 4 * np.sum(np.abs(signal)**2 / noise_psd)

def delay_profile(t, model='planck_echo', params=None):
    """
    Adds model-based delay profile to simulate exotic echoes.
    Parameters:
        t      : array — time values
        model  : str — delay model name
        params : dict — model-specific parameters
    Returns:
        delay : array — delay-modulated response
    """
    if model == 'planck_echo':
        σ, Δt = params.get('σ', 0.01), params.get('Δt', 0.2)
        return np.exp(-((t - Δt)**2) / (2 * σ**2))
    elif model == 'wormhole':
        decay = params.get('decay', 0.05)
        return np.exp(-decay * t) * np.sin(2 * np.pi * params.get('f', 100) * t)
    else:
        return np.zeros_like(t)
