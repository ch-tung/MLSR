import numpy as np

def f_sqt_chain(Q, t, N=100, b=1.0, D=1.0, tau_R=1.0, p_max=10):
    """
    Compute the scattering function S(Q, t) for a Gaussian polymer chain.

    Parameters:
        Q (numpy.ndarray): Scattering vector magnitude (2D array).
        t (numpy.ndarray): Time (2D array).
        N (int): Number of beads in the polymer chain (default 100).
        b (float): Bond length (default 1.0).
        D (float): Diffusion coefficient (default 1.0).
        tau_R (float): Relaxation time (default 1.0).
        p_max (int): Number of terms in the summation for Phi_nm^1 (default 10).

    Returns:
        numpy.ndarray: Scattering function S(Q, t) (2D array).
    """
    n_values, m_values = np.meshgrid(np.arange(N), np.arange(N))
    abs_nm = np.abs(n_values - m_values)

    Phi_D = 6 * D * t
    Phi_nm_0 = abs_nm * b**2
    Phi_nm_1 = ((4 * N * b**2 / np.pi**2) * 
                np.sum([(1 / p**2) * np.einsum('kl,ij->ijkl', 
                                               np.cos(p * np.pi * n_values / N) * np.cos(p * np.pi * m_values / N), 
                                               (1 - np.exp(-t * p**2 / tau_R))) 
                        for p in range(1, p_max+1)], axis=0))
    Phi_nm = Phi_D[:, :, None, None] + Phi_nm_0[None, None, :, :] + Phi_nm_1

    Q_squared = Q**2
    Q_squared_expanded = Q_squared[:, :, None, None]
    sqt = (1 / N) * np.einsum('ijkl,ijkl->ij', np.exp(- (1/6) * Q_squared_expanded * Phi_nm), np.ones_like(Phi_nm))

    return sqt

def f_sqt_chain_offgrid(Q, t, N=100, b=1.0, D=1.0, tau_R=1.0, p_max=10):
    """
    Compute the scattering function S(Q, t) for a Gaussian polymer chain.

    Parameters:
        Q (numpy.ndarray): Scattering vector magnitude (1D array)
        t (numpy.ndarray): Time (1D array)
        N (int): Number of beads in the polymer chain (default 100)
        b (float): Bond length (default 1.0)
        D (float): Diffusion coefficient (default 1.0)
        tau_R (float): Relaxation time (default 1.0)
        p_max (int): Number of terms in the summation for Phi_nm^1 (default 50)

    Returns:
        numpy.ndarray: Value of S(Q, t) (1D array)
    """

    # Indices for polymer beads
    n_values, m_values = np.meshgrid(np.arange(N), np.arange(N))
    abs_nm = np.abs(n_values - m_values)

    # Compute Phi_nm(t) components
    Phi_D = 6 * D * t  # Center-of-mass diffusion
    Phi_nm_0 = abs_nm * b**2  # Gaussian chain contribution

    # Compute Phi_nm^1 (dynamic part)
    Phi_nm_1 = ((4 * N * b**2 / np.pi**2) * 
                np.sum([(1 / p**2) * np.einsum('ij,k->ijk', 
                                               np.cos(p * np.pi * n_values / N) * np.cos(p * np.pi * m_values / N), 
                                               (1 - np.exp(-t * p**2 / tau_R))) 
                        for p in range(1, p_max+1)], # sum over p
                       axis=0))
    
    # Total Phi_nm
    Phi_nm = Phi_D[None, None, :] + Phi_nm_0[:, :, None] + Phi_nm_1
    
    # Compute S(Q, t) using einsum
    Q_squared = Q**2  # Shape [i]
    Q_squared_expanded = Q_squared[None, None, :]  
    sqt = (1 / N) * np.einsum('ijk,ijk->k', np.exp(- (1/6) * Q_squared_expanded * Phi_nm), np.ones_like(Phi_nm))


    return sqt


def f_sqt_sample_t(Q, t, N_count=1000, N=100, b=1.0, D=1.0, tau_R=1.0, p_max=10, seed=None, bg=0.01):
    """
    Generate time-sampled scattering function S(Q, t) with noise.

    Parameters:
        Q (numpy.ndarray): Scattering vector magnitude (2D array).
        t (numpy.ndarray): Time (2D array).
        N_count (int): Total counts for each Q (default 1000).
        N (int): Number of beads in the polymer chain (default 100).
        b (float): Bond length (default 1.0).
        D (float): Diffusion coefficient (default 1.0).
        tau_R (float): Relaxation time (default 1.0).
        p_max (int): Number of terms in the summation for Phi_nm^1 (default 10).
        seed (int): Random seed for reproducibility (default None).
        bg (float): Background noise (default 0.01).

    Returns:
        tuple: Sampled S(Q, t), uncertainty, and true S(Q, t) with background.
    """
    np.random.seed(seed)
    sqt = f_sqt_chain(Q, t, N=N, b=b, D=D, tau_R=tau_R, p_max=p_max) / N + bg
    n_t, n_Q = Q.shape

    Delta_sqt_sample = np.zeros((n_t, n_Q))
    sqt_sample = np.zeros((n_t, n_Q))
    for i in range(n_Q):
        fit = sqt[:, i]
        pit = fit / np.sum(fit)
        Nit = N_count * pit
        Delta_sqt_sample[:, i] = np.sqrt(Nit) / N_count * np.sum(fit)

        indices = np.random.choice(len(t), size=N_count, p=pit)
        N_sample_i, _ = np.histogram(indices, bins=np.arange(len(t) + 1))
        sqt_sample[:, i] = N_sample_i / N_count * np.sum(fit)

    return sqt_sample, Delta_sqt_sample, sqt


def f_sqt_sample_Qt(Q, t, N_count=1000, N=100, b=1.0, D=1.0, tau_R=1.0, p_max=10, seed=None, bg=0.01):
    """
    Generate Q-t sampled scattering function S(Q, t) with noise.

    Parameters:
        Q (numpy.ndarray): Scattering vector magnitude (2D array).
        t (numpy.ndarray): Time (2D array).
        N_count (int): Total counts for each Q-t pair (default 1000).
        N (int): Number of beads in the polymer chain (default 100).
        b (float): Bond length (default 1.0).
        D (float): Diffusion coefficient (default 1.0).
        tau_R (float): Relaxation time (default 1.0).
        p_max (int): Number of terms in the summation for Phi_nm^1 (default 10).
        seed (int): Random seed for reproducibility (default None).
        bg (float): Background noise (default 0.01).

    Returns:
        tuple: Sampled S(Q, t), uncertainty, and true S(Q, t) with background.
    """
    np.random.seed(seed)
    sqt = f_sqt_chain(Q, t, N=N, b=b, D=D, tau_R=tau_R, p_max=p_max) / N + bg
    pqt = sqt / np.sum(sqt)
    Nqt = N_count * pqt
    Delta_sqt_sample = np.sqrt(Nqt) / N_count * np.sum(sqt)

    indices = np.random.choice(np.prod(Q.shape), size=N_count, p=pqt.flatten())
    sqt_sample = np.zeros_like(sqt)
    for idx in indices:
        t_idx, Q_idx = divmod(idx, Q.shape[1])
        sqt_sample[t_idx, Q_idx] += 1
    sqt_sample = sqt_sample / N_count * np.sum(sqt)

    return sqt_sample, Delta_sqt_sample, sqt


def f_sqt_sample_offgrid(Q, t, N_count=1000, N=100, b=1.0, D=1.0, tau_R=1.0, p_max=10, seed=None, bg=0.01):
    """
    Generate off-grid sampled scattering function S(Q, t) with noise.

    Parameters:
        Q (numpy.ndarray): Scattering vector magnitude (1D array).
        t (numpy.ndarray): Time (1D array).
        N_count (int): Total counts for each Q-t pair (default 1000).
        N (int): Number of beads in the polymer chain (default 100).
        b (float): Bond length (default 1.0).
        D (float): Diffusion coefficient (default 1.0).
        tau_R (float): Relaxation time (default 1.0).
        p_max (int): Number of terms in the summation for Phi_nm^1 (default 10).
        seed (int): Random seed for reproducibility (default None).
        bg (float): Background noise (default 0.01).

    Returns:
        tuple: Sampled S(Q, t), uncertainty, and true S(Q, t) with background.
    """
    np.random.seed(seed)
    sqt = f_sqt_chain_offgrid(Q, t, N=N, b=b, D=D, tau_R=tau_R, p_max=p_max) / N + bg
    pqt = sqt / np.sum(sqt)
    Nqt = N_count * pqt
    Delta_sqt_sample = np.sqrt(Nqt) / N_count * np.sum(sqt)

    indices = np.random.choice(len(Q), size=N_count, p=pqt)
    sqt_sample = np.zeros_like(Q)
    for idx in indices:
        sqt_sample[idx] += 1
    sqt_sample = sqt_sample / N_count * np.sum(sqt)

    return sqt_sample, Delta_sqt_sample, sqt


