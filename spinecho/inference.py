import numpy as np
from scipy.linalg import cho_solve, cho_factor

# Tool functions for Bayesian inference
def rbf_kernel(Q, lambda_):
    """
    Compute the spatial correlation kernel k_{i j}.

    Parameters:
        Q (numpy.ndarray): Scattering vector magnitude (1D array).
        lambda_ (float): Length scale parameter for the RBF kernel.

    Returns:
        numpy.ndarray: Spatial correlation kernel matrix of shape (len(Q), len(Q)).
    """
    Q_i, Q_j = np.meshgrid(Q, Q, indexing='ij')
    return np.exp(-((Q_i - Q_j) ** 2) / (2 * lambda_ ** 2))


def construct_transformation_tensor(Q, t, tau, bg_mode=False):
    """
    Compute the transformation tensor C_{i m n}.

    Parameters:
        Q (numpy.ndarray): Scattering vector magnitude (1D array).
        t (numpy.ndarray): Time points (1D array).
        tau (numpy.ndarray): Relaxation times (1D array).
        bg_mode (bool): Whether to include a background mode (default False).

    Returns:
        numpy.ndarray: Transformation tensor of shape (len(Q), len(t), len(tau) + bg_mode).
    """
    C = np.exp(-t[:, None] / tau[None, :])  # Shape (M, N)
    if bg_mode:
        C = np.hstack([np.ones((t.shape[0], 1)), C])
    return np.tile(C[None, :, :], (len(Q), 1, 1))  # Expand to (L, M, N)


def construct_transformation_matrix(t, tau, bg_mode=False):
    """
    Compute the transformation matrix C_{a n}.

    Parameters:
        t (numpy.ndarray): Time points (1D array).
        tau (numpy.ndarray): Relaxation times (1D array).
        bg_mode (bool): Whether to include a background mode (default False).

    Returns:
        numpy.ndarray: Transformation matrix of shape (len(t), len(tau) + bg_mode).
    """
    C = np.exp(-t[:, None] / tau[None, :])  # Shape (M, N)
    if bg_mode:
        C = np.hstack([np.ones((t.shape[0], 1)), C])
    return C

# -----------------------------------------------------------------------------------------
# Bayesian inference function for grid-based scattering data
def bayesian_inference_grid(S_exp, delta_S_exp, Q, t, tau, lambda_, bg_mode=False):
    """
    Perform Bayesian inference to estimate coefficients and their covariance.

    This function performs Bayesian inference using a grid-based approach to estimate 
    the coefficients A_{i n} and their covariance K_GPR_{i, n, j, q} for a given 
    experimental dataset S_exp.

    Parameters:
        S_exp (numpy.ndarray): Experimental data matrix of shape (L, M), where L is the 
            number of spatial points and M is the number of time points.
        Q (numpy.ndarray): Scattering vector magnitude (1D array of length L).
        t (numpy.ndarray): Time points (1D array of length M).
        tau (numpy.ndarray): Expansion parameters (1D array).
        delta_S_exp (numpy.ndarray): Uncertainty in the experimental data S_exp, 
            with shape (L, M).
        lambda_ (float): Length scale parameter for the radial basis function (RBF) kernel.
        bg_mode (bool, optional): If True, includes a background mode in the expansion. 
            Default is False.

    Returns:
        tuple: 
            - A_GPR (numpy.ndarray): Posterior mean of the coefficients, with shape (L, N), 
              where N is the number of expansion parameters (including background mode if enabled).
            - K_GPR (numpy.ndarray): Posterior covariance of the coefficients, with shape 
              (L, N, L, N).
    """
    L, M = S_exp.shape  # L: spatial points, M: time points
    N = len(tau) + (1 if bg_mode else 0)  # Expansion parameter size, adjusted for background mode

    # Construct transformation tensor C_{i m n}
    C = construct_transformation_tensor(Q, t, tau, bg_mode)  # Shape (L, M, N)

    # Construct spatial covariance K_{i j}
    K = rbf_kernel(Q, lambda_)  # Shape (L, L)

    # Construct noise covariance Sigma_{i m, j p}
    Sigma = np.einsum('im,ij,mp->imjp', delta_S_exp**2, np.eye(L), np.eye(M))  # Shape (L, M, L, M)

    # Expand K to K_{i n, j q} using Kronecker product property
    K_expanded = np.einsum('ij,nq->injq', K, np.eye(N))  # Shape (L, N, L, N)

    # Construct modified covariance tensor \tilde{\Sigma}_{i, m, j, p}
    K_tilde = Sigma + np.einsum('imn,injq,jpq->imjp', C, K_expanded, C)  # Shape (L, M, L, M)

    # Reshape for inversion
    K_tilde_reshaped = K_tilde.reshape(L * M, L * M)
    K_tilde_inv = cho_solve(cho_factor(K_tilde_reshaped), np.eye(L * M))
    K_tilde_inv = K_tilde_inv.reshape(L, M, L, M)  # Restore shape

    # Compute posterior mean A_GPR_{i n}
    K_expanded_C = np.einsum('injq,jmq->imn', K_expanded, C)  # Shape (L, M, N)
    A_GPR = np.einsum('imn,imjp,jp->in', K_expanded_C, K_tilde_inv, S_exp)  # Shape (L, N)

    # Compute posterior covariance K_GPR_{i, n, j, q}
    K_GPR = K_expanded - np.einsum('imn,imjp,jpq->injq', K_expanded_C, K_tilde_inv, K_expanded_C)

    return A_GPR, K_GPR


# Bayesian inference function for off-grid scattering data
def bayesian_inference(S_exp, delta_S_exp, Q_obs, t_obs, Q_eval, t_eval, tau, mu_, lambda_, bg_mode=False):
    """
        Parameters:
            S_exp (numpy.ndarray): Experimental scattering data (1D array of size M).
            Q_obs (numpy.ndarray): Observed scattering vector magnitudes (1D array of size M).
            t_obs (numpy.ndarray): Observed time points (1D array of size M).
            Q_eval (numpy.ndarray): Evaluation scattering vector magnitudes (1D array of size L).
            t_eval (numpy.ndarray): Evaluation time points (1D array).
            tau (numpy.ndarray): Relaxation times (1D array of size N).
            delta_S_exp (numpy.ndarray): Measurement noise standard deviations for S_exp (1D array of size M).
            mu_ (float): Length scale for the spatial kernel.
            lambda_ (float): Length scale for the prior spatial kernel.
            bg_mode (bool, optional): Whether to include a background mode in the model (default False).

        Returns:
            tuple:
                - A_GPR (numpy.ndarray): Posterior mean of the coefficients (2D array of shape (L, N)).
                - K_GPR (numpy.ndarray): Posterior covariance matrix of the coefficients (4D array of shape (L, N, L, N)).
                - S_reconstructed (numpy.ndarray): Reconstructed scattering function at evaluation points (1D array of size L).
    """ 
    L, M, N = len(Q_eval), len(Q_obs), len(tau)
    print(f"L: {L}, M: {M}, N: {N}")

    # Compute normalized spatial kernel (M x L)
    spatial_kernel = np.exp(-((Q_obs[:, None] - Q_eval[None, :]) ** 2) / (2 * mu_ ** 2))
    spatial_kernel /= spatial_kernel.sum(axis=1, keepdims=True)

    # Compute temporal kernel (M x N)
    temporal_kernel = np.exp(-np.outer(t_obs, 1 / tau))

    # Construct design matrix G (M x LN)
    G = (spatial_kernel[:, :, None] * temporal_kernel[:, None, :]).reshape(M, L * N)

    # Compute prior covariance matrix (LN x LN)
    Q_dist = (Q_eval[:, None] - Q_eval[None, :]) ** 2
    spatial_prior_kernel = np.exp(-Q_dist / (2 * lambda_ ** 2))
    K_prior = np.kron(np.eye(N), spatial_prior_kernel)

    # Compute measurement noise covariance matrix (M x M)
    Sigma = np.diag(delta_S_exp ** 2)

    # Compute posterior mean using Cholesky decomposition
    GK = G @ K_prior
    K_tilde = GK @ G.T + Sigma
    cho_K_tilde = cho_factor(K_tilde + 1e-8 * np.eye(M))
    A_GPR_flat = K_prior @ G.T @ cho_solve(cho_K_tilde, S_exp)

    # Reshape posterior mean to (L, N)
    A_GPR = A_GPR_flat.reshape(L, N)

    # Compute posterior covariance matrix (L, N, L, N)
    K_GPR_flat = K_prior - K_prior @ G.T @ cho_solve(cho_K_tilde, G @ K_prior)
    K_GPR = K_GPR_flat.reshape(L, N, L, N)

    # Reconstruct scattering function at evaluation points
    spatial_kernel_eval = np.exp(-((Q_eval[:, None] - Q_eval[None, :]) ** 2) / (2 * lambda_ ** 2))
    spatial_kernel_eval /= spatial_kernel_eval.sum(axis=1, keepdims=True)
    temporal_kernel_eval = np.exp(-np.outer(t_eval, 1 / tau))
    S_reconstructed = np.einsum('ij,jn,in->i', spatial_kernel_eval, A_GPR, temporal_kernel_eval)

    return A_GPR, K_GPR, S_reconstructed

# Bayesian inference function for off-grid scattering data, explicitly using the einsum notation (slower but clearer)
def bayesian_inference_ein(S_exp, delta_S_exp, Q_obs, t_obs, Q_eval, t_eval, tau, mu_, lambda_, bg_mode=False):
    """
    Perform Bayesian inference and reconstruct the scattering function using Gaussian Process Regression.

    Parameters:
        S_exp (numpy.ndarray): Experimental scattering data (1D array of size A).
        Q_obs (numpy.ndarray): Observed scattering vector magnitudes (1D array of size A).
        t_obs (numpy.ndarray): Observed time points (1D array of size A).
        Q_eval (numpy.ndarray): Evaluation scattering vector magnitudes (1D array of size L).
        t_eval (numpy.ndarray): Evaluation time points (1D array).
        tau (numpy.ndarray): Relaxation times (1D array of size N).
        delta_S_exp (numpy.ndarray): Measurement noise standard deviations for S_exp (1D array of size A).
        mu_ (float): Length scale for the spatial kernel.
        lambda_ (float): Length scale for the prior spatial kernel.
        bg_mode (bool, optional): Whether to include a background mode in the model (default False).

    Returns:
        tuple:
            - A_GPR (numpy.ndarray): Posterior mean of the scattering function (2D array of shape (L, N)).
            - K_GPR (numpy.ndarray): Posterior covariance matrix of the scattering function (4D array of shape (L, N, L, N)).
            - S_reconstructed (numpy.ndarray): Reconstructed scattering function at evaluation points (1D array of size L).
    """
    L, A, N = len(Q_eval), len(Q_obs), len(tau)

    # Compute normalized spatial kernel (A x L)
    spatial_kernel = np.exp(-((Q_obs[:, None] - Q_eval[None, :]) ** 2) / (2 * mu_ ** 2))
    spatial_kernel /= spatial_kernel.sum(axis=1, keepdims=True)

    # Compute temporal kernel (A x N)
    temporal_kernel = np.exp(-np.outer(t_obs, 1 / tau))

    # Construct design matrix G (A x L x N)
    G = np.einsum('al,an->aln', spatial_kernel, temporal_kernel)

    # Compute prior covariance matrix (L x N x L x N)
    Q_dist = (Q_eval[:, None] - Q_eval[None, :]) ** 2
    spatial_prior_kernel = np.exp(-Q_dist / (2 * lambda_ ** 2))
    K_prior = np.einsum('ij,nm->injm', spatial_prior_kernel, np.eye(N)).reshape(L * N, L * N)

    # Compute measurement noise covariance matrix (A x A)
    Sigma = np.diag(delta_S_exp ** 2)

    # Compute posterior mean using Cholesky decomposition
    G_flat = G.reshape(A, L * N)
    K_tilde = np.einsum('ai,ij,bj->ab', G_flat, K_prior, G_flat) + Sigma
    cho_K_tilde = cho_factor(K_tilde + 1e-8 * np.eye(A))
    A_GPR_flat = K_prior @ G_flat.T @ cho_solve(cho_K_tilde, S_exp)

    # Reshape posterior mean to (L, N)
    A_GPR = A_GPR_flat.reshape(L, N)

    # Compute posterior covariance matrix (L, N, L, N)
    K_GPR_flat = K_prior - K_prior @ G_flat.T @ cho_solve(cho_K_tilde, G_flat @ K_prior)
    K_GPR = K_GPR_flat.reshape(L, N, L, N)

    # Reconstruct scattering function at evaluation points
    spatial_kernel_eval = np.exp(-((Q_eval[:, None] - Q_eval[None, :]) ** 2) / (2 * lambda_ ** 2))
    spatial_kernel_eval /= spatial_kernel_eval.sum(axis=1, keepdims=True)
    temporal_kernel_eval = np.exp(-np.outer(t_eval, 1 / tau))
    S_reconstructed = np.einsum('ijn,in->j', np.einsum('ij,jn->ijn', spatial_kernel_eval, temporal_kernel_eval), A_GPR)

    return A_GPR, K_GPR, S_reconstructed
