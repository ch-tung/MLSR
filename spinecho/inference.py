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
def bayesian_inference(S_exp, delta_S_exp, Q_obs, t_obs, Q_eval, t_eval, tau,
                       mu_, lambda_, bg_mode=False, sigma_scale=1.0,
                       prior_mean_scale=1.0, use_diffusivity_prior=False):

    L, M, N = len(Q_eval), len(Q_obs), len(tau)
    print(f"L: {L}, M: {M}, N: {N}")

    # --- Compute normalized spatial and temporal kernels ---
    spatial_kernel = np.exp(-((Q_obs[:, None] - Q_eval[None, :]) ** 2) / (2 * mu_ ** 2))
    spatial_kernel /= spatial_kernel.sum(axis=1, keepdims=True)

    temporal_kernel = np.exp(-np.outer(t_obs, 1 / tau))
    G = (spatial_kernel[:, :, None] * temporal_kernel[:, None, :]).reshape(M, L * N)

    # --- Base spatial kernel for prior covariance ---
    Q_dist = (Q_eval[:, None] - Q_eval[None, :]) ** 2
    spatial_prior_kernel = np.exp(-Q_dist / (2 * lambda_ ** 2))  # (L, L)
    K_Q = spatial_prior_kernel
    K_base = np.kron(np.eye(N), K_Q)  # (LN x LN)

    # --- Optional: Estimate diffusivity D and apply exponential weights ---
    if use_diffusivity_prior:
        def single_exp(t, A, Gamma):
            return A * np.exp(-Gamma * t)

        D_list = []
        for q in np.unique(Q_obs):
            mask = Q_obs == q
            t_group = t_obs[mask]
            S_group = S_exp[mask]

            if len(t_group) < 3:
                continue

            # Sort by time
            sorted_idx = np.argsort(t_group)
            t_sorted = t_group[sorted_idx]
            S_sorted = S_group[sorted_idx]

            S0 = S_sorted[0]
            target = S0 / np.e

            # Find the first time where S < S0 / e
            below_threshold = np.where(S_sorted <= target)[0]
            if len(below_threshold) == 0:
                continue

            idx = below_threshold[0]
            t_star = t_sorted[idx]
            if t_star > 0:
                D_q = 1.0 / (q**2 * t_star)
                D_list.append(D_q)

        D_est = np.median(D_list) if D_list else 1.0
        print(f"Estimated D: {D_est:.4g}")

        # Apply weighting to prior covariance
        Q_mesh, tau_mesh = np.meshgrid(Q_eval, tau, indexing='ij')  # (L, N)
        weight_matrix = np.exp(-D_est * Q_mesh**2 * tau_mesh)
        weight_flat = weight_matrix.flatten()
        K_prior = (weight_flat[:, None] * K_base) * weight_flat[None, :]
    else:
        K_prior = K_base

    # --- Prior mean ---
    if not bg_mode:
        mu_prior_flat = np.zeros(L * N)
    else:
        mu_prior_flat = (np.ones((L, N)) / N * prior_mean_scale).flatten()

    # --- Posterior computation ---
    Sigma = np.diag((delta_S_exp * sigma_scale) ** 2)
    GK = G @ K_prior
    K_tilde = GK @ G.T + Sigma
    cho_K_tilde = cho_factor(K_tilde + 1e-8 * np.eye(M))

    residual = S_exp - G @ mu_prior_flat
    A_GPR_flat = mu_prior_flat + K_prior @ G.T @ cho_solve(cho_K_tilde, residual)
    A_GPR = A_GPR_flat.reshape(L, N)

    K_GPR_flat = K_prior - K_prior @ G.T @ cho_solve(cho_K_tilde, G @ K_prior)
    K_GPR = K_GPR_flat.reshape(L, N, L, N)

    # --- Reconstruct S(Q, t) ---
    spatial_kernel_eval = np.exp(-((Q_eval[:, None] - Q_eval[None, :]) ** 2) / (2 * lambda_ ** 2))
    spatial_kernel_eval /= spatial_kernel_eval.sum(axis=1, keepdims=True)
    temporal_kernel_eval = np.exp(-np.outer(t_eval, 1 / tau))
    S_reconstructed = np.einsum('ij,jn,in->i', spatial_kernel_eval, A_GPR, temporal_kernel_eval)

    return A_GPR, K_GPR, S_reconstructed

def _init_prior_mean_from_data(
    S_exp, delta_S_exp, G, K_prior=None,
    mode="wls_ridge", ridge=1e-2, project_nonneg=False, eps=1e-12
):
    """
    Returns mu_prior_flat estimated from S_exp and basis G.

    mode:
      - "wls_ridge": (G^T Σ^{-1} G + ρ I)^{-1} G^T Σ^{-1} S
      - "wls_kridge": (G^T Σ^{-1} G + ρ K_prior^{-1})^{-1} G^T Σ^{-1} S
    """
    M, D = G.shape
    Sigma_inv = np.diag(1.0 / np.maximum(delta_S_exp, eps)**2)

    GT_Sinv = G.T @ Sigma_inv
    normal = GT_Sinv @ G  # D x D
    rhs    = GT_Sinv @ S_exp  # D

    if mode == "wls_kridge":
        # Use K_prior^{-1} as Tikhonov metric if provided, else fall back to I
        if K_prior is not None:
            # Stable inverse via solve with small jitter
            # Solve K_prior * X = I  -> X = K_prior^{-1}
            I_D = np.eye(D)
            try:
                K_inv = np.linalg.solve(K_prior + eps*np.eye(D), I_D)
            except np.linalg.LinAlgError:
                K_inv = np.linalg.pinv(K_prior + eps*np.eye(D))
            reg = ridge * K_inv
        else:
            reg = ridge * np.eye(D)
    else:  # "wls_ridge"
        reg = ridge * np.eye(D)

    # Solve (normal + reg) * A0 = rhs
    try:
        cho = cho_factor(normal + reg + eps*np.eye(D))
        A0 = cho_solve(cho, rhs)
    except np.linalg.LinAlgError:
        A0 = np.linalg.lstsq(normal + reg + eps*np.eye(D), rhs, rcond=None)[0]

    if project_nonneg:
        A0 = np.maximum(A0, 0.0)

    return A0  # flat length L*N


def bayesian_inference_scale(
    S_exp, delta_S_exp, Q_obs, t_obs, Q_eval, t_eval, tau,
    mu_, lambda_, bg_mode=False, sigma_scale=1.0,
    prior_mean_scale=1.0, use_diffusivity_prior=False,
    match_observation_uncertainty=True, match_stat="mean", eps=1e-12
):
    """
    match_observation_uncertainty:
        If True, rescales the prior covariance K_prior by alpha so that
        diag(G K_prior G^T) on average matches (delta_S_exp * sigma_scale)^2.
    match_stat:
        "mean" or "median" to choose how alpha is computed.
    """

    L, M, N = len(Q_eval), len(Q_obs), len(tau)
    print(f"L: {L}, M: {M}, N: {N}")

    # --- Compute normalized spatial and temporal kernels (forward operator G) ---
    spatial_kernel = np.exp(-((Q_obs[:, None] - Q_eval[None, :]) ** 2) / (2 * mu_ ** 2))
    spatial_kernel /= np.clip(spatial_kernel.sum(axis=1, keepdims=True), eps, None)

    temporal_kernel = np.exp(-np.outer(t_obs, 1.0 / tau))
    G = (spatial_kernel[:, :, None] * temporal_kernel[:, None, :]).reshape(M, L * N)

    # --- Base spatial kernel for prior covariance ---
    Q_dist = (Q_eval[:, None] - Q_eval[None, :]) ** 2
    spatial_prior_kernel = np.exp(-Q_dist / (2 * lambda_ ** 2))  # (L, L)
    K_Q = spatial_prior_kernel
    K_base = np.kron(np.eye(N), K_Q)  # (LN x LN)

    # --- Optional: Estimate diffusivity D and apply exponential weights ---
    if use_diffusivity_prior:
        D_list = []
        for q in np.unique(Q_obs):
            mask = (Q_obs == q)
            t_group = t_obs[mask]
            S_group = S_exp[mask]
            if len(t_group) < 3:
                continue

            # Sort by time
            idx = np.argsort(t_group)
            t_sorted = t_group[idx]
            S_sorted = S_group[idx]

            S0 = S_sorted[0]
            target = S0 / np.e
            below = np.where(S_sorted <= target)[0]
            if len(below) == 0:
                continue
            t_star = t_sorted[below[0]]
            if t_star > 0:
                D_list.append(1.0 / (q**2 * t_star))

        D_est = np.median(D_list) if D_list else 1.0
        print(f"Estimated D: {D_est:.4g}")

        Q_mesh, tau_mesh = np.meshgrid(Q_eval, tau, indexing='ij')  # (L, N)
        weight = np.exp(-D_est * (Q_mesh**2) * tau_mesh).flatten()  # (L*N,)
        K_prior = (weight[:, None] * K_base) * weight[None, :]
    else:
        K_prior = K_base

    # --- Scale prior so that forward uncertainty matches observed uncertainty level ---
    if match_observation_uncertainty:
        # Project prior covariance to observation space: diag(G K_prior G^T)
        GK = G @ K_prior                         # (M x LN)
        prior_obs_var = np.sum(GK * G, axis=1)   # diagonal via row-wise dot with G
        prior_obs_var = np.maximum(prior_obs_var, 0.0)  # numerical safety

        target_var = (delta_S_exp * sigma_scale)**2

        if match_stat.lower() == "median":
            num = np.median(target_var)
            den = max(np.median(prior_obs_var), eps)
        else:  # "mean" (default)
            num = float(np.mean(target_var))
            den = float(max(np.mean(prior_obs_var), eps))

        alpha = num / den
        # Optional: clip to avoid extreme scaling (tune if desired)
        # alpha = np.clip(alpha, 1e-4, 1e4)
        print(f"Prior scaling alpha: {alpha:.4g} (match_stat={match_stat})")
        K_prior = alpha * K_prior
    else:
        alpha = 1.0

    # --- Prior mean ---
    if bg_mode:
        # keep your background prior if desired
        mu_prior_flat = (np.ones((L, N)) / N * prior_mean_scale).flatten()
    else:
        # Estimate prior mean from data and basis
        mu_prior_flat = _init_prior_mean_from_data(
            S_exp=S_exp,
            delta_S_exp=delta_S_exp * sigma_scale,
            G=G,
            K_prior=K_prior,          # used if mode="wls_kridge"
            mode="wls_ridge",         # or "wls_kridge"
            ridge=1e-2,               # tune 1e-4 ~ 1e-1
            project_nonneg=False,     # set True if you want A >= 0
            eps=1e-12
        )

    # --- Posterior computation ---
    Sigma = np.diag((delta_S_exp * sigma_scale) ** 2)
    GK = G @ K_prior
    K_tilde = GK @ G.T + Sigma
    cho_K_tilde = cho_factor(K_tilde + 1e-8 * np.eye(M))

    residual = S_exp - G @ mu_prior_flat
    A_GPR_flat = mu_prior_flat + K_prior @ G.T @ cho_solve(cho_K_tilde, residual)
    A_GPR = A_GPR_flat.reshape(L, N)

    K_GPR_flat = K_prior - K_prior @ G.T @ cho_solve(cho_K_tilde, G @ K_prior)
    K_GPR = K_GPR_flat.reshape(L, N, L, N)

    # --- Reconstruct S(Q, t) at evaluation grid ---
    spatial_kernel_eval = np.exp(-((Q_eval[:, None] - Q_eval[None, :]) ** 2) / (2 * lambda_ ** 2))
    spatial_kernel_eval /= np.clip(spatial_kernel_eval.sum(axis=1, keepdims=True), eps, None)

    temporal_kernel_eval = np.exp(-np.outer(t_eval, 1.0 / tau))  # (T_eval x N)
    # S_reconstructed[i_t] = sum_j A_GPR[j_n]*temporal_kernel_eval[i_t,n] * spatial_kernel_eval[j_q,j_q’]
    # Equivalent compact einsum:
    S_reconstructed = np.einsum('ij,jn,in->i', spatial_kernel_eval, A_GPR, temporal_kernel_eval)

    return A_GPR, K_GPR, S_reconstructed, alpha

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


def reconstruct_scattering_function(Q_eval, t_eval, A_GPR, K_GPR, tau, lambda_, bg_mode=False):
    """
    Reconstruct the scattering function and its uncertainty from posterior mean and covariance.

    Returns:
        S_reconstructed: 1D array of size L*K (flattened over Q_eval and t_eval).
    """
    L = len(Q_eval)
    N = len(tau)
    K = len(t_eval)

    # Reconstruct scattering function at evaluation points
    spatial_kernel_eval = np.exp(-((Q_eval[:, None] - Q_eval[None, :]) ** 2) / (2 * lambda_ ** 2)) # Shape (L, L)
    spatial_kernel_eval /= spatial_kernel_eval.sum(axis=1, keepdims=True) # Shape (L, L)
    temporal_kernel_eval = np.exp(-np.outer(t_eval, 1 / tau)) # Shape (K, N)
    S_reconstructed = np.einsum('ij,jn,kn->ik', spatial_kernel_eval, A_GPR, temporal_kernel_eval)

    return S_reconstructed


