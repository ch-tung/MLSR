"""USANS allocation benchmark library."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.linalg import solve


EPS = np.finfo(float).eps

plt.rcParams.update(
    {
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    }
)


def ensure_1d(x):
    """Return x as a finite one-dimensional float array."""
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("array must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("array contains non-finite values")
    return arr


def gauss_legendre_ab(a, b, n):
    """Return n-point Gauss-Legendre nodes and weights on [a, b]."""
    nodes, weights = np.polynomial.legendre.leggauss(n)
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    return mid + half * nodes, half * weights


def gaussian_smoother(Q, mu_Q):
    """Return a row-normalized Gaussian smoother in linear Q, or None if disabled."""
    Q = ensure_1d(Q)
    if mu_Q is None or mu_Q <= 0:
        return None
    dQ = Q[:, None] - Q[None, :]
    S = np.exp(-0.5 * (dQ / mu_Q) ** 2)
    row_sum = np.maximum(S.sum(axis=1, keepdims=True), EPS)
    return S / row_sum


def trapz_weights(Q):
    """Return trapezoidal integration weights for a one-dimensional grid."""
    Q = ensure_1d(Q)
    if Q.size == 1:
        return np.ones(1)
    weights = np.empty_like(Q)
    weights[0] = 0.5 * (Q[1] - Q[0])
    weights[-1] = 0.5 * (Q[-1] - Q[-2])
    weights[1:-1] = 0.5 * (Q[2:] - Q[:-2])
    return np.maximum(weights, EPS)


def renormalize_rows(R_raw, Q_model):
    """Normalize a raw kernel so each row integrates to one over Q_model."""
    R_raw = np.asarray(R_raw, dtype=float)
    weights = trapz_weights(Q_model)
    row_int = R_raw @ weights
    row_int = np.maximum(row_int, EPS)
    return R_raw / row_int[:, None]


def R_rect(
    Qx_obs,
    Q_mod,
    sigma_x,
    sigma_y,
    *,
    nx=64,
    mu_Q=0.0,
    cell_average=True,
    quad="gauss",
):
    r"""
    Rectangular pinhole PSF via horizontal averaging of slit PSFs:

        R_rect(Qx, Q') = (1/(2 sigma_x)) int_{Qx-sigma_x}^{Qx+sigma_x}
        R_slit(Qx', Q'; sigma_y) dQx'.

    Returns R such that I_exp approximately equals R @ I_mod.
    """
    Qx = ensure_1d(Qx_obs)
    Qp = ensure_1d(Q_mod)

    if sigma_x <= 0 or sigma_y <= 0:
        raise ValueError("sigma_x and sigma_y must be positive")
    if nx < 2:
        raise ValueError("nx must be >= 2")

    quad = quad.lower()
    if quad not in {"gauss", "trapz"}:
        raise ValueError("quad must be 'gauss' or 'trapz'")

    M_obs, M_mod = Qx.size, Qp.size
    R = np.zeros((M_obs, M_mod))

    if cell_average:
        edges = np.empty(M_mod + 1)
        edges[1:-1] = 0.5 * (Qp[1:] + Qp[:-1])
        edges[0] = Qp[0] - 0.5 * (Qp[1] - Qp[0])
        edges[-1] = Qp[-1] + 0.5 * (Qp[-1] - Qp[-2])
        dQ = np.diff(edges)

    for j, qx in enumerate(Qx):
        a, b = qx - sigma_x, qx + sigma_x

        if quad == "gauss":
            x_nodes, w_nodes = gauss_legendre_ab(a, b, nx)
            accum = np.zeros(M_mod)

            for xprime, w in zip(x_nodes, w_nodes):
                a_qp = xprime
                b_qp = np.sqrt(xprime * xprime + sigma_y * sigma_y)

                if cell_average:
                    Lk = np.maximum(edges[:-1], a_qp)
                    Uk = np.minimum(edges[1:], b_qp)
                    valid = Uk > Lk

                    r_line = np.zeros_like(Qp)
                    if np.any(valid):
                        with np.errstate(invalid="ignore"):
                            termU = np.sqrt(np.maximum(Uk[valid]**2 - a_qp * a_qp, 0.0))
                            termL = np.sqrt(np.maximum(Lk[valid]**2 - a_qp * a_qp, 0.0))
                            r_line[valid] = (1.0 / sigma_y) * (termU - termL) / dQ[valid]
                else:
                    qp = Qp
                    mask = (qp > a_qp) & (qp <= b_qp)
                    denom = np.sqrt(np.maximum(qp**2 - a_qp * a_qp, 0.0))
                    r_line = np.zeros_like(qp)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        r_line[mask] = (1.0 / sigma_y) * (qp[mask] / denom[mask])

                accum += w * r_line

            R[j, :] = (1.0 / (2.0 * sigma_x)) * accum

        else:
            x_nodes = np.linspace(a, b, nx)
            R_lines = np.zeros((nx, M_mod))

            for i, xprime in enumerate(x_nodes):
                a_qp = xprime
                b_qp = np.sqrt(xprime * xprime + sigma_y * sigma_y)

                if cell_average:
                    Lk = np.maximum(edges[:-1], a_qp)
                    Uk = np.minimum(edges[1:], b_qp)
                    valid = Uk > Lk

                    r_line = np.zeros_like(Qp)
                    if np.any(valid):
                        with np.errstate(invalid="ignore"):
                            termU = np.sqrt(np.maximum(Uk[valid]**2 - a_qp * a_qp, 0.0))
                            termL = np.sqrt(np.maximum(Lk[valid]**2 - a_qp * a_qp, 0.0))
                            r_line[valid] = (1.0 / sigma_y) * (termU - termL) / dQ[valid]
                else:
                    qp = Qp
                    mask = (qp > a_qp) & (qp <= b_qp)
                    denom = np.sqrt(np.maximum(qp**2 - a_qp * a_qp, 0.0))
                    r_line = np.zeros_like(qp)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        r_line[mask] = (1.0 / sigma_y) * (qp[mask] / denom[mask])

                R_lines[i, :] = r_line

            accum = np.trapezoid(R_lines, x_nodes, axis=0)
            R[j, :] = (1.0 / (2.0 * sigma_x)) * accum

    S = gaussian_smoother(Qp, mu_Q)
    if S is not None:
        R = R @ S.T

    R = renormalize_rows(R, Qp)
    return R


def psf(sigma_y, Q, Q_prime, sigma_x=None):
    """Return a row-stochastic Bonse-Hart rectangular-slit PSF matrix."""
    if sigma_x is None:
        sigma_x = sigma_y * 1e-7

    R_raw = R_rect(
        Q,
        Q_prime,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        nx=4,
        quad="gauss",
        cell_average=True,
    )

    w_model = trapz_weights(Q_prime)
    num = R_raw * w_model[None, :]
    row_sum = num.sum(axis=1, keepdims=True)
    row_sum = np.maximum(row_sum, EPS)
    R_mat = num / row_sum
    return R_mat


def make_q_grid(Q_min, Q_max, N):
    """Return a linearly spaced Q grid."""
    return np.linspace(Q_min, Q_max, N)


def make_resolution_model_grid(Q_obs, Q_model_max, N_tail):
    """Return a model Q grid that contains Q_obs and extends to higher Q."""
    Q_obs = ensure_1d(Q_obs)
    if Q_model_max <= Q_obs[-1]:
        return Q_obs.copy()
    if N_tail <= 0:
        raise ValueError("N_tail must be positive when extending the model grid")

    h_obs = Q_obs[1] - Q_obs[0]
    Q_tail_min = Q_obs[-1] + h_obs
    Q_tail = np.linspace(Q_tail_min, Q_model_max, N_tail)
    return np.concatenate([Q_obs, Q_tail])


def make_ground_truth(Q, I0, Qc, m, background):
    """Return the smooth ground-truth scattering intensity."""
    Q = ensure_1d(Q)
    return I0 / (1.0 + (Q / Qc) ** m) + background


def make_lorentzian_ground_truth(Q, I0, xi, background):
    """Return a Lorentzian scattering curve I(Q) = I0 / (1 + (Q xi)**2) + background."""
    Q = ensure_1d(Q)
    return I0 / (1.0 + (Q * xi) ** 2) + background


def make_sinc_ground_truth(Q, I0, Q_min, Q_max, n_oscillations, background):
    """Return a positive oscillatory curve proportional to 1 + sin(x) / x."""
    Q = ensure_1d(Q)
    x = 2.0 * np.pi * n_oscillations * (Q - Q_min) / (Q_max - Q_min)
    x = np.maximum(x, 1.0e-9)
    return I0 * (1.0 + np.sin(x) / x) + background


def make_log_gaussian_correlation_operator(Q, ell_log):
    """Return a row-normalized Gaussian operator on log(Q)."""
    Q = ensure_1d(Q)
    logQ = np.log(Q)
    dlog = logQ[:, None] - logQ[None, :]
    K = np.exp(-0.5 * (dlog / ell_log) ** 2)
    row_sum = np.maximum(K.sum(axis=1, keepdims=True), EPS)
    return K / row_sum


def make_log_gaussian_covariance(Q, ell_log, variance=1.0, jitter=1.0e-6):
    """Return a positive-definite squared-exponential covariance on log(Q)."""
    Q = ensure_1d(Q)
    logQ = np.log(Q)
    dlog = logQ[:, None] - logQ[None, :]
    K = variance * np.exp(-0.5 * (dlog / ell_log) ** 2)
    K += jitter * np.eye(Q.size)
    return K


def make_gaussian_correlation_operator(Q, ell):
    """Return a row-normalized Gaussian operator on linear Q."""
    Q = ensure_1d(Q)
    dQ = Q[:, None] - Q[None, :]
    K = np.exp(-0.5 * (dQ / ell) ** 2)
    row_sum = np.maximum(K.sum(axis=1, keepdims=True), EPS)
    return K / row_sum


def make_gaussian_covariance(Q, ell, variance=1.0, jitter=1.0e-6):
    """Return a positive-definite squared-exponential covariance on linear Q."""
    Q = ensure_1d(Q)
    dQ = Q[:, None] - Q[None, :]
    K = variance * np.exp(-0.5 * (dQ / ell) ** 2)
    K += jitter * np.eye(Q.size)
    return K


def _matern32_kernel(r):
    """Return the Matérn 3/2 kernel evaluated at nonnegative r."""
    r = np.asarray(r, dtype=float)
    s3 = np.sqrt(3.0)
    return (1.0 + s3 * r) * np.exp(-s3 * r)


def _matern52_kernel(r):
    """Return the Matérn 5/2 kernel evaluated at nonnegative r."""
    r = np.asarray(r, dtype=float)
    s5 = np.sqrt(5.0)
    return (1.0 + s5 * r + 5.0 * r * r / 3.0) * np.exp(-s5 * r)


def _matern_kernel(r, nu=1.5):
    """Return a Matérn kernel for common half-integer smoothness values."""
    if nu == 0.5:
        return np.exp(-r)
    if nu == 1.5:
        return _matern32_kernel(r)
    if nu == 2.5:
        return _matern52_kernel(r)
    raise ValueError("supported Matérn smoothness values are 0.5, 1.5, and 2.5")


def make_matern_correlation_operator(Q, ell, nu=1.5):
    """Return a row-normalized Matérn operator on linear Q."""
    Q = ensure_1d(Q)
    dQ = np.abs(Q[:, None] - Q[None, :]) / ell
    K = _matern_kernel(dQ, nu=nu)
    row_sum = np.maximum(K.sum(axis=1, keepdims=True), EPS)
    return K / row_sum


def make_matern_covariance(Q, ell, variance=1.0, jitter=1.0e-6, nu=1.5):
    """Return a positive-definite Matérn covariance on linear Q."""
    Q = ensure_1d(Q)
    dQ = np.abs(Q[:, None] - Q[None, :]) / ell
    K = variance * _matern_kernel(dQ, nu=nu)
    K += jitter * np.eye(Q.size)
    return K


def make_log_matern_correlation_operator(Q, ell_log, nu=1.5):
    """Return a row-normalized Matérn operator on log(Q)."""
    Q = ensure_1d(Q)
    logQ = np.log(Q)
    dlog = np.abs(logQ[:, None] - logQ[None, :]) / ell_log
    K = _matern_kernel(dlog, nu=nu)
    row_sum = np.maximum(K.sum(axis=1, keepdims=True), EPS)
    return K / row_sum


def make_log_matern_covariance(Q, ell_log, variance=1.0, jitter=1.0e-6, nu=1.5):
    """Return a positive-definite Matérn covariance on log(Q)."""
    Q = ensure_1d(Q)
    logQ = np.log(Q)
    dlog = np.abs(logQ[:, None] - logQ[None, :]) / ell_log
    K = variance * _matern_kernel(dlog, nu=nu)
    K += jitter * np.eye(Q.size)
    return K


def estimate_optimal_log_kernel_width(Q, I, n_counts):
    """Estimate an optimal Gaussian kernel width in log(Q) from curvature scaling."""
    Q = ensure_1d(Q)
    I = ensure_1d(I)
    if Q.size != I.size:
        raise ValueError("Q and I must have the same length")
    if n_counts <= 0:
        raise ValueError("n_counts must be positive")

    x = np.log(Q)
    order = np.argsort(x)
    x = x[order]
    y = I[order]

    L_log = x[-1] - x[0]
    if L_log <= 0:
        raise ValueError("log-Q domain length must be positive")

    C_I = np.trapezoid(y, x) / L_log
    dy_dx = np.gradient(y, x, edge_order=2)
    d2y_dx2 = np.gradient(dy_dx, x, edge_order=2)

    beta = np.trapezoid(dy_dx * dy_dx, x) / (12.0 * L_log)
    gamma = np.trapezoid(d2y_dx2 * d2y_dx2, x) / (4.0 * L_log)
    alpha = C_I * C_I * L_log / n_counts

    gamma = max(gamma, EPS)
    ell_log = (alpha / (8.0 * np.sqrt(np.pi) * gamma)) ** 0.2
    h_instr_log = float(np.median(np.diff(x)))

    return {
        "ell_log": float(ell_log),
        "h_instr_log": h_instr_log,
        "C_I": float(C_I),
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
    }


def make_second_difference_matrix(N):
    """Return the second-difference matrix with shape (N - 2, N)."""
    if N < 3:
        raise ValueError("N must be at least 3")
    L = np.zeros((N - 2, N))
    rows = np.arange(N - 2)
    L[rows, rows] = 1.0
    L[rows, rows + 1] = -2.0
    L[rows, rows + 2] = 1.0
    return L


def make_reconstruction_operator(R, L, alpha_reg, ridge=0.0):
    """Return Tikhonov-regularized reconstruction operator inv(R.T R + alpha L.T L + ridge I) R.T."""
    lhs = R.T @ R + alpha_reg * (L.T @ L) + ridge * np.eye(R.shape[1])
    rhs = R.T
    return solve(lhs, rhs, assume_a="sym")


def compute_a_coefficients(A, v):
    """Return a_i = v_i * sum_k A[k, i]**2."""
    A = np.asarray(A, dtype=float)
    v = ensure_1d(v)
    if A.shape[1] != v.size:
        raise ValueError("A columns must match v length")
    a = v * np.sum(A * A, axis=0)
    return np.maximum(a, EPS)


def optimal_allocation_from_a(a, T_tot):
    """Return closed-form optimal allocation under sum_i t_i = T_tot."""
    a = ensure_1d(a)
    weights = np.sqrt(np.maximum(a, EPS))
    return T_tot * weights / np.maximum(weights.sum(), EPS)


def uniform_allocation(N, T_tot):
    """Return uniform allocation across N points."""
    if N <= 0:
        raise ValueError("N must be positive")
    return np.full(N, T_tot / N)


def inverse_intensity_allocation(I, T_tot):
    """Return allocation proportional to inverse intensity."""
    I = ensure_1d(I)
    weights = 1.0 / np.maximum(I, EPS)
    return T_tot * weights / np.maximum(weights.sum(), EPS)


def inverse_intensity_power_allocation(I, power, T_tot):
    """Return allocation proportional to I**(-power)."""
    I = ensure_1d(I)
    weights = np.maximum(I, EPS) ** (-power)
    return T_tot * weights / np.maximum(weights.sum(), EPS)


def uncertainty_from_a(a, t):
    """Return scalar uncertainty E = sum_i a_i / t_i."""
    a = ensure_1d(a)
    t = ensure_1d(t)
    if a.size != t.size:
        raise ValueError("a and t must have the same length")
    return float(np.sum(a / np.maximum(t, EPS)))


def marginal_scores(a, t):
    """Return marginal scores g_i = a_i / t_i**2."""
    a = ensure_1d(a)
    t = ensure_1d(t)
    if a.size != t.size:
        raise ValueError("a and t must have the same length")
    return a / np.maximum(t, EPS) ** 2


def base_relative_error_and_scores(I, t):
    """Return averaged relative-counting error and its marginal scores."""
    I = ensure_1d(I)
    t = ensure_1d(t)
    N = I.size
    E = float(np.mean(1.0 / (np.maximum(t, EPS) * np.maximum(I, EPS))))
    g = 1.0 / (N * np.maximum(I, EPS) * np.maximum(t, EPS) ** 2)
    return E, g


def correlation_posterior_error_and_scores(I, K_inv, t):
    """Return posterior relative-error objective and scores for the correlation model."""
    I = ensure_1d(I)
    t = ensure_1d(t)
    N = I.size
    precision = K_inv + np.diag(t * I)
    sigma = solve(precision, np.eye(N), assume_a="sym")
    sigma_sq = sigma @ sigma
    E = float(np.trace(sigma) / N)
    g = I * np.diag(sigma_sq) / N
    return E, np.maximum(g, EPS)


def resolution_posterior_error_and_scores(R, I_meas, K_latent_inv, t, N_display):
    """Return displayed-latent posterior objective and scores for the resolution model."""
    I_meas = ensure_1d(I_meas)
    t = ensure_1d(t)
    M = K_latent_inv.shape[0]
    weights = t / np.maximum(I_meas, EPS)
    precision = K_latent_inv + R.T @ (weights[:, None] * R)
    sigma = solve(precision, np.eye(M), assume_a="sym")
    sigma_display = sigma[:, :N_display]
    row_projection = R @ sigma_display
    E = float(np.trace(sigma[:N_display, :N_display]) / N_display)
    g = np.sum(row_projection * row_projection, axis=1) / (
        np.maximum(I_meas, EPS) * N_display
    )
    return E, np.maximum(g, EPS)


def optimize_allocation_by_scores(score_func, t_initial, T_tot, p=0.5, max_iter=120, tol=1.0e-3):
    """Iteratively equalize marginal scores while preserving total time."""
    t = ensure_1d(t_initial)
    t = T_tot * np.maximum(t, EPS) / np.maximum(np.sum(t), EPS)
    for _ in range(max_iter):
        E, g = score_func(t)
        active = g > EPS
        ratio = np.max(g[active]) / np.min(g[active])
        if ratio - 1.0 < tol:
            return t, E, g
        t = t * np.maximum(g, EPS) ** p
        t = T_tot * t / np.maximum(np.sum(t), EPS)

    E, g = score_func(t)
    return t, E, g


def plot_allocation_profiles_with_inset(Q, I_true, I_smeared, allocations, output_path):
    """Save a one-panel allocation figure with an inset for I(Q)."""
    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)

    styles = {
        "Base case": ("black", "Base case"),
        "Correlation-aware": ("blue", "Correlation-aware"),
    }
    for name, t in allocations.items():
        color, label = styles[name]
        ax.plot(Q, t, color=color, lw=2.0, label=label)

    ax.set_xlabel(r"$Q_i$ ($\mathrm{\AA}^{-1}$)")
    ax.set_ylabel(r"$t_i$ (s)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_box_aspect(1)
    # ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="upper right")

    inset = inset_axes(
    ax,
    width="43%",
    height="43%",
    loc="lower left",
    borderpad=0.8,
    bbox_to_anchor=(0.1, 0.4, 1, 1),
    bbox_transform=ax.transAxes,
    )
    inset.loglog(Q, I_true, color="black", lw=1.4, label="I(Q)")
    inset.set_xlabel(r"$Q$ ($\mathrm{\AA}^{-1}$)", fontsize=10)
    inset.set_ylabel(r"$I(Q)$ ($\mathrm{cm}^{-1}$)", fontsize=10)
    inset.tick_params(labelsize=8)
    inset.grid(True, which="both", alpha=0.15)

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_figure1(Q, I_true, I_smeared, allocations, output_path):
    """Compatibility wrapper for the allocation-profile figure."""
    plot_allocation_profiles_with_inset(Q, I_true, I_smeared, allocations, output_path)


def plot_uncertainty_vs_total_time(summary_rows, output_path, case_order=None):
    """Save the three-panel uncertainty-versus-total-time figure."""
    if case_order is None:
        case_order = ["Base case", "Bonse-Hart silt smeared"]
    strategy_order = ["Uniform", "Inverse intensity", "Optimal"]
    colors = {"Uniform": "0.45", "Inverse intensity": "tab:green", "Optimal": "red"}
    linestyles = {"Uniform": "--", "Inverse intensity": "-.", "Optimal": "-"}

    fig, axes = plt.subplots(
        1,
        len(case_order),
        figsize=(4.5 * len(case_order), 4.5),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for ax, case in zip(axes, case_order):
        for strategy in strategy_order:
            sub = [
                row
                for row in summary_rows
                if row["case"] == case and row["strategy"] == strategy
            ]
            ax.loglog(
                [row["T_tot"] for row in sub],
                [row["E"] for row in sub],
                color=colors[strategy],
                ls=linestyles[strategy],
                lw=2.0,
                label=strategy,
            )
        ax.set_xlabel(r"Total measurement time $T_{tot}$ (s)")
        ax.set_ylabel(r"$\mathcal{E}$")
        ax.set_box_aspect(1)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False)

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_figure2(summary_rows, output_path, case_order=None):
    """Compatibility wrapper for the uncertainty-versus-time figure."""
    plot_uncertainty_vs_total_time(summary_rows, output_path, case_order=case_order)


def plot_marginal_score_profiles(Q, scores, output_path):
    """Save marginal equal-gradient scores versus Q for each uncertainty model."""
    styles = {
        "Base case": ("black", "Base case"),
        "Correlation-aware": ("blue", "Correlation-aware"),
        "Bonse-Hart silt smeared": ("red", "Bonse-Hart silt smeared"),
    }

    fig, ax = plt.subplots(figsize=(5.4, 5.4), constrained_layout=True)
    for name, g in scores.items():
        color, label = styles[name]
        ax.plot(Q, g, color=color, lw=2.0, label=label)

    ax.set_xscale("log")
    ax.set_xlabel(r"$Q$ ($\mathrm{\AA}^{-1}$)")
    ax.set_ylabel(r"Marginal score $g_i = a_i / t_i^2$")
    ax.set_box_aspect(1)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_marginal_scores(Q, scores, output_path):
    """Compatibility wrapper for marginal-score profiles."""
    plot_marginal_score_profiles(Q, scores, output_path)


def plot_inverse_power_scan(power_rows, optimal_rows, output_path, case_order=None):
    """Save uncertainty versus inverse-intensity allocation power."""
    if case_order is None:
        case_order = ["Base case", "Bonse-Hart silt smeared"]
    colors = {
        "Base case": "black",
        "Correlation-aware": "blue",
        "Bonse-Hart silt smeared": "red",
    }

    fig, ax = plt.subplots(figsize=(5.4, 5.4), constrained_layout=True)
    for case in case_order:
        sub = [row for row in power_rows if row["case"] == case]
        ax.plot(
            [row["power_n"] for row in sub],
            [row["E"] for row in sub],
            color=colors[case],
            lw=2.0,
            label=case,
        )

        opt = next(row for row in optimal_rows if row["case"] == case)
        ax.axhline(
            opt["E_opt"],
            color=colors[case],
            lw=1.5,
            ls="--",
            alpha=0.8,
        )

    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$\mathcal{E}$")
    ax.set_yscale("log")
    ax.set_box_aspect(1)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_power_scan(power_rows, optimal_rows, output_path, case_order=None):
    """Compatibility wrapper for the inverse-power scan figure."""
    plot_inverse_power_scan(power_rows, optimal_rows, output_path, case_order=case_order)


def plot_errorbar_bands(Q, I_gt, cases, output_path, T_tot_by_case, seed=12345, case_order=None):
    """Save a base-case two-panel band comparison figure."""
    I_gt = ensure_1d(I_gt)
    if case_order is None:
        case_order = ["Base case"]
    case_labels = {
        "Base case": "Base",
        "Correlation-aware": "Correlation-aware",
        "Bonse-Hart silt smeared": "Bonse-Hart silt smeared",
    }
    strategy_specs = [
        ("n=0", "0.0", "black"),
        ("n=1", "1.0", "blue"),
        ("Optimal", "optimal", "red"),
    ]
    shift_factors = [1.0, 8.0, 64.0]
    rng = np.random.default_rng(seed)

    if len(case_order) != 1:
        raise ValueError("This figure is configured for a single visible case.")

    case_name = case_order[0]
    case = cases[case_name]
    I_ref = ensure_1d(case["I_ref"])
    T_tot = T_tot_by_case[case_name]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.5), constrained_layout=True)
    ax_abs, ax_rel = axes

    relative_lows = []
    relative_highs = []
    for (label, power, color), shift in zip(strategy_specs, shift_factors):
        if power == "optimal":
            t = case["t_opt_figure3"]
        else:
            t = inverse_intensity_power_allocation(I_ref, float(power), T_tot)

        mean_counts = np.maximum(I_ref * t, EPS)
        sampled_counts = rng.poisson(mean_counts)
        sampled_intensity = sampled_counts / np.maximum(t, EPS)
        sigma_intensity = np.sqrt(np.maximum(I_ref, EPS) / np.maximum(t, EPS))

        y_true = I_ref * shift
        y_low = np.maximum((I_ref - sigma_intensity) * shift, EPS)
        y_high = (I_ref + sigma_intensity) * shift
        y_sample = np.maximum(sampled_intensity * shift, EPS)

        ax_abs.fill_between(Q, y_low, y_high, color=color, alpha=0.18, linewidth=0)
        ax_abs.plot(Q, y_true, color="black", lw=1.0, alpha=0.75)
        ax_abs.scatter(Q, y_sample, s=8, color=color, alpha=0.75, edgecolors="none", label=label)
        ax_abs.text(Q[0] * 1.07, y_true[0] * 1.12, label, color=color, fontsize=8, va="bottom")

        rel_true = I_ref / np.maximum(I_gt, EPS)
        rel_low = np.maximum(I_ref - sigma_intensity, EPS) / np.maximum(I_gt, EPS)
        rel_high = (I_ref + sigma_intensity) / np.maximum(I_gt, EPS)
        relative_lows.append(np.min(rel_low))
        relative_highs.append(np.max(rel_high))
        ax_rel.fill_between(Q, rel_low, rel_high, color=color, alpha=0.18, linewidth=0)
        ax_rel.plot(Q, rel_true, color="black", lw=1.0, alpha=0.75)

    ax_abs.set_ylim(np.min(I_ref) * shift_factors[0] / 10.0, np.max(I_ref) * shift_factors[-1] * 10.0)
    ax_abs.set_xscale("log")
    ax_abs.set_yscale("log")
    ax_abs.set_xlabel(r"$Q$ ($\mathrm{\AA}^{-1}$)")
    ax_abs.set_ylabel(r"$I$")
    ax_abs.set_box_aspect(1)
    ax_abs.grid(True, which="both", alpha=0.25)
    ax_abs.text(0.04, 0.96, r"$I(Q)$", transform=ax_abs.transAxes, ha="left", va="top", fontsize=9)

    rel_min = min(relative_lows)
    rel_max = max(relative_highs)
    rel_pad = 0.08 * (rel_max - rel_min)
    ax_rel.axhline(1.0, color="0.35", lw=1.0, ls="--", alpha=0.8)
    ax_rel.set_ylim(max(0.0, rel_min - rel_pad), rel_max + rel_pad)
    ax_rel.set_xscale("log")
    ax_rel.set_xlabel(r"$Q$ ($\mathrm{\AA}^{-1}$)")
    ax_rel.set_ylabel(r"$I/I_{\mathrm{gt}}$")
    ax_rel.set_box_aspect(1)
    ax_rel.grid(True, which="both", alpha=0.25)
    ax_rel.text(0.04, 0.96, r"$I(Q)$", transform=ax_rel.transAxes, ha="left", va="top", fontsize=9)

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def print_total_count_diagnostics(cases, T_tot_by_case, case_order=None):
    """Print expected total counts for Figure 3 allocation choices."""
    strategy_specs = [
        ("n=0", "0.0"),
        ("n=1", "1.0"),
        ("Optimal", "optimal"),
    ]
    if case_order is None:
        case_order = ["Base case", "Bonse-Hart silt smeared"]

    print("Expected total counts for Figure 3 allocations")
    for case_name in case_order:
        case = cases[case_name]
        I_ref = ensure_1d(case["I_ref"])
        T_tot = T_tot_by_case[case_name]
        print(case_name)
        print(f"  T_tot: {T_tot:.16e}")
        for label, power in strategy_specs:
            if power == "optimal":
                t = case["t_opt_figure3"]
            else:
                t = inverse_intensity_power_allocation(I_ref, float(power), T_tot)
            total_counts = float(np.sum(I_ref * t))
            print(f"  {label}: {total_counts:.16e}")
