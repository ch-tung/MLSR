"""Minimal reproducible example for the MLSR binning-width estimates.

The MLSR binning notebooks estimate two characteristic scales from a 1D
scattering curve I(Q):

* h_FD: Freedman-Diaconis optimal bin width
* lambda_opt: optimal equivalent Gaussian/RBF smoothing width
* lambda_ab: alpha-beta kernel-length heuristic

Both are computed from the signal roughness and a counting-noise scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from scipy.signal import savgol_filter
except Exception:  # pragma: no cover - exercised when scipy is absent.
    savgol_filter = None


@dataclass(frozen=True)
class BinningResult:
    """Container for the estimated binning and kernel scales."""

    h_fd: float
    lambda_opt: float
    lambda_ab: float
    alpha: float
    beta: float
    gamma: float
    chi: float
    data_bin_width: float
    length: float
    mean_intensity: float
    a0: float
    window_length: Optional[int]
    polyorder: int
    used_savgol: bool


@dataclass(frozen=True)
class KernelCorrectionResult:
    """Mapping from an equivalent smoothing width to a GP covariance length."""

    smoothing_width: float
    covariance_length: float
    target_noise_variance: float
    achieved_noise_variance: float
    signal_variance: float


def synthetic_scattering_data(
    n_points: int = 401,
    total_counts: float = 2.0e6,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a small synthetic I(Q) curve with Poisson-like noise.

    Returns
    -------
    q, observed_i, observed_i_err, true_i
    """

    rng = np.random.default_rng(seed)
    q = np.linspace(0.004, 0.15, n_points)
    true_i = (
        3.0 / (1.0 + (q / 0.035) ** 4)
        + 0.28 * np.exp(-0.5 * ((q - 0.105) / 0.012) ** 2)
        + 0.06
    )

    weights = true_i / np.sum(true_i)
    expected_counts = total_counts * weights
    counts = rng.poisson(expected_counts)
    counts = np.maximum(counts, 1)

    scale = np.sum(true_i) / total_counts
    observed_i = counts * scale
    observed_i_err = np.sqrt(counts) * scale
    return q, observed_i, observed_i_err, true_i


def estimate_binning_scales(
    q: np.ndarray,
    intensity: np.ndarray,
    *,
    total_counts: Optional[float] = None,
    intensity_error: Optional[np.ndarray] = None,
    window_frac: float = 0.15,
    polyorder: int = 3,
    use_savgol: bool = True,
    warn_on_coarse_data: bool = True,
) -> BinningResult:
    """Estimate the MLSR Freedman-Diaconis bin width and kernel size.

    Parameters
    ----------
    q, intensity:
        1D arrays describing I(Q). Q values are sorted internally.
    total_counts:
        Total number of counts in the measurement. If not supplied, the noise
        scale is estimated from ``intensity_error``.
    intensity_error:
        1-sigma uncertainty for each intensity point. Used only when
        ``total_counts`` is omitted. In that mode, uncertainties are assumed
        to describe observations at the current data spacing.
    window_frac, polyorder:
        Savitzky-Golay settings for derivative estimates. The original
        notebooks commonly use a 15 percent window and cubic polynomial.
    use_savgol:
        Use Savitzky-Golay derivative estimates when SciPy is available.
    warn_on_coarse_data:
        Print a warning if the median input Q spacing is not smaller than the
        estimated Freedman-Diaconis optimal bin width.

    Notes
    -----
    Following the MLSR notebooks:

    beta = integral (I'(Q)^2 dQ) / (12 L)
    gamma = integral (I''(Q)^2 dQ) / (4 L)
    A0 = L * mean(I)^2
    alpha = A0 / total_counts
    h_FD = (alpha / (2 beta))^(1/3)
    lambda_opt = (alpha / (8 sqrt(pi) gamma))^(1/5)
    lambda_ab = h_FD
    """

    q, intensity = _clean_and_sort(q, intensity)
    if len(q) < 5:
        raise ValueError("At least 5 points are required to estimate derivatives.")

    length = float(q[-1] - q[0])
    if length <= 0:
        raise ValueError("Q range must be positive.")
    data_bin_width = float(np.median(np.diff(q)))

    mean_intensity = float(_trapezoid(intensity, q) / length)
    a0 = length * mean_intensity**2

    window_length = None
    if use_savgol and savgol_filter is not None and len(q) >= polyorder + 3:
        window_length = _choose_savgol_window(len(q), window_frac, polyorder)
        delta = float(np.median(np.diff(q)))
        first_derivative = savgol_filter(
            intensity,
            window_length=window_length,
            polyorder=polyorder,
            deriv=1,
            delta=delta,
            mode="interp",
        )
        second_derivative = savgol_filter(
            intensity,
            window_length=window_length,
            polyorder=polyorder,
            deriv=2,
            delta=delta,
            mode="interp",
        )
        used_savgol = True
    else:
        first_derivative = np.gradient(intensity, q, edge_order=2)
        second_derivative = np.gradient(first_derivative, q, edge_order=2)
        used_savgol = False

    beta = float(_trapezoid(first_derivative**2, q) / (12.0 * length))
    gamma = float(_trapezoid(second_derivative**2, q) / (4.0 * length))
    if beta <= 0 or gamma <= 0:
        raise ValueError("Derivative roughness estimates must be positive.")

    if total_counts is None:
        alpha = _estimate_alpha_from_uncertainty(q, intensity_error, length)
    else:
        if total_counts <= 0:
            raise ValueError("total_counts must be positive.")
        alpha = a0 / float(total_counts)

    h_fd = float((alpha / (2.0 * beta)) ** (1.0 / 3.0))
    lambda_opt = float((alpha / (8.0 * np.sqrt(np.pi) * gamma)) ** (1.0 / 5.0))
    lambda_ab = h_fd
    chi = float((gamma / beta) * (alpha / beta) ** (2.0 / 3.0))
    if warn_on_coarse_data and data_bin_width >= h_fd:
        print(
            "WARNING: median data bin width "
            f"({data_bin_width:.6e}) is not smaller than optimal bin width "
            f"h_FD ({h_fd:.6e}). The input data may already be too coarsely binned."
        )

    return BinningResult(
        h_fd=h_fd,
        lambda_opt=lambda_opt,
        lambda_ab=lambda_ab,
        alpha=float(alpha),
        beta=beta,
        gamma=gamma,
        chi=chi,
        data_bin_width=data_bin_width,
        length=length,
        mean_intensity=mean_intensity,
        a0=float(a0),
        window_length=window_length,
        polyorder=polyorder,
        used_savgol=used_savgol,
    )


def rbf_gpr_predict(
    q_train: np.ndarray,
    intensity_train: np.ndarray,
    intensity_error: np.ndarray,
    q_predict: Optional[np.ndarray] = None,
    *,
    kernel_size: float,
    kernel: str = "rbf",
    signal_variance: Optional[float] = None,
    jitter: float = 1.0e-10,
    return_std: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Gaussian-process posterior mean using a covariance-kernel length.

    Use :func:`gp_covariance_length_from_smoothing_width` to map an equivalent
    RBF smoothing width to the covariance length expected by this exact GP
    posterior, then pass the corrected length as ``kernel_size``. Set
    ``return_std=True`` to also return the posterior 1-sigma uncertainty.

    Parameters
    ----------
    kernel:
        ``"rbf"``, ``"matern32"``, or ``"matern52"``. Matern kernels are
        parameterized so their normalized second moment equals that of an RBF
        kernel with length ``kernel_size``.
    """

    q_train = np.asarray(q_train, dtype=float).reshape(-1)
    intensity_train = np.asarray(intensity_train, dtype=float).reshape(-1)
    intensity_error = np.asarray(intensity_error, dtype=float).reshape(-1)
    if q_train.shape != intensity_train.shape or q_train.shape != intensity_error.shape:
        raise ValueError("q_train, intensity_train, and intensity_error must match.")
    mask = np.isfinite(q_train) & np.isfinite(intensity_train) & np.isfinite(intensity_error)
    if not np.all(mask):
        q_train = q_train[mask]
        intensity_train = intensity_train[mask]
        intensity_error = intensity_error[mask]
    order = np.argsort(q_train)
    q_train = q_train[order]
    intensity_train = intensity_train[order]
    intensity_error = intensity_error[order]
    unique = np.concatenate(([True], np.diff(q_train) > 0))
    q_train = q_train[unique]
    intensity_train = intensity_train[unique]
    intensity_error = intensity_error[unique]

    if q_predict is None:
        q_predict = q_train
    q_predict = np.asarray(q_predict, dtype=float).reshape(-1)
    if kernel_size <= 0:
        raise ValueError("kernel_size must be positive.")

    if signal_variance is None:
        signal_variance = float(np.var(intensity_train))
    signal_variance = max(float(signal_variance), jitter)

    k_train = _kernel_matrix(q_train, q_train, kernel_size, signal_variance, kernel)
    noise = np.maximum(intensity_error**2, jitter)
    cov = k_train + np.diag(noise + jitter)
    k_predict = _kernel_matrix(q_predict, q_train, kernel_size, signal_variance, kernel)

    used_cholesky = False
    try:
        chol = np.linalg.cholesky(cov)
        tmp = np.linalg.solve(chol, intensity_train)
        weights = np.linalg.solve(chol.T, tmp)
        used_cholesky = True
        if return_std:
            projected = np.linalg.solve(chol, k_predict.T)
    except np.linalg.LinAlgError:
        weights = np.linalg.solve(cov, intensity_train)
        if return_std:
            projected = np.linalg.solve(cov, k_predict.T)

    mean = k_predict @ weights
    if not return_std:
        return mean

    prior_var = np.full(q_predict.shape, signal_variance, dtype=float)
    if used_cholesky:
        posterior_var = prior_var - np.sum(projected**2, axis=0)
    else:
        posterior_var = prior_var - np.sum(k_predict * projected.T, axis=1)
    posterior_std = np.sqrt(np.maximum(posterior_var, 0.0))
    return mean, posterior_std


def kernel_smoothing_weights(
    q_predict: np.ndarray,
    q_train: np.ndarray,
    *,
    kernel_size: float,
    kernel: str = "rbf",
) -> np.ndarray:
    """Normalized smoothing weights for the supported kernels.

    This is useful for visual diagnostics such as local counting-error
    propagation. The Matern options use the same second-moment matching as
    ``rbf_gpr_predict``.
    """

    weights = _kernel_matrix(q_predict, q_train, kernel_size, 1.0, kernel)
    sums = weights.sum(axis=1, keepdims=True)
    return weights / np.maximum(sums, np.finfo(float).eps)


def gp_covariance_length_from_smoothing_width(
    q_train: np.ndarray,
    intensity_error: np.ndarray,
    *,
    smoothing_width: float,
    signal_variance: float,
    kernel: str = "rbf",
    min_factor: float = 0.2,
    max_factor: float = 20.0,
    tolerance: float = 1.0e-5,
    max_iterations: int = 40,
) -> KernelCorrectionResult:
    """Map an equivalent smoothing width to an exact GP covariance length.

    The normalized equivalent smoother with width ``smoothing_width`` propagates
    observational covariance as ``W Sigma W.T``. The exact GP posterior acts
    through ``S = K (K + Sigma)^-1`` instead. This routine chooses the
    covariance length for which ``trace(S Sigma S.T)`` matches
    ``trace(W Sigma W.T)`` on the supplied grid. The calculation is
    deterministic and supports finite boundaries and heteroscedastic errors.

    The correction is derived directly from the two smoothing operators. It
    depends on sampling, noise, and GP signal variance, so it is recomputed for
    each dataset.
    """

    q_train = np.asarray(q_train, dtype=float).reshape(-1)
    intensity_error = np.asarray(intensity_error, dtype=float).reshape(-1)
    if len(q_train) != len(intensity_error):
        raise ValueError("q_train and intensity_error must have equal length.")
    if len(q_train) < 2:
        raise ValueError("At least two training points are required.")
    if not np.all(np.isfinite(q_train)) or not np.all(np.isfinite(intensity_error)):
        raise ValueError("Training coordinates and errors must be finite.")
    if np.any(intensity_error < 0):
        raise ValueError("intensity_error must be non-negative.")
    if smoothing_width <= 0 or signal_variance <= 0:
        raise ValueError("smoothing_width and signal_variance must be positive.")
    if not (0 < min_factor < max_factor):
        raise ValueError("Require 0 < min_factor < max_factor.")

    noise_variance = np.maximum(intensity_error**2, 1.0e-12)
    identity = np.eye(len(q_train))

    equivalent = _kernel_matrix(
        q_train, q_train, smoothing_width, 1.0, kernel
    )
    equivalent /= np.maximum(
        equivalent.sum(axis=1, keepdims=True), np.finfo(float).eps
    )
    target_noise_variance = float(
        np.sum(equivalent**2 * noise_variance[None, :])
    )

    def propagated_noise_difference(covariance_length: float) -> tuple[float, float]:
        covariance_kernel = _kernel_matrix(
            q_train, q_train, covariance_length, signal_variance, kernel
        )
        covariance = covariance_kernel + np.diag(noise_variance + 1.0e-10)
        smoother = covariance_kernel @ np.linalg.solve(covariance, identity)
        achieved = float(np.sum(smoother**2 * noise_variance[None, :]))
        return achieved - target_noise_variance, achieved

    lower = min_factor * smoothing_width
    upper = max_factor * smoothing_width
    lower_value, _ = propagated_noise_difference(lower)
    upper_value, _ = propagated_noise_difference(upper)
    if lower_value == 0:
        covariance_length = lower
    elif upper_value == 0:
        covariance_length = upper
    elif lower_value * upper_value > 0:
        raise ValueError(
            "Could not bracket the GP covariance-length correction. "
            "Increase max_factor or inspect the supplied noise and signal variance."
        )
    else:
        for _ in range(max_iterations):
            midpoint = 0.5 * (lower + upper)
            midpoint_value, _ = propagated_noise_difference(midpoint)
            if abs(midpoint_value) <= tolerance * max(target_noise_variance, 1.0e-15):
                lower = upper = midpoint
                break
            if lower_value * midpoint_value <= 0:
                upper = midpoint
                upper_value = midpoint_value
            else:
                lower = midpoint
                lower_value = midpoint_value
        covariance_length = 0.5 * (lower + upper)

    _, achieved_noise_variance = propagated_noise_difference(covariance_length)
    return KernelCorrectionResult(
        smoothing_width=float(smoothing_width),
        covariance_length=float(covariance_length),
        target_noise_variance=target_noise_variance,
        achieved_noise_variance=achieved_noise_variance,
        signal_variance=float(signal_variance),
    )


def _clean_and_sort(q: np.ndarray, intensity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(q, dtype=float).reshape(-1)
    intensity = np.asarray(intensity, dtype=float).reshape(-1)
    if q.shape != intensity.shape:
        raise ValueError("q and intensity must have the same shape.")

    mask = np.isfinite(q) & np.isfinite(intensity)
    q = q[mask]
    intensity = intensity[mask]
    order = np.argsort(q)
    q = q[order]
    intensity = intensity[order]

    unique = np.concatenate(([True], np.diff(q) > 0))
    q = q[unique]
    intensity = intensity[unique]
    return q, intensity


def _choose_savgol_window(n: int, window_frac: float, polyorder: int) -> int:
    window = max(polyorder + 2, int(round(window_frac * n)))
    window = min(window, n if n % 2 == 1 else n - 1)
    if window % 2 == 0:
        window += 1
    if window <= polyorder:
        window = polyorder + 2
        if window % 2 == 0:
            window += 1
    return min(window, n if n % 2 == 1 else n - 1)


def _kernel_matrix(
    x: np.ndarray,
    y: np.ndarray,
    length: float,
    signal_variance: float,
    kernel: str,
) -> np.ndarray:
    dx = np.asarray(x, dtype=float)[:, None] - np.asarray(y, dtype=float)[None, :]
    r = np.abs(dx)
    kernel_key = kernel.lower().replace("-", "").replace("_", "")
    if kernel_key in {"rbf", "gaussian"}:
        values = np.exp(-0.5 * (r / length) ** 2)
    elif kernel_key in {"matern32", "mat32"}:
        scaled = 2.0 * r / length
        values = (1.0 + scaled) * np.exp(-scaled)
    elif kernel_key in {"matern52", "mat52"}:
        scaled = np.sqrt(6.0) * r / length
        values = (1.0 + scaled + 2.0 * (r / length) ** 2) * np.exp(-scaled)
    else:
        raise ValueError('kernel must be "rbf", "matern32", or "matern52".')
    return signal_variance * values


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _estimate_alpha_from_uncertainty(
    q: np.ndarray,
    intensity_error: Optional[np.ndarray],
    length: float,
) -> float:
    if intensity_error is None:
        raise ValueError("Provide either total_counts or intensity_error.")
    err = np.asarray(intensity_error, dtype=float).reshape(-1)
    if err.shape != q.shape:
        raise ValueError("intensity_error must have the same shape as q.")
    if not np.all(np.isfinite(err)) or np.any(err <= 0):
        raise ValueError("intensity_error values must be finite and positive.")
    data_bin_width = float(np.median(np.diff(q))) if len(q) > 1 else float(length)
    return float(data_bin_width * np.mean(err**2))


def main() -> None:
    q, intensity, intensity_error, _ = synthetic_scattering_data()
    result = estimate_binning_scales(
        q,
        intensity,
        total_counts=2.0e6,
        intensity_error=intensity_error,
    )

    print("MLSR binning example")
    print(f"data width = {result.data_bin_width:.6e}")
    print(f"h_FD       = {result.h_fd:.6e}")
    print(f"lambda_opt = {result.lambda_opt:.6e}")
    print(f"lambda_ab  = {result.lambda_ab:.6e}")
    print(f"alpha      = {result.alpha:.6e}")
    print(f"beta       = {result.beta:.6e}")
    print(f"gamma      = {result.gamma:.6e}")
    print(f"chi        = {result.chi:.6e}")


if __name__ == "__main__":
    main()
