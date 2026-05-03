"""Synthetic data tools for Speckle Visibility Spectroscopy benchmarks.

Given a normalized intensity autocorrelation C(t), this module generates a
stationary Gaussian fluctuation process with covariance

    E[delta_I(t) delta_I(t + tau)] = mean_I**2 * C(tau)

and optionally applies a nonnegative intensity correction and Poisson counting
observation model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from scipy import integrate
from scipy import linalg
from scipy import signal


CorrelationModel = Callable[[np.ndarray], np.ndarray]
NonnegativeMode = Literal["none", "clip", "softplus"]


@dataclass(frozen=True)
class SVSMetadata:
    """Metadata describing an SVS synthetic time series."""

    model_name: str
    model_params: dict
    n_samples: int
    dt: float
    mean_I: float
    seed: int | None
    nonnegative: NonnegativeMode
    poisson_counts: bool
    kappa: float | None
    spectrum_min_before_clip: float
    spectrum_negative_fraction: float
    target_variance: float
    realized_variance_before_nonnegative: float
    realized_variance_after_nonnegative: float


@dataclass(frozen=True)
class VisibilityResult:
    """Exposure-time-dependent visibility estimates for one intensity trace."""

    T_values: np.ndarray
    effective_T_values: np.ndarray
    window_samples: np.ndarray
    stride_samples: np.ndarray
    segment_averages: tuple[np.ndarray, ...]
    muhat: np.ndarray
    vhat: np.ndarray
    noise_correction: np.ndarray
    Khat2: np.ndarray
    M: np.ndarray
    overlapping: bool


@dataclass(frozen=True)
class VisibilityUncertainty:
    """Moment-based uncertainty estimates for an SVS visibility vector."""

    mu: np.ndarray
    v: np.ndarray
    m3: np.ndarray
    m4: np.ndarray
    A: np.ndarray
    B: np.ndarray
    Sigma_K_diag: np.ndarray
    stderr: np.ndarray


@dataclass(frozen=True)
class VisibilityCovarianceResult:
    """Empirical covariance estimate for an SVS visibility vector.

    For long records with total duration much larger than all exposure times
    and the correlation time, ``L >> T_i, T_j, tau_c``, off-diagonal covariance
    terms are often small. In that common regime, the diagonal
    moment-propagation approximation is usually acceptable.
    """

    Sigma_K: np.ndarray
    stderr: np.ndarray
    block_Khat2: np.ndarray
    block_slices: tuple[slice, ...]


@dataclass(frozen=True)
class BayesianReconstructionResult:
    """Gaussian posterior reconstruction of normalized autocorrelation."""

    tau_grid: np.ndarray
    T_values: np.ndarray
    R: np.ndarray
    C0: np.ndarray
    Sigma_C: np.ndarray
    posterior_mean: np.ndarray
    posterior_cov: np.ndarray
    posterior_std: np.ndarray
    forward_prediction: np.ndarray


def exponential_correlation(
    t: np.ndarray,
    beta: float = 0.5,
    tau_c: float = 1.0,
) -> np.ndarray:
    """Return C(t) = beta * exp(-t / tau_c)."""

    _validate_positive("tau_c", tau_c)
    _validate_nonnegative("beta", beta)
    return beta * np.exp(-np.asarray(t, dtype=float) / tau_c)


def stretched_exponential_correlation(
    t: np.ndarray,
    beta: float = 0.5,
    tau_c: float = 1.0,
    alpha: float = 0.7,
) -> np.ndarray:
    """Return C(t) = beta * exp(-(t / tau_c)**alpha)."""

    _validate_positive("tau_c", tau_c)
    _validate_positive("alpha", alpha)
    _validate_nonnegative("beta", beta)
    return beta * np.exp(-((np.asarray(t, dtype=float) / tau_c) ** alpha))


def double_exponential_correlation(
    t: np.ndarray,
    beta: float = 0.5,
    a: float = 0.5,
    tau1: float = 0.2,
    tau2: float = 2.0,
) -> np.ndarray:
    """Return C(t) = beta * (a * exp(-t/tau1) + (1-a) * exp(-t/tau2))."""

    _validate_positive("tau1", tau1)
    _validate_positive("tau2", tau2)
    _validate_nonnegative("beta", beta)
    if not 0.0 <= a <= 1.0:
        raise ValueError("a must be between 0 and 1.")

    t = np.asarray(t, dtype=float)
    return beta * (a * np.exp(-t / tau1) + (1.0 - a) * np.exp(-t / tau2))


def damped_oscillation_correlation(
    t: np.ndarray,
    beta: float = 0.5,
    tau_c: float = 1.0,
    period: float = 1.0,
) -> np.ndarray:
    """Return C(t) = beta * exp(-t / tau_c) * cos(2*pi*t / period)."""

    _validate_nonnegative("beta", beta)
    _validate_positive("tau_c", tau_c)
    _validate_positive("period", period)
    t = np.asarray(t, dtype=float)
    return beta * np.exp(-t / tau_c) * np.cos(2.0 * np.pi * t / period)


def make_correlation_model(model: str, **params: float) -> CorrelationModel:
    """Create one of the built-in normalized autocorrelation models.

    Parameters
    ----------
    model:
        One of ``"exponential"``, ``"stretched_exponential"``, or
        ``"double_exponential"``.
    **params:
        Parameters passed to the selected model.
    """

    models: dict[str, Callable[..., np.ndarray]] = {
        "exponential": exponential_correlation,
        "stretched_exponential": stretched_exponential_correlation,
        "double_exponential": double_exponential_correlation,
        "damped_oscillation": damped_oscillation_correlation,
    }
    try:
        fn = models[model]
    except KeyError as exc:
        names = ", ".join(sorted(models))
        raise ValueError(f"Unknown model {model!r}. Choose from: {names}.") from exc

    return lambda t: fn(t, **params)


def generate_svs_timeseries(
    n_samples: int,
    dt: float,
    mean_I: float,
    correlation: CorrelationModel,
    *,
    model_name: str = "custom",
    model_params: dict | None = None,
    seed: int | None = None,
    nonnegative: NonnegativeMode = "clip",
    softplus_scale: float | None = None,
    poisson_counts: bool = False,
    kappa: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, SVSMetadata]:
    """Generate a synthetic SVS intensity time series.

    Parameters
    ----------
    n_samples:
        Number of time samples to generate.
    dt:
        Sampling interval.
    mean_I:
        Mean intensity before any optional nonnegative correction.
    correlation:
        Callable returning normalized C(t) for nonnegative lag times.
    model_name, model_params:
        Optional labels included in the returned metadata.
    seed:
        NumPy random seed for reproducibility.
    nonnegative:
        ``"none"`` leaves the Gaussian intensity unchanged, ``"clip"`` clips
        negative values to zero, and ``"softplus"`` smoothly maps values to
        positive values.
    softplus_scale:
        Scale for the softplus transform. Defaults to ``0.05 * mean_I``.
    poisson_counts:
        If true, draw base-sample counts
        ``N_k ~ Poisson(kappa * dt * I_true(t_k))`` and return the count-rate
        equivalent intensity ``I_obs = N_k / (kappa * dt)``. Summing these
        counts over an exposure window of length ``T = n_T * dt`` gives
        ``N_T ~ Poisson(kappa * T * Ibar_T)``, so
        ``E[N_T | Ibar_T] = Var[N_T | Ibar_T] = kappa * T * Ibar_T``.
    kappa:
        Detector/counting proportionality constant for the Poisson model.

    Returns
    -------
    t, I_true, I_obs, metadata:
        ``I_obs`` is ``None`` when ``poisson_counts`` is false.
    """

    _validate_positive_integer("n_samples", n_samples)
    _validate_positive("dt", dt)
    _validate_positive("mean_I", mean_I)
    if poisson_counts:
        _validate_positive("kappa", kappa)
    if nonnegative not in {"none", "clip", "softplus"}:
        raise ValueError("nonnegative must be one of: 'none', 'clip', 'softplus'.")

    rng = np.random.default_rng(seed)
    t = np.arange(n_samples, dtype=float) * dt
    covariance = _periodic_covariance_from_correlation(n_samples, dt, mean_I, correlation)
    delta_I, spectrum_min, neg_frac = _sample_gaussian_from_periodic_covariance(
        covariance,
        rng,
    )

    I_gaussian = mean_I + delta_I
    I_true = apply_nonnegative_intensity(
        I_gaussian,
        mode=nonnegative,
        softplus_scale=softplus_scale,
        reference_mean=mean_I,
    )

    I_obs = None
    if poisson_counts:
        rates = kappa * dt * I_true
        counts = rng.poisson(rates)
        I_obs = counts / (kappa * dt)

    metadata = SVSMetadata(
        model_name=model_name,
        model_params={} if model_params is None else dict(model_params),
        n_samples=int(n_samples),
        dt=float(dt),
        mean_I=float(mean_I),
        seed=seed,
        nonnegative=nonnegative,
        poisson_counts=poisson_counts,
        kappa=float(kappa) if poisson_counts else None,
        spectrum_min_before_clip=float(spectrum_min),
        spectrum_negative_fraction=float(neg_frac),
        target_variance=float(covariance[0]),
        realized_variance_before_nonnegative=float(np.var(delta_I)),
        realized_variance_after_nonnegative=float(np.var(I_true - np.mean(I_true))),
    )
    return t, I_true, I_obs, metadata


def generate_svs_lognormal_timeseries(
    n_samples: int,
    dt: float,
    mean_I: float,
    correlation: CorrelationModel,
    *,
    model_name: str = "custom_lognormal",
    model_params: dict | None = None,
    seed: int | None = None,
    poisson_counts: bool = False,
    kappa: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, SVSMetadata]:
    """Generate a positive SVS trace with the requested normalized correlation.

    The Gaussian generator samples additive fluctuations and therefore needs
    clipping or another nonlinear transform when fluctuations make intensity
    negative. Those transforms change the target normalized correlation. This
    lognormal generator instead samples a latent Gaussian process ``Y`` with

        Cov[Y(t), Y(t + tau)] = log(1 + C(tau))

    and returns

        I(t) = mean_I * exp(Y(t) - Var[Y] / 2).

    Then, in expectation, ``E[I] = mean_I`` and
    ``Cov[I(t), I(t + tau)] / mean_I**2 = C(tau)`` while ``I(t)`` stays
    strictly positive. If Poisson counting is enabled, base-sample counts are
    drawn as ``N_k ~ Poisson(kappa * dt * I_true(t_k))``. Therefore, after
    summing over any exposure window of length ``T``,
    ``E[N_T | Ibar_T] = Var[N_T | Ibar_T] = kappa * T * Ibar_T``. This is the
    preferred synthetic model for large contrast cases such as ``beta = 1``.
    """

    _validate_positive_integer("n_samples", n_samples)
    _validate_positive("dt", dt)
    _validate_positive("mean_I", mean_I)
    if poisson_counts:
        _validate_positive("kappa", kappa)

    rng = np.random.default_rng(seed)
    t = np.arange(n_samples, dtype=float) * dt

    def latent_correlation(lags: np.ndarray) -> np.ndarray:
        target = np.asarray(correlation(lags), dtype=float)
        if np.any(target <= -1.0):
            raise ValueError("Lognormal generation requires C(tau) > -1.")
        return np.log1p(target)

    latent_covariance = _periodic_covariance_from_correlation(
        n_samples,
        dt,
        mean_I=1.0,
        correlation=latent_correlation,
    )
    Y, spectrum_min, neg_frac = _sample_gaussian_from_periodic_covariance(
        latent_covariance,
        rng,
    )
    latent_variance = float(latent_covariance[0])
    I_true = mean_I * np.exp(Y - 0.5 * latent_variance)

    I_obs = None
    if poisson_counts:
        rates = kappa * dt * I_true
        counts = rng.poisson(rates)
        I_obs = counts / (kappa * dt)

    target_variance = float(mean_I**2 * np.asarray(correlation(np.asarray([0.0])))[0])
    metadata = SVSMetadata(
        model_name=model_name,
        model_params={} if model_params is None else dict(model_params),
        n_samples=int(n_samples),
        dt=float(dt),
        mean_I=float(mean_I),
        seed=seed,
        nonnegative="none",
        poisson_counts=poisson_counts,
        kappa=float(kappa) if poisson_counts else None,
        spectrum_min_before_clip=float(spectrum_min),
        spectrum_negative_fraction=float(neg_frac),
        target_variance=target_variance,
        realized_variance_before_nonnegative=float(np.var(I_true - np.mean(I_true))),
        realized_variance_after_nonnegative=float(np.var(I_true - np.mean(I_true))),
    )
    return t, I_true, I_obs, metadata


def apply_nonnegative_intensity(
    intensity: np.ndarray,
    *,
    mode: NonnegativeMode = "clip",
    softplus_scale: float | None = None,
    reference_mean: float | None = None,
) -> np.ndarray:
    """Apply an optional transform that ensures nonnegative intensity."""

    intensity = np.asarray(intensity, dtype=float)
    if mode == "none":
        return intensity.copy()
    if mode == "clip":
        return np.clip(intensity, 0.0, None)
    if mode == "softplus":
        if reference_mean is None:
            reference_mean = float(np.mean(intensity))
        if softplus_scale is None:
            softplus_scale = max(0.05 * reference_mean, np.finfo(float).eps)
        _validate_positive("softplus_scale", softplus_scale)
        x = intensity / softplus_scale
        return softplus_scale * np.logaddexp(0.0, x)
    raise ValueError("mode must be one of: 'none', 'clip', 'softplus'.")


def estimate_normalized_autocorrelation(
    intensity: np.ndarray,
    *,
    dt: float = 1.0,
    max_lag: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate normalized fluctuation autocorrelation from an intensity trace.

    The returned estimate is

        C_hat(tau) = <dI(t) dI(t+tau)> / mean(I)**2

    using an FFT-based unbiased autocovariance estimate.
    """

    _validate_positive("dt", dt)
    intensity = np.asarray(intensity, dtype=float)
    if intensity.ndim != 1:
        raise ValueError("intensity must be one-dimensional.")
    if intensity.size < 2:
        raise ValueError("intensity must contain at least two samples.")
    if max_lag is None:
        max_lag = intensity.size - 1
    if not 0 <= max_lag < intensity.size:
        raise ValueError("max_lag must satisfy 0 <= max_lag < len(intensity).")

    fluctuations = intensity - np.mean(intensity)
    autocov = signal.correlate(fluctuations, fluctuations, mode="full", method="fft")
    autocov = autocov[intensity.size - 1 : intensity.size + max_lag]
    autocov /= np.arange(intensity.size, intensity.size - max_lag - 1, -1)
    lags = np.arange(max_lag + 1, dtype=float) * dt
    return lags, autocov / (np.mean(intensity) ** 2)


def autocorrelation_diagnostics(
    intensity: np.ndarray,
    correlation: CorrelationModel,
    *,
    dt: float,
    max_lag: int,
) -> dict[str, np.ndarray | float]:
    """Compare empirical and target normalized autocorrelation values."""

    lags, empirical = estimate_normalized_autocorrelation(
        intensity,
        dt=dt,
        max_lag=max_lag,
    )
    target = correlation(lags)
    residual = empirical - target
    return {
        "lags": lags,
        "empirical": empirical,
        "target": target,
        "residual": residual,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "max_abs_error": float(np.max(np.abs(residual))),
    }


def segment_averages_for_exposure(
    intensity: np.ndarray,
    *,
    dt: float,
    T: float,
    stride: int | None = None,
) -> tuple[np.ndarray, int, int, float]:
    """Compute segment-averaged intensities for one exposure time.

    Parameters
    ----------
    intensity:
        Uniformly sampled intensity time series.
    dt:
        Sampling interval.
    T:
        Requested exposure time. The discrete window length is
        ``n_T = round(T / dt)``.
    stride:
        Window stride in samples. Defaults to ``n_T`` for non-overlapping
        windows. Use smaller values for sliding/overlapping windows.

    Returns
    -------
    X_T, n_T, stride_samples, effective_T:
        Segment averages, window length in samples, stride in samples, and
        effective exposure time ``n_T * dt``.
    """

    _validate_positive("dt", dt)
    _validate_positive("T", T)
    intensity = np.asarray(intensity, dtype=float)
    if intensity.ndim != 1:
        raise ValueError("intensity must be one-dimensional.")

    n_T = int(round(T / dt))
    if n_T < 1:
        raise ValueError("T is too short for the sampling interval; round(T / dt) < 1.")
    if n_T > intensity.size:
        raise ValueError("T is too long for the supplied intensity trace.")
    if stride is None:
        stride_samples = n_T
    else:
        _validate_positive_integer("stride", stride)
        stride_samples = int(stride)

    starts = np.arange(0, intensity.size - n_T + 1, stride_samples)
    if starts.size == 0:
        raise ValueError("No complete exposure windows fit in the supplied trace.")

    cumsum = np.concatenate(([0.0], np.cumsum(intensity, dtype=float)))
    window_sums = cumsum[starts + n_T] - cumsum[starts]
    return window_sums / n_T, n_T, stride_samples, n_T * dt


def estimate_visibility_vs_exposure(
    intensity: np.ndarray,
    *,
    dt: float,
    T_values: np.ndarray,
    kappa: float,
    stride: int | None = None,
) -> VisibilityResult:
    """Estimate SVS visibility K^2(T) from one long intensity time series.

    By default, each exposure uses non-overlapping windows. Pass ``stride`` as
    a sample count to use sliding/overlapping windows for every exposure time.
    The input ``intensity`` can be a count-rate equivalent trace
    ``I_obs(t_k) = N_k / (kappa * dt)``. Averaging it over an exposure window
    gives ``X_T = N_T / (kappa * T)``, where
    ``E[N_T | Ibar_T] = Var[N_T | Ibar_T] = kappa * T * Ibar_T``. The resulting
    Poisson noise correction is ``muhat_T / (kappa * effective_T)``.
    """

    _validate_positive("dt", dt)
    _validate_positive("kappa", kappa)
    T_values = np.asarray(T_values, dtype=float)
    if T_values.ndim != 1 or T_values.size == 0:
        raise ValueError("T_values must be a nonempty one-dimensional array.")

    segment_averages = []
    window_samples = []
    stride_samples = []
    effective_T_values = []
    muhat = []
    vhat = []
    noise_correction = []
    Khat2 = []
    M = []

    for T in T_values:
        X_T, n_T, stride_T, effective_T = segment_averages_for_exposure(
            intensity,
            dt=dt,
            T=float(T),
            stride=stride,
        )
        if X_T.size < 2:
            raise ValueError(f"Exposure T={T!r} produced fewer than two segments.")

        mu = float(np.mean(X_T))
        variance = float(np.var(X_T, ddof=1))
        correction = mu / (kappa * effective_T)

        segment_averages.append(X_T)
        window_samples.append(n_T)
        stride_samples.append(stride_T)
        effective_T_values.append(effective_T)
        muhat.append(mu)
        vhat.append(variance)
        noise_correction.append(correction)
        Khat2.append((variance - correction) / (mu**2))
        M.append(X_T.size)

    return VisibilityResult(
        T_values=T_values,
        effective_T_values=np.asarray(effective_T_values, dtype=float),
        window_samples=np.asarray(window_samples, dtype=int),
        stride_samples=np.asarray(stride_samples, dtype=int),
        segment_averages=tuple(segment_averages),
        muhat=np.asarray(muhat, dtype=float),
        vhat=np.asarray(vhat, dtype=float),
        noise_correction=np.asarray(noise_correction, dtype=float),
        Khat2=np.asarray(Khat2, dtype=float),
        M=np.asarray(M, dtype=int),
        overlapping=bool(stride is not None),
    )


def estimate_visibility_uncertainty(
    result: VisibilityResult,
    *,
    kappa: float,
) -> VisibilityUncertainty:
    """Estimate diagonal uncertainty for ``Khat^2(T)`` by moment propagation.

    For each exposure time, this uses segment averages ``X_T`` to estimate
    ``mu_T``, ``v_T``, ``m3_T``, and ``m4_T`` directly. The propagated function is

        F_T(mu, v) = (v - mu / (kappa * T)) / mu**2

    with derivatives

        A_T = -1 / (kappa * T * mu_T**2) - 2 * Khat^2(T) / mu_T
        B_T = 1 / mu_T**2.

    The unknown ``K^2(T)`` in ``A_T`` is replaced by the measured ``Khat^2(T)``.
    For independent segment averages at fixed ``T``,

        Var[Khat^2(T)] ~= (1 / M_T) * [
            v_T * A_T**2
            + 2 * m3_T * A_T * B_T
            + (m4_T - v_T**2) * B_T**2
        ].

    Returns a diagonal covariance matrix ``Sigma_K_diag`` and its square-root
    standard-error vector. When the same long record is reused for multiple
    exposure times, cross-covariances can be estimated with
    :func:`estimate_visibility_covariance_from_blocks`. For long records with
    ``L >> T_i, T_j, tau_c``, those off-diagonal terms are often small and this
    diagonal approximation is acceptable.
    """

    _validate_positive("kappa", kappa)
    n_T = result.Khat2.size
    mu = np.empty(n_T, dtype=float)
    v = np.empty(n_T, dtype=float)
    m3 = np.empty(n_T, dtype=float)
    m4 = np.empty(n_T, dtype=float)
    A = np.empty(n_T, dtype=float)
    B = np.empty(n_T, dtype=float)
    variance = np.empty(n_T, dtype=float)

    for i, X_T in enumerate(result.segment_averages):
        X_T = np.asarray(X_T, dtype=float)
        if X_T.ndim != 1 or X_T.size < 2:
            raise ValueError("Each exposure must contain at least two segment averages.")

        mu_i = float(np.mean(X_T))
        if mu_i <= 0.0:
            raise ValueError("Segment-average mean must be positive.")
        centered = X_T - mu_i
        v_i = float(np.mean(centered**2))
        m3_i = float(np.mean(centered**3))
        m4_i = float(np.mean(centered**4))
        T_i = float(result.effective_T_values[i])
        M_i = int(result.M[i])
        A_i = -1.0 / (kappa * T_i * mu_i**2) - 2.0 * float(result.Khat2[i]) / mu_i
        B_i = 1.0 / mu_i**2
        var_i = (
            v_i * A_i**2
            + 2.0 * m3_i * A_i * B_i
            + (m4_i - v_i**2) * B_i**2
        ) / M_i

        mu[i] = mu_i
        v[i] = v_i
        m3[i] = m3_i
        m4[i] = m4_i
        A[i] = A_i
        B[i] = B_i
        variance[i] = max(float(var_i), 0.0)

    Sigma_K_diag = np.diag(variance)
    stderr = np.sqrt(variance)
    return VisibilityUncertainty(
        mu=mu,
        v=v,
        m3=m3,
        m4=m4,
        A=A,
        B=B,
        Sigma_K_diag=Sigma_K_diag,
        stderr=stderr,
    )


def estimate_visibility_covariance_from_blocks(
    intensity: np.ndarray,
    *,
    dt: float,
    T_values: np.ndarray,
    kappa: float,
    n_blocks: int = 16,
    stride: int | None = None,
) -> VisibilityCovarianceResult:
    """Estimate full ``Khat^2(T)`` covariance from block-aligned subrecords.

    This optional estimator is useful when the same long time series is reused
    to estimate several exposure times and a full covariance matrix is desired.
    It splits the record into contiguous blocks, computes one visibility vector
    per block using the same ``T_values``, and returns
    ``cov(block_Khat2) / n_blocks`` as an empirical covariance for the full
    record estimate.

    The interface is intentionally general: callers can choose ``n_blocks`` and
    ``stride`` to match their segmentation assumptions. Blocks that cannot
    support every requested exposure are rejected. For long records with
    ``L >> T_i, T_j, tau_c``, off-diagonal covariance is often small and
    :func:`estimate_visibility_uncertainty` is usually sufficient.
    """

    _validate_positive("dt", dt)
    _validate_positive("kappa", kappa)
    _validate_positive_integer("n_blocks", n_blocks)
    intensity = np.asarray(intensity, dtype=float)
    if intensity.ndim != 1:
        raise ValueError("intensity must be one-dimensional.")
    if n_blocks < 2:
        raise ValueError("n_blocks must be at least 2 for covariance estimation.")

    T_values = np.asarray(T_values, dtype=float)
    if T_values.ndim != 1 or T_values.size == 0:
        raise ValueError("T_values must be a nonempty one-dimensional array.")

    block_size = intensity.size // n_blocks
    if block_size < 2:
        raise ValueError("n_blocks is too large for the supplied intensity trace.")
    max_window = int(round(float(np.max(T_values)) / dt))
    if block_size < max_window * 2:
        raise ValueError(
            "Each block must contain at least two complete windows for the largest T."
        )

    block_Khat2 = []
    block_slices = []
    for i in range(n_blocks):
        start = i * block_size
        stop = intensity.size if i == n_blocks - 1 else (i + 1) * block_size
        block = intensity[start:stop]
        block_result = estimate_visibility_vs_exposure(
            block,
            dt=dt,
            T_values=T_values,
            kappa=kappa,
            stride=stride,
        )
        block_Khat2.append(block_result.Khat2)
        block_slices.append(slice(start, stop))

    block_Khat2_array = np.asarray(block_Khat2, dtype=float)
    Sigma_K = np.cov(block_Khat2_array, rowvar=False, ddof=1) / block_Khat2_array.shape[0]
    if T_values.size == 1:
        Sigma_K = np.asarray([[float(Sigma_K)]], dtype=float)
    stderr = np.sqrt(np.clip(np.diag(Sigma_K), 0.0, None))
    return VisibilityCovarianceResult(
        Sigma_K=Sigma_K,
        stderr=stderr,
        block_Khat2=block_Khat2_array,
        block_slices=tuple(block_slices),
    )


def theoretical_visibility_vs_exposure(
    T_values: np.ndarray,
    correlation: CorrelationModel,
    *,
    epsabs: float = 1e-10,
    epsrel: float = 1e-8,
) -> np.ndarray:
    """Compute theoretical forward visibility K^2(T) from C(t).

    Uses numerical quadrature for

        K^2(T) = (2/T) * integral_0^T (1 - tau/T) C(tau) dtau.
    """

    T_values = np.asarray(T_values, dtype=float)
    if T_values.ndim != 1 or T_values.size == 0:
        raise ValueError("T_values must be a nonempty one-dimensional array.")

    K2 = np.empty_like(T_values, dtype=float)
    for i, T in enumerate(T_values):
        _validate_positive("T", float(T))

        def integrand(tau: float) -> float:
            return float((1.0 - tau / T) * correlation(np.asarray([tau]))[0])

        integral, _ = integrate.quad(integrand, 0.0, float(T), epsabs=epsabs, epsrel=epsrel)
        K2[i] = (2.0 / T) * integral
    return K2


def visibility_from_correlation_grid(
    T_values: np.ndarray,
    tau_grid: np.ndarray,
    C_values: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Compute ``K^2(T)`` by applying the SVS forward matrix to gridded ``C``."""

    R = build_svs_forward_matrix(T_values, tau_grid, weights=weights)
    C_values = np.asarray(C_values, dtype=float)
    if C_values.shape != np.asarray(tau_grid, dtype=float).shape:
        raise ValueError("C_values must have the same shape as tau_grid.")
    return R @ C_values


def quadrature_weights_from_grid(tau_grid: np.ndarray) -> np.ndarray:
    """Return trapezoidal quadrature weights for an arbitrary 1-D grid."""

    tau_grid = np.asarray(tau_grid, dtype=float)
    if tau_grid.ndim != 1 or tau_grid.size < 2:
        raise ValueError("tau_grid must contain at least two points.")
    diffs = np.diff(tau_grid)
    if np.any(diffs <= 0.0):
        raise ValueError("tau_grid must be strictly increasing.")

    weights = np.empty_like(tau_grid, dtype=float)
    weights[0] = 0.5 * diffs[0]
    weights[-1] = 0.5 * diffs[-1]
    if tau_grid.size > 2:
        weights[1:-1] = 0.5 * (diffs[:-1] + diffs[1:])
    return weights


def build_svs_forward_matrix(
    T_values: np.ndarray,
    tau_grid: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Build the SVS forward matrix mapping ``C(tau)`` to ``K^2(T)``.

    The discrete model is ``K = R C`` with

        R_ij = (2/T_i) * (1 - tau_j/T_i) * I(0 <= tau_j <= T_i) * w_j,

    where ``w_j`` are quadrature weights. If ``weights`` is omitted,
    trapezoidal weights are computed from ``tau_grid``.
    """

    T_values = np.asarray(T_values, dtype=float)
    tau_grid = np.asarray(tau_grid, dtype=float)
    if T_values.ndim != 1 or T_values.size == 0:
        raise ValueError("T_values must be a nonempty one-dimensional array.")
    if tau_grid.ndim != 1 or tau_grid.size < 2:
        raise ValueError("tau_grid must contain at least two points.")
    if np.any(T_values <= 0.0):
        raise ValueError("All T_values must be positive.")
    if np.any(tau_grid < 0.0):
        raise ValueError("tau_grid must be nonnegative.")
    if np.any(np.diff(tau_grid) <= 0.0):
        raise ValueError("tau_grid must be strictly increasing.")

    if weights is None:
        weights = quadrature_weights_from_grid(tau_grid)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != tau_grid.shape:
            raise ValueError("weights must have the same shape as tau_grid.")
        if np.any(weights < 0.0):
            raise ValueError("weights must be nonnegative.")

    R = np.zeros((T_values.size, tau_grid.size), dtype=float)
    for i, T in enumerate(T_values):
        mask = tau_grid <= T
        R[i, mask] = (2.0 / T) * (1.0 - tau_grid[mask] / T) * weights[mask]
    return R


def squared_exponential_prior_covariance(
    tau_grid: np.ndarray,
    *,
    lambda_: float,
    sigma_C: float,
    jitter: float = 1e-8,
    covariance_space: Literal["linear", "log"] = "linear",
) -> np.ndarray:
    """Build a squared-exponential GP prior covariance on ``tau_grid``.

    With ``covariance_space="linear"``, distances are measured as
    ``tau_i - tau_j``. With ``covariance_space="log"``, distances are measured
    as ``log(tau_i) - log(tau_j)``, so ``lambda_`` is a length scale in log time.
    """

    _validate_positive("lambda_", lambda_)
    _validate_positive("sigma_C", sigma_C)
    _validate_nonnegative("jitter", jitter)
    tau_grid = np.asarray(tau_grid, dtype=float)
    if tau_grid.ndim != 1 or tau_grid.size < 2:
        raise ValueError("tau_grid must contain at least two points.")
    if covariance_space not in {"linear", "log"}:
        raise ValueError("covariance_space must be 'linear' or 'log'.")

    if covariance_space == "log":
        if np.any(tau_grid <= 0.0):
            raise ValueError("tau_grid must be strictly positive for log-time covariance.")
        covariance_grid = np.log(tau_grid)
    else:
        covariance_grid = tau_grid

    delta = covariance_grid[:, None] - covariance_grid[None, :]
    Sigma_C = sigma_C**2 * np.exp(-(delta**2) / (2.0 * lambda_**2))
    Sigma_C += jitter * np.eye(tau_grid.size)
    return Sigma_C


def condition_prior_at_index(
    C0: np.ndarray,
    Sigma_C: np.ndarray,
    *,
    index: int,
    value: float,
    sigma: float,
    jitter: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Condition a Gaussian prior on a noisy observation of one grid point."""

    _validate_positive("sigma", sigma)
    _validate_nonnegative("jitter", jitter)
    C0 = np.asarray(C0, dtype=float)
    Sigma_C = np.asarray(Sigma_C, dtype=float)
    if Sigma_C.shape != (C0.size, C0.size):
        raise ValueError("Sigma_C must be square with one row per prior mean value.")
    if not 0 <= index < C0.size:
        raise ValueError("index is outside the prior grid.")

    covariance_column = Sigma_C[:, index].copy()
    innovation_variance = float(Sigma_C[index, index] + sigma**2)
    gain = covariance_column / innovation_variance
    conditioned_mean = C0 + gain * (float(value) - C0[index])
    conditioned_cov = Sigma_C - np.outer(covariance_column, covariance_column) / innovation_variance
    conditioned_cov = 0.5 * (conditioned_cov + conditioned_cov.T)
    conditioned_cov += jitter * np.eye(C0.size)
    return conditioned_mean, conditioned_cov


def make_prior_mean(
    tau_grid: np.ndarray,
    C0: Literal["zero", "exponential"] | np.ndarray | Callable[[np.ndarray], np.ndarray] = "zero",
    *,
    beta: float = 0.1,
    tau_c: float = 1.0,
) -> np.ndarray:
    """Create a prior mean vector for Bayesian reconstruction.

    ``C0`` may be ``"zero"``, ``"exponential"``, a user-provided vector, or a
    callable evaluated on ``tau_grid``.
    """

    tau_grid = np.asarray(tau_grid, dtype=float)
    if isinstance(C0, str):
        if C0 == "zero":
            return np.zeros_like(tau_grid, dtype=float)
        if C0 == "exponential":
            return exponential_correlation(tau_grid, beta=beta, tau_c=tau_c)
        raise ValueError("C0 must be 'zero', 'exponential', an array, or a callable.")
    if callable(C0):
        C0_vec = np.asarray(C0(tau_grid), dtype=float)
    else:
        C0_vec = np.asarray(C0, dtype=float)
    if C0_vec.shape != tau_grid.shape:
        raise ValueError("C0 must have the same shape as tau_grid.")
    if not np.all(np.isfinite(C0_vec)):
        raise ValueError("C0 contains non-finite values.")
    return C0_vec


def reconstruct_correlation_bayes(
    Khat: np.ndarray,
    Sigma_K: np.ndarray,
    T_values: np.ndarray,
    tau_grid: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    C0: Literal["zero", "exponential"] | np.ndarray | Callable[[np.ndarray], np.ndarray] = "zero",
    C0_beta: float = 0.1,
    C0_tau_c: float = 1.0,
    lambda_: float = 0.5,
    sigma_C: float = 0.25,
    jitter: float = 1e-8,
    covariance_space: Literal["linear", "log"] = "linear",
    low_tau_anchor_value: float | None = None,
    low_tau_anchor_sigma: float = 0.02,
) -> BayesianReconstructionResult:
    """Reconstruct normalized autocorrelation from noisy SVS visibility.

    The forward model is ``Khat = R C + eta``, with
    ``eta ~ Normal(0, Sigma_K)``. The prior is a Gaussian process,
    ``C ~ Normal(C0, Sigma_C)``, with squared-exponential covariance. The
    covariance can be built in linear time or log time using
    ``covariance_space``.

    The posterior is computed from the precision form

        posterior_cov = inv(R.T @ inv(Sigma_K) @ R + inv(Sigma_C))
        posterior_mean = posterior_cov @ (
            R.T @ inv(Sigma_K) @ Khat + inv(Sigma_C) @ C0
        )

    using Cholesky-based linear solves rather than explicit matrix inversion
    for the weighted terms.
    """

    Khat = np.asarray(Khat, dtype=float)
    T_values = np.asarray(T_values, dtype=float)
    tau_grid = np.asarray(tau_grid, dtype=float)
    Sigma_K = np.asarray(Sigma_K, dtype=float)
    if Khat.ndim != 1:
        raise ValueError("Khat must be one-dimensional.")
    if T_values.shape != Khat.shape:
        raise ValueError("T_values must have the same shape as Khat.")
    if Sigma_K.shape != (Khat.size, Khat.size):
        raise ValueError("Sigma_K must be square with one row per Khat value.")

    R = build_svs_forward_matrix(T_values, tau_grid, weights=weights)
    C0_vec = make_prior_mean(tau_grid, C0, beta=C0_beta, tau_c=C0_tau_c)
    Sigma_C = squared_exponential_prior_covariance(
        tau_grid,
        lambda_=lambda_,
        sigma_C=sigma_C,
        jitter=jitter,
        covariance_space=covariance_space,
    )
    if low_tau_anchor_value is not None:
        C0_vec, Sigma_C = condition_prior_at_index(
            C0_vec,
            Sigma_C,
            index=0,
            value=low_tau_anchor_value,
            sigma=low_tau_anchor_sigma,
            jitter=jitter,
        )

    I_C = np.eye(tau_grid.size)
    cho_K = _cho_factor_with_jitter(Sigma_K, "Sigma_K")
    cho_C = _cho_factor_with_jitter(Sigma_C, "Sigma_C")
    SigmaK_inv_R = linalg.cho_solve(cho_K, R, check_finite=False)
    SigmaK_inv_K = linalg.cho_solve(cho_K, Khat, check_finite=False)
    SigmaC_inv_I = linalg.cho_solve(cho_C, I_C, check_finite=False)
    SigmaC_inv_C0 = linalg.cho_solve(cho_C, C0_vec, check_finite=False)

    posterior_precision = R.T @ SigmaK_inv_R + SigmaC_inv_I
    rhs = R.T @ SigmaK_inv_K + SigmaC_inv_C0
    cho_post = _cho_factor_with_jitter(posterior_precision, "posterior precision")
    posterior_mean = linalg.cho_solve(cho_post, rhs, check_finite=False)
    posterior_cov = linalg.cho_solve(cho_post, I_C, check_finite=False)
    posterior_cov = 0.5 * (posterior_cov + posterior_cov.T)
    posterior_std = np.sqrt(np.clip(np.diag(posterior_cov), 0.0, None))
    forward_prediction = R @ posterior_mean

    return BayesianReconstructionResult(
        tau_grid=tau_grid,
        T_values=T_values,
        R=R,
        C0=C0_vec,
        Sigma_C=Sigma_C,
        posterior_mean=posterior_mean,
        posterior_cov=posterior_cov,
        posterior_std=posterior_std,
        forward_prediction=forward_prediction,
    )


def plot_visibility_comparison(
    result: VisibilityResult,
    theoretical_K2: np.ndarray,
    *,
    Sigma_K: np.ndarray | None = None,
    stderr: np.ndarray | None = None,
    ax=None,
):
    """Plot measured ``Khat^2(T)`` against theory, optionally with errors."""

    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)

    theoretical_K2 = np.asarray(theoretical_K2, dtype=float)
    if theoretical_K2.shape != result.Khat2.shape:
        raise ValueError("theoretical_K2 must have the same shape as result.Khat2.")
    if Sigma_K is not None and stderr is not None:
        raise ValueError("Pass either Sigma_K or stderr, not both.")
    if Sigma_K is not None:
        Sigma_K = np.asarray(Sigma_K, dtype=float)
        if Sigma_K.shape != (result.Khat2.size, result.Khat2.size):
            raise ValueError("Sigma_K must be square with one row per T value.")
        stderr = np.sqrt(np.clip(np.diag(Sigma_K), 0.0, None))
    if stderr is not None:
        stderr = np.asarray(stderr, dtype=float)
        if stderr.shape != result.Khat2.shape:
            raise ValueError("stderr must have the same shape as result.Khat2.")

    ax.plot(result.effective_T_values, theoretical_K2, marker="o", label=r"theoretical $K^2(T)$")
    if stderr is None:
        ax.plot(result.effective_T_values, result.Khat2, marker="s", label=r"measured $\hat{K}^2(T)$")
    else:
        ax.errorbar(
            result.effective_T_values,
            result.Khat2,
            yerr=stderr,
            marker="s",
            capsize=3,
            linestyle="-",
            label=r"measured $\hat{K}^2(T)$",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"exposure time $T$")
    ax.set_ylabel(r"visibility $K^2$")
    ax.legend()
    return ax


def plot_bayesian_reconstruction(
    reconstruction: BayesianReconstructionResult,
    Khat: np.ndarray,
    *,
    Sigma_K: np.ndarray | None = None,
    stderr: np.ndarray | None = None,
    true_C: np.ndarray | None = None,
    empirical_tau: np.ndarray | None = None,
    empirical_C: np.ndarray | None = None,
    ax_c=None,
    ax_k=None,
):
    """Plot reconstructed ``C(t)`` and forward-predicted visibility.

    The first panel shows the true autocorrelation when provided, posterior
    mean, and a 1-sigma pointwise uncertainty band. The second panel shows noisy
    measured ``Khat^2(T)`` and ``R @ posterior_mean``.
    """

    if ax_c is None or ax_k is None:
        import matplotlib.pyplot as plt

        _, axes = plt.subplots(2, 1, figsize=(8, 7), constrained_layout=True)
        ax_c = axes[0]
        ax_k = axes[1]

    Khat = np.asarray(Khat, dtype=float)
    if Khat.shape != reconstruction.forward_prediction.shape:
        raise ValueError("Khat must match the reconstructed forward prediction shape.")
    if Sigma_K is not None and stderr is not None:
        raise ValueError("Pass either Sigma_K or stderr, not both.")
    if Sigma_K is not None:
        Sigma_K = np.asarray(Sigma_K, dtype=float)
        if Sigma_K.shape != (Khat.size, Khat.size):
            raise ValueError("Sigma_K must be square with one row per Khat value.")
        stderr = np.sqrt(np.clip(np.diag(Sigma_K), 0.0, None))
    if stderr is not None:
        stderr = np.asarray(stderr, dtype=float)
        if stderr.shape != Khat.shape:
            raise ValueError("stderr must have the same shape as Khat.")
    if true_C is not None:
        true_C = np.asarray(true_C, dtype=float)
        if true_C.shape != reconstruction.posterior_mean.shape:
            raise ValueError("true_C must match posterior_mean shape.")
        ax_c.plot(
            reconstruction.tau_grid,
            true_C,
            color="black",
            linewidth=2.0,
            label=r"ground truth $C(\tau)$",
        )
    if empirical_tau is not None or empirical_C is not None:
        if empirical_tau is None or empirical_C is None:
            raise ValueError("Pass empirical_tau and empirical_C together.")
        empirical_tau = np.asarray(empirical_tau, dtype=float)
        empirical_C = np.asarray(empirical_C, dtype=float)
        if empirical_tau.shape != empirical_C.shape:
            raise ValueError("empirical_tau and empirical_C must have the same shape.")
        ax_c.plot(
            empirical_tau,
            empirical_C,
            color="blue",
            linewidth=1.5,
            alpha=0.85,
            label=r"FFT empirical $C(\tau)$",
        )

    lower = reconstruction.posterior_mean - reconstruction.posterior_std
    upper = reconstruction.posterior_mean + reconstruction.posterior_std
    ax_c.fill_between(
        reconstruction.tau_grid,
        lower,
        upper,
        color="red",
        alpha=0.25,
        label=r"$1\sigma$ posterior band",
    )
    ax_c.plot(
        reconstruction.tau_grid,
        reconstruction.posterior_mean,
        color="red",
        linewidth=2.0,
        label=r"reconstructed $C(\tau)$",
    )
    ax_c.set_xlabel(r"delay time $\tau$")
    ax_c.set_ylabel(r"normalized autocorrelation $C(\tau)$")
    ax_c.legend()

    if stderr is None:
        ax_k.plot(reconstruction.T_values, Khat, marker="s", label=r"measured $\hat{K}^2(T)$")
    else:
        ax_k.errorbar(
            reconstruction.T_values,
            Khat,
            yerr=stderr,
            marker="s",
            capsize=3,
            linestyle="none",
            label=r"measured $\hat{K}^2(T)$",
        )
    ax_k.plot(
        reconstruction.T_values,
        reconstruction.forward_prediction,
        marker="o",
        color="red",
        label=r"forward-predicted $K^2(T)$",
    )
    ax_k.set_xscale("log")
    ax_k.set_xlabel(r"exposure time $T$")
    ax_k.set_ylabel(r"visibility $K^2$")
    ax_k.legend()
    return ax_c, ax_k


def _periodic_covariance_from_correlation(
    n_samples: int,
    dt: float,
    mean_I: float,
    correlation: CorrelationModel,
) -> np.ndarray:
    """Build the first row of an even periodic covariance matrix."""

    lag_indices = np.arange(n_samples)
    circular_lags = np.minimum(lag_indices, n_samples - lag_indices) * dt
    normalized_covariance = np.asarray(correlation(circular_lags), dtype=float)
    if normalized_covariance.shape != (n_samples,):
        raise ValueError("correlation must return an array with the same shape as its input.")
    if not np.all(np.isfinite(normalized_covariance)):
        raise ValueError("correlation returned non-finite values.")
    return (mean_I**2) * normalized_covariance


def _sample_gaussian_from_periodic_covariance(
    covariance: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float]:
    """Sample a real Gaussian vector with the given circulant covariance."""

    spectrum = np.real(np.fft.fft(covariance))
    spectrum_min = float(np.min(spectrum))
    tolerance = 1e-12 * max(float(np.max(np.abs(spectrum))), 1.0)
    negative = spectrum < -tolerance
    negative_fraction = float(np.mean(negative))
    spectrum = np.clip(spectrum, 0.0, None)

    white_noise = rng.normal(size=covariance.size)
    white_spectrum = np.fft.fft(white_noise)
    colored = np.fft.ifft(white_spectrum * np.sqrt(spectrum)).real
    return colored, spectrum_min, negative_fraction


def _cho_factor_with_jitter(
    matrix: np.ndarray,
    name: str,
    *,
    initial_jitter: float = 1e-12,
    max_tries: int = 8,
):
    """Cholesky factorize a symmetric matrix, adding tiny jitter if needed."""

    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values.")

    symmetric = 0.5 * (matrix + matrix.T)
    eye = np.eye(symmetric.shape[0])
    scale = max(float(np.max(np.abs(np.diag(symmetric)))), 1.0)
    jitter = 0.0
    last_error: Exception | None = None
    for attempt in range(max_tries):
        try:
            return linalg.cho_factor(
                symmetric + jitter * eye,
                lower=True,
                check_finite=False,
            )
        except linalg.LinAlgError as exc:
            last_error = exc
            jitter = initial_jitter * scale * (10.0**attempt)
    raise ValueError(f"{name} is not positive definite, even after jitter.") from last_error


def _validate_positive_integer(name: str, value: int) -> None:
    if not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_nonnegative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")
