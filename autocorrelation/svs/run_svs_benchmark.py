"""Run a complete synthetic SVS benchmark.

Pipeline:
known C(t) -> synthetic I(t) -> Poisson observations -> SVS visibility
-> uncertainty estimate -> Bayesian/GPR reconstruction of C(t).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg

import svs_lib as svs


@dataclass(frozen=True)
class BenchmarkConfig:
    # signal quality and correlation parameters
    n_samples: int = 2048
    total_time: float = 1200
    dt: float = 0.01
    mean_I: float = 1.0
    kappa: float = 10.0
    correlation_model: str = "double_exponential"
    beta: float = 1.0
    tau_c: float = 1.0
    f: float = 0.6
    tau1: float = 0.5
    tau2: float = 2.5
    period: float = 0.5
    
    # time lag
    T_min: float = 0.01
    T_max: float = 5.0
    N_T: int = 31
    T_spacing: str = "log"
    tau_grid_size: int = 50
    
    # Bayesian reconstruction parameters
    lambda_: float = 0.2
    sigma_C: float = 0.1
    jitter: float = 1e-8
    prior_covariance_space: str = "log"
    seed: int = 20
    output_folder: str = "svs_benchmark_output_two_step_decay"
    intensity_model: str = "lognormal"
    
    # prior parameters
    prior_mean: str = "direct_smooth"
    prior_beta: float = 1.0
    prior_tau_c: float | None = None
    prior_tau_max_lag_time: float = 5.0
    prior_smooth_sigma: float = 20
    low_tau_prior_anchor: bool = False
    low_tau_prior_anchor_value: float = 1.0
    low_tau_prior_anchor_sigma: float = 0.1
    
    # misc
    nonnegative: str = "softplus"
    softplus_scale_fraction: float = 0.5
    full_covariance: bool = False
    covariance_blocks: int = 16
    max_lag_time: float = 8.0
    show_autocorr_fft: bool = False
    show_autocorr_direct: bool = True
    show_autocorr_pinv: bool = False
    show_autocorr_bayes: bool = True
    show_prior_mean: bool = False
    autocorr_scale: str = "linear"
    autocorr_range: tuple[float, float] = (-0.2, 1.2)
    autocorr_time_scale: str = "log"
    autocorr_time_range: tuple[float, float] = (0.05, 5.0)
    pinv_rcond: float = 1e-8


@dataclass(frozen=True)
class BenchmarkMetrics:
    mse_C: float
    visibility_chi_square: float
    visibility_reduced_chi_square: float
    visibility_rmse: float
    visibility_max_abs_residual: float
    calibrated_visibility_rmse: float
    calibrated_visibility_max_abs_residual: float
    pinv_mse_C: float
    min_observed_intensity: float
    observed_zero_count: int
    stderr_min: float
    stderr_max: float
    prior_tau_c: float


@dataclass(frozen=True)
class LogSpacingDiagnostics:
    exposure_log_width: float | None
    tau_log_width: float


@dataclass(frozen=True)
class BenchmarkResult:
    config: BenchmarkConfig
    metrics: BenchmarkMetrics
    figures: dict[str, str]
    log_spacing: LogSpacingDiagnostics


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mean-I", type=float, default=BenchmarkConfig.mean_I)
    parser.add_argument(
        "--correlation-model",
        choices=("exponential", "double_exponential", "damped_oscillation"),
        default=BenchmarkConfig.correlation_model,
    )
    parser.add_argument("--beta", type=float, default=BenchmarkConfig.beta)
    parser.add_argument("--tau-c", type=float, default=BenchmarkConfig.tau_c)
    parser.add_argument("--f", type=float, default=BenchmarkConfig.f)
    parser.add_argument("--tau1", type=float, default=BenchmarkConfig.tau1)
    parser.add_argument("--tau2", type=float, default=BenchmarkConfig.tau2)
    parser.add_argument("--period", type=float, default=BenchmarkConfig.period)
    parser.add_argument("--total-time", type=float, default=BenchmarkConfig.total_time)
    parser.add_argument("--n-samples", type=int, default=BenchmarkConfig.n_samples)
    parser.add_argument("--dt", type=float, default=BenchmarkConfig.dt)
    parser.add_argument("--kappa", type=float, default=BenchmarkConfig.kappa)
    parser.add_argument("--T-min", type=float, default=BenchmarkConfig.T_min)
    parser.add_argument("--T-max", type=float, default=BenchmarkConfig.T_max)
    parser.add_argument("--N-T", type=int, default=BenchmarkConfig.N_T)
    parser.add_argument(
        "--T-spacing",
        choices=("powers_of_two", "log"),
        default=BenchmarkConfig.T_spacing,
    )
    parser.add_argument("--tau-grid-size", type=int, default=BenchmarkConfig.tau_grid_size)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=BenchmarkConfig.lambda_)
    parser.add_argument("--sigma-C", type=float, default=BenchmarkConfig.sigma_C)
    parser.add_argument("--jitter", type=float, default=BenchmarkConfig.jitter)
    parser.add_argument(
        "--prior-covariance-space",
        choices=("linear", "log"),
        default=BenchmarkConfig.prior_covariance_space,
        help="build the squared-exponential GP prior covariance in linear tau or log tau",
    )
    parser.add_argument("--seed", type=int, default=BenchmarkConfig.seed)
    parser.add_argument("--output-folder", default=BenchmarkConfig.output_folder)
    parser.add_argument(
        "--intensity-model",
        choices=("lognormal", "gaussian"),
        default=BenchmarkConfig.intensity_model,
    )
    parser.add_argument(
        "--prior-mean",
        choices=("zero", "exponential", "direct_smooth"),
        default=BenchmarkConfig.prior_mean,
    )
    parser.add_argument("--prior-beta", type=float, default=BenchmarkConfig.prior_beta)
    parser.add_argument(
        "--prior-tau-c",
        type=float,
        default=BenchmarkConfig.prior_tau_c,
        help="prior exponential decay time; omit to estimate from I(t)",
    )
    parser.add_argument(
        "--prior-tau-max-lag-time",
        type=float,
        default=BenchmarkConfig.prior_tau_max_lag_time,
    )
    parser.add_argument("--prior-smooth-sigma", type=float, default=BenchmarkConfig.prior_smooth_sigma)
    parser.add_argument(
        "--low-tau-prior-anchor",
        action=argparse.BooleanOptionalAction,
        default=BenchmarkConfig.low_tau_prior_anchor,
        help="condition the prior so its lowest tau point is close to the anchor value",
    )
    parser.add_argument(
        "--low-tau-prior-anchor-value",
        type=float,
        default=BenchmarkConfig.low_tau_prior_anchor_value,
    )
    parser.add_argument(
        "--low-tau-prior-anchor-sigma",
        type=float,
        default=BenchmarkConfig.low_tau_prior_anchor_sigma,
        help="smaller values more strongly shrink prior variance at the lowest tau point",
    )
    parser.add_argument(
        "--nonnegative",
        choices=("none", "clip", "softplus"),
        default=BenchmarkConfig.nonnegative,
    )
    parser.add_argument(
        "--softplus-scale-fraction",
        type=float,
        default=BenchmarkConfig.softplus_scale_fraction,
    )
    parser.add_argument("--full-covariance", action="store_true")
    parser.add_argument("--covariance-blocks", type=int, default=BenchmarkConfig.covariance_blocks)
    parser.add_argument("--max-lag-time", type=float, default=BenchmarkConfig.max_lag_time)
    parser.add_argument(
        "--show-autocorr-fft",
        action=argparse.BooleanOptionalAction,
        default=BenchmarkConfig.show_autocorr_fft,
    )
    parser.add_argument(
        "--show-autocorr-direct",
        action=argparse.BooleanOptionalAction,
        default=BenchmarkConfig.show_autocorr_direct,
    )
    parser.add_argument(
        "--show-autocorr-pinv",
        action=argparse.BooleanOptionalAction,
        default=BenchmarkConfig.show_autocorr_pinv,
    )
    parser.add_argument(
        "--show-autocorr-bayes",
        action=argparse.BooleanOptionalAction,
        default=BenchmarkConfig.show_autocorr_bayes,
    )
    parser.add_argument(
        "--show-prior-mean",
        action=argparse.BooleanOptionalAction,
        default=BenchmarkConfig.show_prior_mean,
        help="show the GP prior mean used for Bayesian reconstruction",
    )
    parser.add_argument(
        "--autocorr-time-scale",
        choices=("linear", "log"),
        default=BenchmarkConfig.autocorr_time_scale,
    )
    parser.add_argument(
        "--autocorr-scale",
        choices=("linear", "log"),
        default=BenchmarkConfig.autocorr_scale,
        help="y-axis scale for the autocorrelation panel",
    )
    parser.add_argument(
        "--autocorr-range",
        nargs=2,
        type=float,
        default=BenchmarkConfig.autocorr_range,
        help="y-axis range (min max) for the autocorrelation panel; omit for automatic range",
    )
    parser.add_argument(
        "--autocorr-time-range",
        nargs=2,
        type=float,
        default=BenchmarkConfig.autocorr_time_range,
        help="time range (min max) for autocorrelation plots; omit for full range",
    )
    parser.add_argument("--pinv-rcond", type=float, default=BenchmarkConfig.pinv_rcond)
    args = parser.parse_args()
    return BenchmarkConfig(
        mean_I=args.mean_I,
        correlation_model=args.correlation_model,
        beta=args.beta,
        tau_c=args.tau_c,
        f=args.f,
        tau1=args.tau1,
        tau2=args.tau2,
        period=args.period,
        n_samples=args.n_samples,
        total_time=args.total_time,
        dt=args.dt,
        kappa=args.kappa,
        T_min=args.T_min,
        T_max=args.T_max,
        N_T=args.N_T,
        T_spacing=args.T_spacing,
        tau_grid_size=args.tau_grid_size,
        lambda_=args.lambda_,
        sigma_C=args.sigma_C,
        jitter=args.jitter,
        prior_covariance_space=args.prior_covariance_space,
        seed=args.seed,
        output_folder=args.output_folder,
        intensity_model=args.intensity_model,
        prior_mean=args.prior_mean,
        prior_beta=args.prior_beta,
        prior_tau_c=args.prior_tau_c,
        prior_tau_max_lag_time=args.prior_tau_max_lag_time,
        prior_smooth_sigma=args.prior_smooth_sigma,
        low_tau_prior_anchor=args.low_tau_prior_anchor,
        low_tau_prior_anchor_value=args.low_tau_prior_anchor_value,
        low_tau_prior_anchor_sigma=args.low_tau_prior_anchor_sigma,
        nonnegative=args.nonnegative,
        softplus_scale_fraction=args.softplus_scale_fraction,
        full_covariance=args.full_covariance,
        covariance_blocks=args.covariance_blocks,
        max_lag_time=args.max_lag_time,
        show_autocorr_fft=args.show_autocorr_fft,
        show_autocorr_direct=args.show_autocorr_direct,
        show_autocorr_pinv=args.show_autocorr_pinv,
        show_autocorr_bayes=args.show_autocorr_bayes,
        show_prior_mean=args.show_prior_mean,
        autocorr_scale=args.autocorr_scale,
        autocorr_range=tuple(args.autocorr_range) if args.autocorr_range else None,
        autocorr_time_scale=args.autocorr_time_scale,
        autocorr_time_range=tuple(args.autocorr_time_range) if args.autocorr_time_range else None,
        pinv_rcond=args.pinv_rcond,
    )


def generate_observed_series(config: BenchmarkConfig):
    if config.total_time is None:
        n_samples = int(config.n_samples)
    else:
        n_samples = int(round(config.total_time / config.dt))
    if n_samples < 2:
        raise ValueError("The benchmark must contain at least two samples.")

    correlation, model_params = make_target_correlation(config)
    if config.intensity_model == "lognormal":
        generated = svs.generate_svs_lognormal_timeseries(
            n_samples,
            config.dt,
            config.mean_I,
            correlation,
            model_name=f"{config.correlation_model}_lognormal",
            model_params=model_params,
            seed=config.seed,
            poisson_counts=True,
            kappa=config.kappa,
        )
    elif config.intensity_model == "gaussian":
        generated = svs.generate_svs_timeseries(
            n_samples,
            config.dt,
            config.mean_I,
            correlation,
            model_name=f"{config.correlation_model}_gaussian",
            model_params=model_params,
            seed=config.seed,
            nonnegative=config.nonnegative,
            softplus_scale=config.softplus_scale_fraction * config.mean_I,
            poisson_counts=True,
            kappa=config.kappa,
        )
    else:
        raise ValueError(f"Unknown intensity_model: {config.intensity_model!r}")
    return (*generated, correlation)


def make_target_correlation(config: BenchmarkConfig) -> tuple[svs.CorrelationModel, dict[str, float]]:
    """Create the requested target normalized autocorrelation model."""

    if config.correlation_model == "exponential":
        params = {"beta": config.beta, "tau_c": config.tau_c}
    elif config.correlation_model == "double_exponential":
        params = {
            "beta": config.beta,
            "a": config.f,
            "tau1": config.tau1,
            "tau2": config.tau2,
        }
    elif config.correlation_model == "damped_oscillation":
        params = {
            "beta": config.beta,
            "tau_c": config.tau_c,
            "period": config.period,
        }
    else:
        raise ValueError(f"Unknown correlation_model: {config.correlation_model!r}")
    return svs.make_correlation_model(config.correlation_model, **params), params


def make_exposure_times(config: BenchmarkConfig, n_samples: int) -> np.ndarray:
    """Create exposure times, optionally using power-of-two window sizes."""

    if config.T_spacing == "log":
        return np.geomspace(config.T_min, config.T_max, config.N_T)
    if config.T_spacing != "powers_of_two":
        raise ValueError(f"Unknown T_spacing: {config.T_spacing!r}")

    min_window = max(1, int(np.ceil(config.T_min / config.dt)))
    max_window = max(min_window, int(np.floor(config.T_max / config.dt)))
    max_window = min(max_window, n_samples // 2)
    powers = []
    window = 1
    while window < min_window:
        window *= 2
    while window <= max_window:
        powers.append(window)
        window *= 2
    if not powers:
        raise ValueError("No power-of-two exposure windows fit the requested T range.")
    return config.dt * np.asarray(powers, dtype=float)


def estimate_visibility_and_covariance(I_obs: np.ndarray, config: BenchmarkConfig):
    T_values = make_exposure_times(config, I_obs.size)
    visibility = svs.estimate_visibility_vs_exposure(
        I_obs,
        dt=config.dt,
        T_values=T_values,
        kappa=config.kappa,
    )
    diagonal = svs.estimate_visibility_uncertainty(visibility, kappa=config.kappa)
    Sigma_K = diagonal.Sigma_K_diag
    full = None
    if config.full_covariance:
        full = svs.estimate_visibility_covariance_from_blocks(
            I_obs,
            dt=config.dt,
            T_values=T_values,
            kappa=config.kappa,
            n_blocks=config.covariance_blocks,
        )
        Sigma_K = full.Sigma_K
    return visibility, diagonal, Sigma_K, full


def estimate_exponential_tau_from_intensity(
    intensity: np.ndarray,
    *,
    dt: float,
    beta: float,
    max_lag_time: float,
) -> float:
    """Estimate a rough exponential decay time from FFT autocorrelation."""

    max_lag = min(int(round(max_lag_time / dt)), np.asarray(intensity).size - 1)
    lags, empirical = svs.estimate_normalized_autocorrelation(
        intensity,
        dt=dt,
        max_lag=max_lag,
    )
    if lags.size < 2:
        raise ValueError("Not enough lags to estimate prior tau_c.")

    threshold = beta / np.e
    positive = empirical > 0.0
    below = np.where((lags > 0.0) & positive & (empirical <= threshold))[0]
    if below.size:
        i = int(below[0])
        prev = max(i - 1, 0)
        x0, x1 = float(lags[prev]), float(lags[i])
        y0, y1 = float(empirical[prev]), float(empirical[i])
        if y1 != y0:
            return x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)
        return x1

    fit_mask = (lags > 0.0) & positive & (empirical < beta) & (empirical > 0.05 * beta)
    if np.count_nonzero(fit_mask) >= 2:
        x = lags[fit_mask]
        y = np.log(empirical[fit_mask] / beta)
        slope, _ = np.polyfit(x, y, deg=1)
        if slope < 0.0:
            return float(-1.0 / slope)

    return max_lag_time


def estimate_direct_normalized_autocorrelation(
    intensity: np.ndarray,
    *,
    dt: float,
    max_lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate normalized autocorrelation by direct time-domain averaging."""

    intensity = np.asarray(intensity, dtype=float)
    if intensity.ndim != 1:
        raise ValueError("intensity must be one-dimensional.")
    if not 0 <= max_lag < intensity.size:
        raise ValueError("max_lag must satisfy 0 <= max_lag < len(intensity).")

    mean_I = float(np.mean(intensity))
    fluctuations = intensity - mean_I
    autocov = np.empty(max_lag + 1, dtype=float)
    for lag in range(max_lag + 1):
        left = fluctuations[: intensity.size - lag]
        right = fluctuations[lag:]
        autocov[lag] = float(np.mean(left * right))
    lags = np.arange(max_lag + 1, dtype=float) * dt
    return lags, autocov / (mean_I**2)


def subtract_poisson_zero_lag(
    C: np.ndarray,
    intensity: np.ndarray,
    *,
    dt: float,
    kappa: float,
) -> np.ndarray:
    """Remove the zero-lag Poisson counting variance from normalized C."""

    corrected = np.asarray(C, dtype=float).copy()
    if corrected.size:
        corrected[0] -= 1.0 / (kappa * dt * float(np.mean(intensity)))
    return corrected


def gaussian_smooth_1d(values: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth a 1-D vector with a normalized Gaussian kernel."""

    values = np.asarray(values, dtype=float)
    if sigma <= 0.0:
        return values.copy()
    radius = max(1, int(np.ceil(4.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= np.sum(kernel)
    padded = np.pad(values, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def smoothed_direct_prior_mean(
    I_obs: np.ndarray,
    tau_grid: np.ndarray,
    config: BenchmarkConfig,
) -> np.ndarray:
    """Build a smoothed direct-autocorrelation prior mean from ``I_obs``."""

    max_lag = min(int(round(float(np.max(tau_grid)) / config.dt)), I_obs.size - 1)
    tau_direct, C_direct = estimate_direct_normalized_autocorrelation(
        I_obs,
        dt=config.dt,
        max_lag=max_lag,
    )
    C_direct = subtract_poisson_zero_lag(
        C_direct,
        I_obs,
        dt=config.dt,
        kappa=config.kappa,
    )
    C_smooth = gaussian_smooth_1d(C_direct, config.prior_smooth_sigma)
    return np.interp(tau_grid, tau_direct, C_smooth)


def reconstruct_correlation(
    visibility: svs.VisibilityResult,
    Sigma_K: np.ndarray,
    I_obs: np.ndarray,
    config: BenchmarkConfig,
) -> tuple[svs.BayesianReconstructionResult, np.ndarray, float]:
    tau_min = max(
        np.finfo(float).tiny,
        min(config.dt, float(np.min(visibility.effective_T_values))) / 100.0,
    )
    tau_grid = np.geomspace(
        tau_min,
        float(np.max(visibility.effective_T_values)),
        config.tau_grid_size,
    )
    prior_tau_c = config.prior_tau_c
    if prior_tau_c is None:
        prior_tau_c = estimate_exponential_tau_from_intensity(
            I_obs,
            dt=config.dt,
            beta=config.prior_beta,
            max_lag_time=config.prior_tau_max_lag_time,
        )
    if config.prior_mean == "direct_smooth":
        C0: str | np.ndarray = smoothed_direct_prior_mean(I_obs, tau_grid, config)
    else:
        C0 = config.prior_mean
    reconstruction = svs.reconstruct_correlation_bayes(
        visibility.Khat2,
        Sigma_K,
        visibility.effective_T_values,
        tau_grid,
        C0=C0,
        C0_beta=config.prior_beta,
        C0_tau_c=prior_tau_c,
        lambda_=config.lambda_,
        sigma_C=config.sigma_C,
        jitter=config.jitter,
        covariance_space=config.prior_covariance_space,
        low_tau_anchor_value=(
            config.low_tau_prior_anchor_value if config.low_tau_prior_anchor else None
        ),
        low_tau_anchor_sigma=config.low_tau_prior_anchor_sigma,
    )
    target_correlation, _ = make_target_correlation(config)
    true_C = target_correlation(tau_grid)
    return reconstruction, true_C, float(prior_tau_c)


def estimate_pinv_correlation(
    visibility: svs.VisibilityResult,
    tau_grid: np.ndarray,
    *,
    rcond: float,
) -> np.ndarray:
    """Estimate C(tau) by direct pseudoinverse, C = pinv(R) Khat."""

    R = svs.build_svs_forward_matrix(visibility.effective_T_values, tau_grid)
    return np.linalg.pinv(R, rcond=rcond) @ visibility.Khat2


def direct_visibility_calibration(
    I_obs: np.ndarray,
    visibility: svs.VisibilityResult,
    config: BenchmarkConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward-project direct autocorrelation from ``I_obs`` to visibility."""

    max_lag = min(
        int(round(float(np.max(visibility.effective_T_values)) / config.dt)),
        I_obs.size - 1,
    )
    tau, C_direct = estimate_direct_normalized_autocorrelation(
        I_obs,
        dt=config.dt,
        max_lag=max_lag,
    )
    C_direct = subtract_poisson_zero_lag(
        C_direct,
        I_obs,
        dt=config.dt,
        kappa=config.kappa,
    )
    calibrated_K = svs.visibility_from_correlation_grid(
        visibility.effective_T_values,
        tau,
        C_direct,
    )
    return calibrated_K, tau, C_direct


def forward_visibility_from_target(
    visibility: svs.VisibilityResult,
    correlation: svs.CorrelationModel,
    *,
    tau_step: float,
) -> np.ndarray:
    """Forward-project the target C(tau) through the same discrete R matrix."""

    tau_grid = np.arange(
        0.0,
        float(np.max(visibility.effective_T_values)) + 0.5 * tau_step,
        tau_step,
        dtype=float,
    )
    return svs.visibility_from_correlation_grid(
        visibility.effective_T_values,
        tau_grid,
        correlation(tau_grid),
    )


def plot_measured_and_forward_visibility(
    visibility: svs.VisibilityResult,
    uncertainty: svs.VisibilityUncertainty,
    forward_K: np.ndarray,
    *,
    ax,
) -> None:
    """Plot measured visibility and the reconstruction's forward prediction."""

    ax.errorbar(
        visibility.effective_T_values,
        visibility.Khat2,
        yerr=uncertainty.stderr,
        marker="s",
        capsize=3,
        linestyle="none",
        label=r"measured $\widehat{K}^2(T)$",
    )
    ax.plot(
        visibility.effective_T_values,
        forward_K,
        color="black",
        marker=None,
        linestyle="-",
        label=r"$\int_0^\infty R_{\mathrm{SVS}}(T,\tau)\widehat{C}_{\mathrm{Bayes}}(\tau)\,d\tau$",
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"exposure time $T$")
    ax.set_ylabel(r"visibility $K^2$")
    ax.legend(fontsize=8)


def compute_metrics(
    I_obs: np.ndarray,
    visibility: svs.VisibilityResult,
    uncertainty: svs.VisibilityUncertainty,
    Sigma_K: np.ndarray,
    reconstruction: svs.BayesianReconstructionResult,
    true_C: np.ndarray,
    prior_tau_c: float,
    calibrated_K: np.ndarray,
    pinv_C: np.ndarray,
) -> BenchmarkMetrics:
    c_residual = reconstruction.posterior_mean - true_C
    k_residual = visibility.Khat2 - reconstruction.forward_prediction
    cho = linalg.cho_factor(Sigma_K, lower=True, check_finite=False)
    weighted_residual = linalg.cho_solve(cho, k_residual, check_finite=False)
    chi_square = float(k_residual @ weighted_residual)
    calibrated_residual = visibility.Khat2 - calibrated_K
    return BenchmarkMetrics(
        mse_C=float(np.mean(c_residual**2)),
        visibility_chi_square=chi_square,
        visibility_reduced_chi_square=chi_square / visibility.Khat2.size,
        visibility_rmse=float(np.sqrt(np.mean(k_residual**2))),
        visibility_max_abs_residual=float(np.max(np.abs(k_residual))),
        calibrated_visibility_rmse=float(np.sqrt(np.mean(calibrated_residual**2))),
        calibrated_visibility_max_abs_residual=float(np.max(np.abs(calibrated_residual))),
        pinv_mse_C=float(np.mean((pinv_C - true_C) ** 2)),
        min_observed_intensity=float(np.min(I_obs)),
        observed_zero_count=int(np.sum(I_obs == 0.0)),
        stderr_min=float(np.min(uncertainty.stderr)),
        stderr_max=float(np.max(uncertainty.stderr)),
        prior_tau_c=float(prior_tau_c),
    )


def save_figures(
    output_dir: Path,
    t: np.ndarray,
    I_true: np.ndarray,
    I_obs: np.ndarray,
    correlation: svs.CorrelationModel,
    visibility: svs.VisibilityResult,
    uncertainty: svs.VisibilityUncertainty,
    reconstruction: svs.BayesianReconstructionResult,
    true_C: np.ndarray,
    pinv_C: np.ndarray,
    calibrated_K: np.ndarray,
    config: BenchmarkConfig,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    view = slice(0, min(10000, t.size))
    ax.plot(t[view], I_true[view], label=r"$I_{\mathrm{true}}(t)$", linewidth=1.1)
    ax.plot(t[view], I_obs[view], label=r"$I_{\mathrm{obs}}(t)$", linewidth=0.8, alpha=0.75)
    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"intensity $I(t)$")
    ax.legend()
    figure_paths["synthetic_time_series"] = str(output_dir / "synthetic_time_series.png")
    fig.savefig(figure_paths["synthetic_time_series"], dpi=200)
    plt.close(fig)

    max_lag = min(int(round(config.max_lag_time / config.dt)), I_true.size - 1)
    direct_tau, direct_C = estimate_direct_normalized_autocorrelation(
        I_obs,
        dt=config.dt,
        max_lag=max_lag,
    )
    direct_C = subtract_poisson_zero_lag(
        direct_C,
        I_obs,
        dt=config.dt,
        kappa=config.kappa,
    )
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.plot(direct_tau, correlation(direct_tau), color="black", label=r"target $C(\tau)$")
    ax.plot(direct_tau, direct_C, label=r"direct $C(\tau)$")
    ax.set_xlabel(r"lag time $\tau$")
    ax.set_ylabel(r"normalized autocorrelation $C(\tau)$")
    ax.legend()
    figure_paths["target_vs_empirical_autocorrelation"] = str(
        output_dir / "target_vs_empirical_autocorrelation.png"
    )
    fig.savefig(figure_paths["target_vs_empirical_autocorrelation"], dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    plot_measured_and_forward_visibility(
        visibility,
        uncertainty,
        reconstruction.forward_prediction,
        ax=ax,
    )
    figure_paths["measured_visibility"] = str(output_dir / "measured_visibility.png")
    fig.savefig(figure_paths["measured_visibility"], dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    ax_signal = axes[0, 0]
    ax_visibility = axes[0, 1]
    ax_c = axes[1, 0]
    ax_residual = axes[1, 1]
    for ax in axes.ravel():
        ax.set_box_aspect(1)

    ax_signal.plot(t[view], I_true[view], label=r"$I_{\mathrm{true}}(t)$", linewidth=1.1)
    ax_signal.plot(t[view], I_obs[view], label=r"$I_{\mathrm{obs}}(t)$", linewidth=0.8, alpha=0.75)
    ax_signal.set_xlabel(r"time $t$")
    ax_signal.set_ylabel(r"intensity $I(t)$")
    ax_signal.legend(fontsize=8)

    plot_measured_and_forward_visibility(
        visibility,
        uncertainty,
        reconstruction.forward_prediction,
        ax=ax_visibility,
    )
    ax_visibility.legend(fontsize=8)

    recon_max_lag = min(
        int(round(float(np.max(visibility.effective_T_values)) / config.dt)),
        I_obs.size - 1,
    )
    direct_tau, direct_C = estimate_direct_normalized_autocorrelation(
        I_obs,
        dt=config.dt,
        max_lag=recon_max_lag,
    )
    direct_C = subtract_poisson_zero_lag(
        direct_C,
        I_obs,
        dt=config.dt,
        kappa=config.kappa,
    )
    fft_tau, fft_C = svs.estimate_normalized_autocorrelation(
        I_obs,
        dt=config.dt,
        max_lag=recon_max_lag,
    )
    fft_C = subtract_poisson_zero_lag(
        fft_C,
        I_obs,
        dt=config.dt,
        kappa=config.kappa,
    )

    lower = reconstruction.posterior_mean - reconstruction.posterior_std
    upper = reconstruction.posterior_mean + reconstruction.posterior_std
    ax_c.plot(
        reconstruction.tau_grid,
        true_C,
        color="black",
        linewidth=2.0,
        label=r"ground truth $C(\tau)$",
    )
    if config.show_autocorr_fft:
        ax_c.plot(
            fft_tau,
            fft_C,
            color="cyan",
            linewidth=1.2,
            alpha=0.85,
            label=r"FFT $C(\tau)$",
        )
    if config.show_autocorr_direct:
        ax_c.plot(
            direct_tau,
            direct_C,
            color="blue",
            linewidth=1.4,
            alpha=0.85,
            label=r"direct $C(\tau)$",
        )
    if config.show_autocorr_pinv:
        ax_c.plot(
            reconstruction.tau_grid,
            pinv_C,
            color="purple",
            linewidth=1.5,
            label=r"$R_{\mathrm{SVS}}^{+}\widehat{K}^2$",
        )
    if config.show_prior_mean:
        ax_c.plot(
            reconstruction.tau_grid,
            reconstruction.C0,
            color="gray",
            linewidth=1.5,
            linestyle="--",
            label=r"prior mean",
        )
    if config.show_autocorr_bayes:
        ax_c.fill_between(
            reconstruction.tau_grid,
            lower,
            upper,
            color="red",
            alpha=0.25,
            label="_nolegend_",
        )
        ax_c.plot(
            reconstruction.tau_grid,
            reconstruction.posterior_mean,
            color="red",
            linewidth=2.0,
            label=r"Bayes inference",
        )
    ax_c.set_xlabel(r"delay time $\tau$")
    ax_c.set_ylabel(r"normalized autocorrelation $C(\tau)$")
    ax_c.set_xscale(config.autocorr_time_scale)
    ax_c.set_yscale(config.autocorr_scale)
    ax_c.set_xlim(config.autocorr_time_range)
    if config.autocorr_range is not None:
        ax_c.set_ylim(config.autocorr_range)
    ax_c.legend(fontsize=8, loc="upper right")

    ax_residual.axhline(0.0, color="black", linewidth=1.0)
    if config.show_autocorr_fft:
        ax_residual.plot(
            fft_tau,
            fft_C - correlation(fft_tau),
            color="cyan",
            linewidth=1.2,
            label=r"FFT $C(\tau)$",
        )
    if config.show_autocorr_direct:
        ax_residual.plot(
            direct_tau,
            direct_C - correlation(direct_tau),
            color="blue",
            linewidth=1.4,
            label=r"direct $C(\tau)$",
        )
    if config.show_autocorr_pinv:
        ax_residual.plot(
            reconstruction.tau_grid,
            pinv_C - correlation(reconstruction.tau_grid),
            color="purple",
            linewidth=1.5,
            label=r"$R_{\mathrm{SVS}}^{+}\widehat{K}^2$",
        )
    if config.show_prior_mean:
        ax_residual.plot(
            reconstruction.tau_grid,
            reconstruction.C0 - correlation(reconstruction.tau_grid),
            color="gray",
            linewidth=1.5,
            linestyle="--",
            label=r"prior mean",
        )
    if config.show_autocorr_bayes:
        ax_residual.plot(
            reconstruction.tau_grid,
            reconstruction.posterior_mean - correlation(reconstruction.tau_grid),
            color="red",
            linewidth=1.8,
            label=r"Bayes inference",
        )
    ax_residual.set_xlabel(r"delay time $\tau$")
    ax_residual.set_ylabel(r"autocorrelation residual")
    ax_residual.set_xscale(config.autocorr_time_scale)
    ax_residual.set_xlim(config.autocorr_time_range)
    ax_residual.set_ylim(-0.5, 0.5)
    ax_residual.legend(fontsize=8)

    figure_paths["reconstructed_c"] = str(output_dir / "reconstructed_c.png")
    fig.savefig(figure_paths["reconstructed_c"], dpi=200)
    plt.close(fig)

    residual = visibility.Khat2 - reconstruction.forward_prediction
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.errorbar(
        visibility.effective_T_values,
        residual,
        yerr=uncertainty.stderr,
        marker="s",
        capsize=3,
        linestyle="none",
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"exposure time $T$")
    ax.set_ylabel(r"$\hat{K}^2 - K^2_{\mathrm{pred}}$")
    figure_paths["visibility_residuals"] = str(output_dir / "visibility_residuals.png")
    fig.savefig(figure_paths["visibility_residuals"], dpi=200)
    plt.close(fig)

    return figure_paths


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    t, I_true, I_obs, metadata, correlation = generate_observed_series(config)
    if I_obs is None:
        raise RuntimeError("Expected Poisson-observed intensity.")

    visibility, uncertainty, Sigma_K, _ = estimate_visibility_and_covariance(I_obs, config)
    reconstruction, true_C, prior_tau_c = reconstruct_correlation(
        visibility,
        Sigma_K,
        I_obs,
        config,
    )
    pinv_C = estimate_pinv_correlation(
        visibility,
        reconstruction.tau_grid,
        rcond=config.pinv_rcond,
    )
    calibrated_K, _, _ = direct_visibility_calibration(I_obs, visibility, config)
    metrics = compute_metrics(
        I_obs,
        visibility,
        uncertainty,
        Sigma_K,
        reconstruction,
        true_C,
        prior_tau_c,
        calibrated_K,
        pinv_C,
    )
    output_dir = Path(config.output_folder)
    figures = save_figures(
        output_dir,
        t,
        I_true,
        I_obs,
        correlation,
        visibility,
        uncertainty,
        reconstruction,
        true_C,
        pinv_C,
        calibrated_K,
        config,
    )

    tau_log_steps = np.diff(np.log(reconstruction.tau_grid))
    if visibility.effective_T_values.size > 1 and config.T_spacing == "log":
        exposure_log_steps = np.diff(np.log(visibility.effective_T_values))
        exposure_log_width = float(np.mean(exposure_log_steps))
    else:
        exposure_log_width = None
    log_spacing = LogSpacingDiagnostics(
        exposure_log_width=exposure_log_width,
        tau_log_width=float(np.mean(tau_log_steps)),
    )

    result = BenchmarkResult(
        config=config,
        metrics=metrics,
        figures=figures,
        log_spacing=log_spacing,
    )
    json_path = output_dir / "benchmark_results.json"
    payload = asdict(result)
    payload["metadata"] = asdict(metadata)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return result


def main() -> None:
    config = parse_args()
    result = run_benchmark(config)
    print("SVS benchmark complete")
    print(f"output_folder: {config.output_folder}")
    if result.log_spacing.exposure_log_width is not None:
        print(f"log spacing width dlog(T): {result.log_spacing.exposure_log_width:.6g}")
    print(f"log spacing width dlog(tau): {result.log_spacing.tau_log_width:.6g}")
    print(f"MSE(C): {result.metrics.mse_C:.6g}")
    print(
        "visibility chi-square:",
        f"{result.metrics.visibility_chi_square:.6g}",
        f"(reduced {result.metrics.visibility_reduced_chi_square:.6g})",
    )


if __name__ == "__main__":
    main()
