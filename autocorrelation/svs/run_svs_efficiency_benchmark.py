"""Benchmark SVS Bayesian inversion efficiency versus direct autocorrelation.

The benchmark sweeps count-rate quality ``kappa`` and compares reconstruction
loss against a direct time-domain autocorrelation estimator using the same
observed intensity record.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import run_svs_benchmark as bench


@dataclass(frozen=True)
class EfficiencyConfig:
    output_folder: str = "svs_efficiency_output"
    kappa_min: float = 1e-2
    kappa_max: float = 1e2
    n_kappa: int = 9
    n_rand: int = 3
    seed0: int = 20
    loss_tau_min: float = 0.05
    loss_tau_max: float = 5.0
    example_seed: int = 20
    tune_sigma_C: bool = False
    sigma_C_min: float = 1e-3
    sigma_C_max: float = 1.0
    n_sigma_C: int = 9


@dataclass(frozen=True)
class EfficiencyRun:
    kappa: float
    seed: int
    sigma_C: float
    mse_bayes: float
    mse_direct: float
    visibility_chi_square: float
    visibility_reduced_chi_square: float
    visibility_rmse: float
    observed_zero_count: int
    empty_sample_rate: float
    min_observed_intensity: float
    runtime_seconds: float


@dataclass(frozen=True)
class EfficiencySummary:
    kappa: float
    sigma_C_mean: float
    sigma_C_std: float
    mse_bayes_mean: float
    mse_bayes_std: float
    mse_direct_mean: float
    mse_direct_std: float
    visibility_reduced_chi_square_mean: float
    visibility_reduced_chi_square_std: float
    runtime_seconds_mean: float
    runtime_seconds_std: float
    observed_zero_count_mean: float
    empty_sample_rate_mean: float
    empty_sample_rate_std: float


@dataclass(frozen=True)
class EfficiencyResult:
    efficiency_config: EfficiencyConfig
    benchmark_config: dict
    runs: list[EfficiencyRun]
    summary: list[EfficiencySummary]
    figures: dict[str, str]


def parse_args() -> tuple[EfficiencyConfig, bench.BenchmarkConfig]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-folder", default=EfficiencyConfig.output_folder)
    parser.add_argument("--kappa-min", type=float, default=EfficiencyConfig.kappa_min)
    parser.add_argument("--kappa-max", type=float, default=EfficiencyConfig.kappa_max)
    parser.add_argument("--n-kappa", type=int, default=EfficiencyConfig.n_kappa)
    parser.add_argument("--n-rand", type=int, default=EfficiencyConfig.n_rand)
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=None,
        help="deprecated alias for --n-rand",
    )
    parser.add_argument("--seed0", type=int, default=EfficiencyConfig.seed0)
    parser.add_argument("--loss-tau-min", type=float, default=EfficiencyConfig.loss_tau_min)
    parser.add_argument("--loss-tau-max", type=float, default=EfficiencyConfig.loss_tau_max)
    parser.add_argument("--example-seed", type=int, default=EfficiencyConfig.example_seed)
    parser.add_argument(
        "--tune-sigma-C",
        action=argparse.BooleanOptionalAction,
        default=EfficiencyConfig.tune_sigma_C,
        help="scan sigma_C and choose the value whose visibility reduced chi-square is closest to 1",
    )
    parser.add_argument("--sigma-C-min", type=float, default=EfficiencyConfig.sigma_C_min)
    parser.add_argument("--sigma-C-max", type=float, default=EfficiencyConfig.sigma_C_max)
    parser.add_argument("--n-sigma-C", type=int, default=EfficiencyConfig.n_sigma_C)

    parser.add_argument("--mean-I", type=float, default=bench.BenchmarkConfig.mean_I)
    parser.add_argument("--total-time", type=float, default=bench.BenchmarkConfig.total_time)
    parser.add_argument("--dt", type=float, default=bench.BenchmarkConfig.dt)
    parser.add_argument(
        "--correlation-model",
        choices=("exponential", "double_exponential", "damped_oscillation"),
        default=bench.BenchmarkConfig.correlation_model,
    )
    parser.add_argument("--beta", type=float, default=bench.BenchmarkConfig.beta)
    parser.add_argument("--tau-c", type=float, default=bench.BenchmarkConfig.tau_c)
    parser.add_argument("--f", type=float, default=bench.BenchmarkConfig.f)
    parser.add_argument("--tau1", type=float, default=bench.BenchmarkConfig.tau1)
    parser.add_argument("--tau2", type=float, default=bench.BenchmarkConfig.tau2)
    parser.add_argument("--period", type=float, default=bench.BenchmarkConfig.period)
    parser.add_argument("--T-min", type=float, default=bench.BenchmarkConfig.T_min)
    parser.add_argument("--T-max", type=float, default=bench.BenchmarkConfig.T_max)
    parser.add_argument("--N-T", type=int, default=bench.BenchmarkConfig.N_T)
    parser.add_argument(
        "--T-spacing",
        choices=("powers_of_two", "log"),
        default=bench.BenchmarkConfig.T_spacing,
    )
    parser.add_argument("--tau-grid-size", type=int, default=bench.BenchmarkConfig.tau_grid_size)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=bench.BenchmarkConfig.lambda_)
    parser.add_argument("--sigma-C", type=float, default=bench.BenchmarkConfig.sigma_C)
    parser.add_argument(
        "--prior-covariance-space",
        choices=("linear", "log"),
        default=bench.BenchmarkConfig.prior_covariance_space,
    )
    parser.add_argument(
        "--prior-mean",
        choices=("zero", "exponential", "direct_smooth"),
        default=bench.BenchmarkConfig.prior_mean,
    )
    parser.add_argument("--prior-smooth-sigma", type=float, default=bench.BenchmarkConfig.prior_smooth_sigma)
    parser.add_argument(
        "--low-tau-prior-anchor",
        action=argparse.BooleanOptionalAction,
        default=bench.BenchmarkConfig.low_tau_prior_anchor,
    )

    args = parser.parse_args()
    efficiency_config = EfficiencyConfig(
        output_folder=args.output_folder,
        kappa_min=args.kappa_min,
        kappa_max=args.kappa_max,
        n_kappa=args.n_kappa,
        n_rand=args.n_seeds if args.n_seeds is not None else args.n_rand,
        seed0=args.seed0,
        loss_tau_min=args.loss_tau_min,
        loss_tau_max=args.loss_tau_max,
        example_seed=args.example_seed,
        tune_sigma_C=args.tune_sigma_C,
        sigma_C_min=args.sigma_C_min,
        sigma_C_max=args.sigma_C_max,
        n_sigma_C=args.n_sigma_C,
    )
    benchmark_config = bench.BenchmarkConfig(
        mean_I=args.mean_I,
        total_time=args.total_time,
        dt=args.dt,
        correlation_model=args.correlation_model,
        beta=args.beta,
        tau_c=args.tau_c,
        f=args.f,
        tau1=args.tau1,
        tau2=args.tau2,
        period=args.period,
        kappa=bench.BenchmarkConfig.kappa,
        T_min=args.T_min,
        T_max=args.T_max,
        N_T=args.N_T,
        T_spacing=args.T_spacing,
        tau_grid_size=args.tau_grid_size,
        lambda_=args.lambda_,
        sigma_C=args.sigma_C,
        prior_covariance_space=args.prior_covariance_space,
        prior_mean=args.prior_mean,
        prior_smooth_sigma=args.prior_smooth_sigma,
        low_tau_prior_anchor=args.low_tau_prior_anchor,
        output_folder=args.output_folder,
    )
    return efficiency_config, benchmark_config


def kappa_values(config: EfficiencyConfig) -> np.ndarray:
    if config.n_kappa < 1:
        raise ValueError("n_kappa must be at least 1.")
    if config.kappa_min <= 0.0 or config.kappa_max <= 0.0:
        raise ValueError("kappa bounds must be positive.")
    return np.geomspace(config.kappa_min, config.kappa_max, config.n_kappa)


def sigma_C_values(config: EfficiencyConfig, default_sigma_C: float) -> np.ndarray:
    if not config.tune_sigma_C:
        return np.array([default_sigma_C], dtype=float)
    if config.n_sigma_C < 1:
        raise ValueError("n_sigma_C must be at least 1.")
    if config.sigma_C_min <= 0.0 or config.sigma_C_max <= 0.0:
        raise ValueError("sigma_C bounds must be positive.")
    return np.geomspace(config.sigma_C_min, config.sigma_C_max, config.n_sigma_C)


def visibility_reduced_chi_square(
    visibility: bench.svs.VisibilityResult,
    Sigma_K: np.ndarray,
    reconstruction: bench.svs.BayesianReconstructionResult,
) -> float:
    residual = visibility.Khat2 - reconstruction.forward_prediction
    cho = bench.linalg.cho_factor(Sigma_K, lower=True, check_finite=False)
    weighted_residual = bench.linalg.cho_solve(cho, residual, check_finite=False)
    return float(residual @ weighted_residual / visibility.Khat2.size)


def reconstruct_with_optional_sigma_C_tuning(
    visibility: bench.svs.VisibilityResult,
    Sigma_K: np.ndarray,
    I_obs: np.ndarray,
    base_config: bench.BenchmarkConfig,
    efficiency_config: EfficiencyConfig,
) -> tuple[bench.svs.BayesianReconstructionResult, np.ndarray, float, float, list[dict[str, float]]]:
    candidates = sigma_C_values(efficiency_config, base_config.sigma_C)
    best = None
    diagnostics: list[dict[str, float]] = []
    for sigma_C in candidates:
        config = bench.BenchmarkConfig(
            **{
                **asdict(base_config),
                "sigma_C": float(sigma_C),
            }
        )
        reconstruction, true_C, prior_tau_c = bench.reconstruct_correlation(
            visibility,
            Sigma_K,
            I_obs,
            config,
        )
        reduced_chi_square = visibility_reduced_chi_square(visibility, Sigma_K, reconstruction)
        diagnostics.append(
            {
                "sigma_C": float(sigma_C),
                "visibility_reduced_chi_square": reduced_chi_square,
            }
        )
        score = abs(np.log(max(reduced_chi_square, np.finfo(float).tiny)))
        if best is None or score < best[0]:
            best = (score, float(sigma_C), reconstruction, true_C, prior_tau_c)
    assert best is not None
    _, selected_sigma_C, reconstruction, true_C, prior_tau_c = best
    return reconstruction, true_C, float(prior_tau_c), selected_sigma_C, diagnostics


def direct_mse_on_reconstruction_grid(
    I_obs: np.ndarray,
    benchmark_config: bench.BenchmarkConfig,
    reconstruction_tau: np.ndarray,
    correlation: bench.svs.CorrelationModel,
    *,
    loss_tau_min: float,
    loss_tau_max: float,
) -> tuple[float, np.ndarray]:
    max_lag = min(
        int(round(float(np.max(reconstruction_tau)) / benchmark_config.dt)),
        I_obs.size - 1,
    )
    direct_tau, direct_C = bench.estimate_direct_normalized_autocorrelation(
        I_obs,
        dt=benchmark_config.dt,
        max_lag=max_lag,
    )
    direct_C = bench.subtract_poisson_zero_lag(
        direct_C,
        I_obs,
        dt=benchmark_config.dt,
        kappa=benchmark_config.kappa,
    )
    direct_on_grid = np.interp(reconstruction_tau, direct_tau, direct_C)
    true_on_grid = correlation(reconstruction_tau)
    mask = (reconstruction_tau >= loss_tau_min) & (reconstruction_tau <= loss_tau_max)
    return float(np.mean((direct_on_grid[mask] - true_on_grid[mask]) ** 2)), direct_on_grid


def run_one(
    benchmark_config: bench.BenchmarkConfig,
    efficiency_config: EfficiencyConfig,
    *,
    kappa: float,
    seed: int,
    loss_tau_min: float,
    loss_tau_max: float,
) -> tuple[EfficiencyRun, dict[str, np.ndarray]]:
    config = bench.BenchmarkConfig(
        **{
            **asdict(benchmark_config),
            "kappa": float(kappa),
            "seed": int(seed),
        }
    )
    start = time.perf_counter()
    t, I_true, I_obs, metadata, correlation = bench.generate_observed_series(config)
    visibility, uncertainty, Sigma_K, _ = bench.estimate_visibility_and_covariance(I_obs, config)
    reconstruction, true_C, prior_tau_c, selected_sigma_C, sigma_C_diagnostics = (
        reconstruct_with_optional_sigma_C_tuning(
            visibility,
            Sigma_K,
            I_obs,
            config,
            efficiency_config,
        )
    )
    selected_config = bench.BenchmarkConfig(
        **{
            **asdict(config),
            "sigma_C": selected_sigma_C,
        }
    )
    pinv_C = bench.estimate_pinv_correlation(
        visibility,
        reconstruction.tau_grid,
        rcond=selected_config.pinv_rcond,
    )
    calibrated_K, _, _ = bench.direct_visibility_calibration(I_obs, visibility, selected_config)
    metrics = bench.compute_metrics(
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
    mse_direct, direct_on_grid = direct_mse_on_reconstruction_grid(
        I_obs,
        selected_config,
        reconstruction.tau_grid,
        correlation,
        loss_tau_min=loss_tau_min,
        loss_tau_max=loss_tau_max,
    )
    mask = (reconstruction.tau_grid >= loss_tau_min) & (reconstruction.tau_grid <= loss_tau_max)
    mse_bayes = float(np.mean((reconstruction.posterior_mean[mask] - true_C[mask]) ** 2))
    runtime = time.perf_counter() - start
    run = EfficiencyRun(
        kappa=float(kappa),
        seed=int(seed),
        sigma_C=float(selected_sigma_C),
        mse_bayes=mse_bayes,
        mse_direct=mse_direct,
        visibility_chi_square=metrics.visibility_chi_square,
        visibility_reduced_chi_square=metrics.visibility_reduced_chi_square,
        visibility_rmse=metrics.visibility_rmse,
        observed_zero_count=metrics.observed_zero_count,
        empty_sample_rate=float(metrics.observed_zero_count / I_obs.size),
        min_observed_intensity=metrics.min_observed_intensity,
        runtime_seconds=float(runtime),
    )
    arrays = {
        "tau": reconstruction.tau_grid,
        "true_C": true_C,
        "bayes_C": reconstruction.posterior_mean,
        "direct_C": direct_on_grid,
        "posterior_std": reconstruction.posterior_std,
        "T": visibility.effective_T_values,
        "Khat2": visibility.Khat2,
        "K_forward": reconstruction.forward_prediction,
        "sigma_C_diagnostics": np.array(
            [
                (item["sigma_C"], item["visibility_reduced_chi_square"])
                for item in sigma_C_diagnostics
            ],
            dtype=float,
        ),
    }
    return run, arrays


def summarize_runs(runs: list[EfficiencyRun]) -> list[EfficiencySummary]:
    kappas = sorted({run.kappa for run in runs})
    summary: list[EfficiencySummary] = []
    for kappa in kappas:
        group = [run for run in runs if run.kappa == kappa]
        bayes = np.array([run.mse_bayes for run in group], dtype=float)
        direct = np.array([run.mse_direct for run in group], dtype=float)
        sigma_C = np.array([run.sigma_C for run in group], dtype=float)
        chi = np.array([run.visibility_reduced_chi_square for run in group], dtype=float)
        runtime = np.array([run.runtime_seconds for run in group], dtype=float)
        zeros = np.array([run.observed_zero_count for run in group], dtype=float)
        empty_rates = np.array([run.empty_sample_rate for run in group], dtype=float)
        summary.append(
            EfficiencySummary(
                kappa=float(kappa),
                sigma_C_mean=float(np.mean(sigma_C)),
                sigma_C_std=float(np.std(sigma_C, ddof=1)) if sigma_C.size > 1 else 0.0,
                mse_bayes_mean=float(np.mean(bayes)),
                mse_bayes_std=float(np.std(bayes, ddof=1)) if bayes.size > 1 else 0.0,
                mse_direct_mean=float(np.mean(direct)),
                mse_direct_std=float(np.std(direct, ddof=1)) if direct.size > 1 else 0.0,
                visibility_reduced_chi_square_mean=float(np.mean(chi)),
                visibility_reduced_chi_square_std=float(np.std(chi, ddof=1)) if chi.size > 1 else 0.0,
                runtime_seconds_mean=float(np.mean(runtime)),
                runtime_seconds_std=float(np.std(runtime, ddof=1)) if runtime.size > 1 else 0.0,
                observed_zero_count_mean=float(np.mean(zeros)),
                empty_sample_rate_mean=float(np.mean(empty_rates)),
                empty_sample_rate_std=float(np.std(empty_rates, ddof=1)) if empty_rates.size > 1 else 0.0,
            )
        )
    return summary


def save_plots(
    output_dir: Path,
    summary: list[EfficiencySummary],
    examples: dict[float, dict[str, np.ndarray]],
    *,
    loss_tau_min: float,
    loss_tau_max: float,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, str] = {}
    kappas = np.array([item.kappa for item in summary], dtype=float)
    bayes_mean = np.array([item.mse_bayes_mean for item in summary], dtype=float)
    bayes_std = np.array([item.mse_bayes_std for item in summary], dtype=float)
    direct_mean = np.array([item.mse_direct_mean for item in summary], dtype=float)
    direct_std = np.array([item.mse_direct_std for item in summary], dtype=float)

    fig, ax = plt.subplots(figsize=(5.0, 5.0), constrained_layout=True)
    ax.errorbar(kappas, bayes_mean, yerr=bayes_std, marker="o", capsize=3, label="Bayes/GPR")
    ax.errorbar(kappas, direct_mean, yerr=direct_std, marker="s", capsize=3, label="direct autocorr")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"count-rate factor $\kappa$")
    ax.set_ylabel("MSE")
    empty_rate = np.array([item.empty_sample_rate_mean for item in summary], dtype=float)
    empty_rate_std = np.array([item.empty_sample_rate_std for item in summary], dtype=float)
    ax2 = ax.twinx()
    ax2.errorbar(
        kappas,
        empty_rate,
        yerr=empty_rate_std,
        color="tab:green",
        marker="^",
        capsize=3,
        linestyle="--",
        label="empty sample rate",
    )
    ax2.set_ylabel("empty sample rate")
    ax2.set_ylim(-0.02, 1.02)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")
    figures["mse_vs_kappa"] = str(output_dir / "mse_vs_kappa.png")
    fig.savefig(figures["mse_vs_kappa"], dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    chi = np.array([item.visibility_reduced_chi_square_mean for item in summary], dtype=float)
    chi_std = np.array([item.visibility_reduced_chi_square_std for item in summary], dtype=float)
    ax.errorbar(kappas, chi, yerr=chi_std, marker="s", capsize=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.1, 10.0)
    ax.set_xlabel(r"count-rate factor $\kappa$")
    ax.set_ylabel(r"visibility reduced $\chi^2$")
    figures["chi_square_vs_kappa"] = str(output_dir / "chi_square_vs_kappa.png")
    fig.savefig(figures["chi_square_vs_kappa"], dpi=200)
    plt.close(fig)

    sigma_C_mean = np.array([item.sigma_C_mean for item in summary], dtype=float)
    sigma_C_std = np.array([item.sigma_C_std for item in summary], dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    ax.errorbar(kappas, sigma_C_mean, yerr=sigma_C_std, marker="o", capsize=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"count-rate factor $\kappa$")
    ax.set_ylabel(r"selected prior scale $\sigma_C$")
    figures["sigma_C_vs_kappa"] = str(output_dir / "sigma_C_vs_kappa.png")
    fig.savefig(figures["sigma_C_vs_kappa"], dpi=200)
    plt.close(fig)

    n_examples = len(examples)
    if n_examples:
        fig, axes = plt.subplots(
            1,
            n_examples,
            figsize=(4.2 * n_examples, 3.8),
            constrained_layout=True,
            squeeze=False,
        )
        for ax, (kappa, arrays) in zip(axes[0], sorted(examples.items())):
            tau = arrays["tau"]
            ax.plot(tau, arrays["true_C"], color="black", linewidth=2.0, label="true")
            ax.plot(tau, arrays["bayes_C"], color="red", linewidth=1.8, label="Bayes")
            ax.plot(tau, arrays["direct_C"], color="blue", linewidth=1.2, label="direct")
            ax.fill_between(
                tau,
                arrays["bayes_C"] - arrays["posterior_std"],
                arrays["bayes_C"] + arrays["posterior_std"],
                color="red",
                alpha=0.2,
            )
            ax.set_xscale("log")
            ax.set_xlim(loss_tau_min, loss_tau_max)
            ax.set_ylim(-0.2, 1.2)
            ax.set_title(rf"$\kappa={kappa:g}$")
            ax.set_xlabel(r"delay time $\tau$")
            ax.set_ylabel(r"$C(\tau)$")
        axes[0, 0].legend(fontsize=8)
        figures["example_reconstructions_by_kappa"] = str(
            output_dir / "example_reconstructions_by_kappa.png"
        )
        fig.savefig(figures["example_reconstructions_by_kappa"], dpi=200)
        plt.close(fig)
    return figures


def run_efficiency_benchmark(
    efficiency_config: EfficiencyConfig,
    benchmark_config: bench.BenchmarkConfig,
) -> EfficiencyResult:
    output_dir = Path(efficiency_config.output_folder)
    kappas = kappa_values(efficiency_config)
    seeds = [efficiency_config.seed0 + i for i in range(efficiency_config.n_rand)]
    example_kappas = {float(kappas[0]), float(kappas[len(kappas) // 2]), float(kappas[-1])}
    runs: list[EfficiencyRun] = []
    examples: dict[float, dict[str, np.ndarray]] = {}

    for kappa in kappas:
        for seed in seeds:
            run, arrays = run_one(
                benchmark_config,
                efficiency_config,
                kappa=float(kappa),
                seed=seed,
                loss_tau_min=efficiency_config.loss_tau_min,
                loss_tau_max=efficiency_config.loss_tau_max,
            )
            runs.append(run)
            if seed == efficiency_config.example_seed and float(kappa) in example_kappas:
                examples[float(kappa)] = arrays
            print(
                f"kappa={kappa:.4g} seed={seed} "
                f"sigma_C={run.sigma_C:.4g} "
                f"MSE_Bayes={run.mse_bayes:.4g} MSE_direct={run.mse_direct:.4g}"
            )

    summary = summarize_runs(runs)
    figures = save_plots(
        output_dir,
        summary,
        examples,
        loss_tau_min=efficiency_config.loss_tau_min,
        loss_tau_max=efficiency_config.loss_tau_max,
    )
    result = EfficiencyResult(
        efficiency_config=efficiency_config,
        benchmark_config=asdict(benchmark_config),
        runs=runs,
        summary=summary,
        figures=figures,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "efficiency_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2)
    return result


def main() -> None:
    efficiency_config, benchmark_config = parse_args()
    result = run_efficiency_benchmark(efficiency_config, benchmark_config)
    print("SVS efficiency benchmark complete")
    print(f"output_folder: {efficiency_config.output_folder}")
    print("kappa values:", ", ".join(f"{item.kappa:.4g}" for item in result.summary))
    print("figures:")
    for name, path in result.figures.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
