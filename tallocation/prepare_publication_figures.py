"""Prepare publication-style figures for the USANS time-allocation study."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.linalg import solve

from usans_lib import (
    EPS,
    base_relative_error_and_scores,
    compute_a_coefficients,
    correlation_posterior_error_and_scores,
    estimate_optimal_log_kernel_width,
    inverse_intensity_power_allocation,
    make_lorentzian_ground_truth,
    make_log_matern_correlation_operator,
    make_log_matern_covariance,
    make_q_grid,
    make_reconstruction_operator,
    make_resolution_model_grid,
    make_second_difference_matrix,
    marginal_scores,
    optimal_allocation_from_a,
    psf,
    uniform_allocation,
)


def build_publication_state():
    """Build the benchmark state needed for the publication figure."""
    Q_min = 1.0e-5
    Q_max = 1.0e-3
    N = 200
    Q_model_max = 2.0e-1
    N_model_tail = 400
    I0 = 1.0e6
    xi_lorentzian = 25000.0
    background = 20.0
    sigma_x = 1.0e-6
    sigma_y = 1.0e-3
    alpha_reg = 1.0e-3
    ridge = 1.0e-8
    eta_H = 0.05
    target_uniform_counts_per_point = 100.0

    Q = make_q_grid(Q_min, Q_max, N)
    I_true = make_lorentzian_ground_truth(Q, I0, xi_lorentzian, background)

    Q_model = make_resolution_model_grid(Q, Q_model_max, N_model_tail)
    I_true_model = make_lorentzian_ground_truth(Q_model, I0, xi_lorentzian, background)

    kernel_info = estimate_optimal_log_kernel_width(Q, I_true, n_counts=N)
    ell_log = kernel_info["ell_log"]

    K = make_log_matern_correlation_operator(Q, ell_log)
    T_tot_fig1 = target_uniform_counts_per_point * N / np.mean(I_true)
    t_uniform_demo = uniform_allocation(N, T_tot_fig1)
    kernel_variance, kernel_chi2, kernel_reduced_chi2 = _estimate_kernel_variance_chi_square(
        I_true, K, t_uniform_demo
    )
    K_cov = make_log_matern_covariance(Q, ell_log, variance=kernel_variance)
    K_inv = solve(K_cov, np.eye(N), assume_a="sym")

    R_full = psf(sigma_y, Q, Q_model, sigma_x=sigma_x)
    low_mask = Q_model <= Q_max
    high_mask = ~low_mask
    R_L = R_full[:, low_mask]
    R_H = R_full[:, high_mask]
    I_L = I_true_model[low_mask]
    I_H = I_true_model[high_mask]
    I_high_known = R_H @ I_H
    I_smeared = R_L @ I_L + I_high_known

    L_L = make_second_difference_matrix(np.count_nonzero(low_mask))
    A_L = make_reconstruction_operator(R_L, L_L, alpha_reg, ridge=ridge)
    v_obs = 1.0 / np.maximum(I_smeared, EPS)
    a_bonse = compute_a_coefficients(A_L, v_obs)
    sigma_H_diag = (eta_H * I_H) ** 2
    Sigma_bH = (R_H * sigma_H_diag[None, :]) @ R_H.T
    E_floor_bonse = float(np.trace(A_L @ Sigma_bH @ A_L.T))

    cases = {
        "Base case": {"kind": "base", "I_ref": I_true},
        "Correlation-aware": {"kind": "corr", "I_ref": I_true},
        "Bonse-Hart silt smeared": {
            "kind": "a_floor",
            "I_ref": I_smeared,
            "a": a_bonse,
            "E_floor": E_floor_bonse,
        },
    }
    cases["Base case"]["score_func"] = lambda t: base_relative_error_and_scores(I_true, t)
    cases["Correlation-aware"]["score_func"] = (
        lambda t: correlation_posterior_error_and_scores(I_true, K_inv, t)
    )

    allocations = {}
    for case_name in ["Base case", "Correlation-aware", "Bonse-Hart silt smeared"]:
        case = cases[case_name]
        I_ref = case["I_ref"]
        if case["kind"] == "base":
            t_opt = optimal_allocation_from_a(1.0 / np.maximum(I_ref, EPS), T_tot_fig1)
            E_opt, g_opt = case["score_func"](t_opt)
        elif case["kind"] == "corr":
            t_initial = inverse_intensity_power_allocation(I_ref, 0.5, T_tot_fig1)
            t_opt, E_opt, g_opt = _optimize_allocation_by_scores(
                case["score_func"], t_initial, T_tot_fig1
            )
        else:
            t_opt = optimal_allocation_from_a(np.maximum(case["a"], EPS), T_tot_fig1)
            E_opt = float(np.sum(case["a"] / np.maximum(t_opt, EPS)) + case["E_floor"])
            g_opt = marginal_scores(case["a"], t_opt)

        allocations[case_name] = t_opt
        case["t_opt_demo"] = t_opt
        case["E_opt_demo"] = E_opt
        case["g_opt_demo"] = g_opt

    T_values = np.logspace(-2, 2, 80)
    power_values = np.linspace(0.0, 1.0, 201)
    power_rows = []
    optimal_rows = []
    summary_rows = []
    for case_name in ["Base case", "Correlation-aware", "Bonse-Hart silt smeared"]:
        case = cases[case_name]
        I_ref = case["I_ref"]
        optimal_rows.append({"case": case_name, "E_opt": case["E_opt_demo"]})
        for power_n in power_values:
            t_power = inverse_intensity_power_allocation(I_ref, power_n, T_tot_fig1)
            if case["kind"] == "a_floor":
                E_value = float(np.sum(case["a"] / np.maximum(t_power, EPS)) + case["E_floor"])
            else:
                E_value, _ = case["score_func"](t_power)
            power_rows.append({"case": case_name, "power_n": power_n, "E": E_value})

        for T_tot in T_values:
            if case["kind"] == "base":
                t_opt_strategy = optimal_allocation_from_a(1.0 / np.maximum(I_ref, EPS), T_tot)
            elif case["kind"] == "corr":
                t_opt_strategy = T_tot * case["t_opt_demo"] / np.sum(case["t_opt_demo"])
            else:
                t_opt_strategy = T_tot * case["t_opt_demo"] / np.sum(case["t_opt_demo"])

            strategy_allocations = {
                "Uniform": uniform_allocation(N, T_tot),
                "Inverse intensity": inverse_intensity_power_allocation(I_ref, 1.0, T_tot),
                "Optimal": t_opt_strategy,
            }
            for strategy, t in strategy_allocations.items():
                if case["kind"] == "a_floor":
                    E_value = float(np.sum(case["a"] / np.maximum(t, EPS)) + case["E_floor"])
                else:
                    E_value, _ = case["score_func"](t)
                summary_rows.append(
                    {"case": case_name, "strategy": strategy, "T_tot": T_tot, "E": E_value}
                )

    return {
        "Q": Q,
        "I_true": I_true,
        "I_smeared": I_smeared,
        "allocations": allocations,
        "cases": cases,
        "power_rows": power_rows,
        "optimal_rows": optimal_rows,
        "summary_rows": summary_rows,
        "T_tot_fig1": T_tot_fig1,
        "kernel_info": kernel_info,
    }


def _plot_publication_figure(state, output_path):
    """Create the three-panel publication figure."""
    Q = state["Q"]
    I_true = state["I_true"]
    allocations = state["allocations"]
    cases = state["cases"]
    power_rows = state["power_rows"]
    optimal_rows = state["optimal_rows"]
    T_tot_fig1 = state["T_tot_fig1"]

    fig = plt.figure(figsize=(14.0, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    _plot_power_panel(ax_a, state, power_rows, optimal_rows, allocations, Q, I_true)
    _plot_error_band_absolute(ax_b, Q, cases["Base case"], T_tot_fig1)
    _plot_error_band_relative(ax_c, Q, I_true, cases["Base case"], T_tot_fig1)

    ax_a.text(0.02, 0.98, "(a)", transform=ax_a.transAxes, ha="left", va="top", fontsize=14, fontweight="bold")
    ax_b.text(0.02, 0.98, "(b)", transform=ax_b.transAxes, ha="left", va="top", fontsize=14, fontweight="bold")
    ax_c.text(0.02, 0.98, "(c)", transform=ax_c.transAxes, ha="left", va="top", fontsize=14, fontweight="bold")

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_power_panel(ax, state, power_rows, optimal_rows, allocations, Q, I_true):
    colors = {"Base case": "black", "Correlation-aware": "blue", "Bonse-Hart silt smeared": "red"}
    case_name = "Base case"
    sub = [row for row in power_rows if row["case"] == case_name]
    ax.plot(
        [row["power_n"] for row in sub],
        [row["E"] for row in sub],
        color=colors[case_name],
        lw=2.0,
        label=case_name,
    )
    opt = next(row for row in optimal_rows if row["case"] == case_name)
    ax.axhline(opt["E_opt"], color=colors[case_name], lw=1.5, ls="--", alpha=0.8)

    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$\mathcal{E}_0$")
    # ax.set_yscale("log")
    ax.set_ylim([0.08,0.12])
    ax.set_box_aspect(1)
    ax.grid(True, alpha=0.25)
    # ax.legend(frameon=False, loc="upper right", fontsize=10)

    # inset = inset_axes(
    #     ax,
    #     width="46%",
    #     height="46%",
    #     loc="lower left",
    #     borderpad=0.9,
    #     bbox_to_anchor=(0.08, 0.20, 1, 1),
    #     bbox_transform=ax.transAxes,
    # )
    # t_n0 = inverse_intensity_power_allocation(I_true, 0.0, state["T_tot_fig1"])
    # t_n05 = inverse_intensity_power_allocation(I_true, 0.5, state["T_tot_fig1"])
    # t_n1 = inverse_intensity_power_allocation(I_true, 1.0, state["T_tot_fig1"])
    # inset.plot(Q, t_n0, color="black", lw=1.6, label=r"$n=0$")
    # inset.plot(Q, t_n05, color="blue", lw=1.6, label=r"$n=0.5$")
    # inset.plot(Q, t_n1, color="red", lw=1.6, label=r"$n=1$")
    # inset.set_xscale("log")
    # inset.set_yscale("log")
    # inset.set_xlabel(r"$Q$ ($\mathrm{\AA}^{-1}$)", fontsize=10)
    # inset.set_ylabel(r"$t_i$ (s)", fontsize=10)
    # inset.tick_params(labelsize=8)
    # inset.grid(True, which="both", alpha=0.15)
    # inset.legend(frameon=False, fontsize=7, loc="upper right")


def _plot_error_band_absolute(ax, Q, case, T_tot):
    rng = np.random.default_rng(12345)
    I_ref = case["I_ref"]
    strategy_specs = [
        ("n=0", 0.0, "black"),
        (r"$n=0.5$", 0.5, "red"),
        (r"$n=1$", 1.0, "blue"),
    ]
    shift_factors = [1.0, 8.0, 64.0]

    for (label, power, color), shift in zip(strategy_specs, shift_factors):
        t = inverse_intensity_power_allocation(I_ref, float(power), T_tot)
        mean_counts = np.maximum(I_ref * t, EPS)
        sampled_counts = rng.poisson(mean_counts)
        sampled_intensity = sampled_counts / np.maximum(t, EPS)
        sigma_intensity = np.sqrt(np.maximum(I_ref, EPS) / np.maximum(t, EPS))

        y_true = I_ref * shift
        y_low = np.maximum((I_ref - sigma_intensity) * shift, EPS)
        y_high = (I_ref + sigma_intensity) * shift
        y_sample = np.maximum(sampled_intensity * shift, EPS)

        ax.fill_between(Q, y_low, y_high, color=color, alpha=0.18, linewidth=0)
        ax.plot(Q, y_true, color="black", lw=1.0, alpha=0.75)
        ax.scatter(Q, y_sample, s=8, color=color, alpha=0.75, edgecolors="none")
        ax.text(Q[0] * 1.07, y_true[0] * 1.2, label, color=color, fontsize=12, va="bottom")

    ax.set_ylim(np.min(I_ref) * shift_factors[0] / 10.0, np.max(I_ref) * shift_factors[-1] * 10.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$Q$ ($\mathrm{\AA}^{-1}$)")
    ax.set_ylabel(r"$I(Q)$ ($\mathrm{cm}^{-1}$)")
    ax.set_box_aspect(1)
    ax.grid(True, which="both", alpha=0.25)
def _plot_error_band_relative(ax, Q, I_gt, case, T_tot):
    I_ref = case["I_ref"]
    strategy_specs = [
        ("n=0", 0.0, "black"),
        ("n=1", 1.0, "blue"),
        ("Optimal", "optimal", "red"),
    ]
    shift_factors = [1.0, 8.0, 64.0]
    relative_lows = []
    relative_highs = []

    for (_, power, color), shift in zip(strategy_specs, shift_factors):
        if power == "optimal":
            t = case["t_opt_demo"]
        else:
            t = inverse_intensity_power_allocation(I_ref, float(power), T_tot)
        sigma_intensity = np.sqrt(np.maximum(I_ref, EPS) / np.maximum(t, EPS))
        rel_true = I_ref / np.maximum(I_gt, EPS)
        rel_low = np.maximum(I_ref - sigma_intensity, EPS) / np.maximum(I_gt, EPS)
        rel_high = (I_ref + sigma_intensity) / np.maximum(I_gt, EPS)
        relative_lows.append(np.min(rel_low))
        relative_highs.append(np.max(rel_high))
        ax.fill_between(Q, rel_low, rel_high, color=color, alpha=0.18, linewidth=0)
        ax.plot(Q, rel_true, color="black", lw=1.0, alpha=0.75)

    rel_min = min(relative_lows)
    rel_max = max(relative_highs)
    rel_pad = 0.08 * (rel_max - rel_min)
    ax.axhline(1.0, color="0.35", lw=1.0, ls="--", alpha=0.8)
    ax.set_ylim(max(0.0, rel_min - rel_pad), rel_max + rel_pad)
    ax.set_xscale("log")
    ax.set_xlabel(r"$Q$ ($\mathrm{\AA}^{-1}$)")
    ax.set_ylabel(r"$I/I_{\mathrm{gt}}$")
    ax.set_box_aspect(1)
    ax.grid(True, which="both", alpha=0.25)

def _optimize_allocation_by_scores(score_func, t_initial, T_tot, p=0.5, max_iter=120, tol=1.0e-3):
    t = np.asarray(t_initial, dtype=float).reshape(-1)
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


def _estimate_kernel_variance_chi_square(I_true, K, t_reference):
    """Estimate a covariance amplitude from a chi-square style residual test."""
    I_true = np.asarray(I_true, dtype=float).reshape(-1)
    t_reference = np.asarray(t_reference, dtype=float).reshape(-1)
    smooth = K @ I_true
    rel_resid = (I_true - smooth) / np.maximum(I_true, EPS)
    sigma2_rel = 1.0 / np.maximum(I_true * t_reference, EPS)
    chi2 = float(np.sum(rel_resid**2 / np.maximum(sigma2_rel, EPS)))
    dof = max(I_true.size - 1, 1)
    reduced_chi2 = chi2 / dof
    kernel_variance = float(reduced_chi2 * np.mean(sigma2_rel))
    return kernel_variance, chi2, reduced_chi2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-folder", default="publication_figures")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    state = build_publication_state()
    _plot_publication_figure(state, output_dir / "publication_figure1.png")
    print(f"Wrote {output_dir / 'publication_figure1.png'}")


if __name__ == "__main__":
    main()
