"""Run the USANS time-allocation benchmark."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.linalg import solve

from usans_lib import (
    EPS,
    base_relative_error_and_scores,
    compute_a_coefficients,
    correlation_posterior_error_and_scores,
    estimate_optimal_log_kernel_width,
    inverse_intensity_power_allocation,
    inverse_intensity_allocation,
    make_lorentzian_ground_truth,
    make_log_matern_correlation_operator,
    make_log_matern_covariance,
    make_q_grid,
    make_reconstruction_operator,
    make_second_difference_matrix,
    make_resolution_model_grid,
    marginal_scores,
    optimal_allocation_from_a,
    plot_errorbar_bands,
    plot_marginal_scores,
    plot_figure1,
    plot_power_scan,
    psf,
    print_total_count_diagnostics,
    uniform_allocation,
)


def run_benchmark(output_folder="usans_benchmark_output", show_correlation_aware=False):
    """Run the benchmark and write figures plus CSV output."""
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

    Q = make_q_grid(Q_min, Q_max, N)
    I_true = make_lorentzian_ground_truth(Q, I0, xi_lorentzian, background)
    Q_model = make_resolution_model_grid(Q, Q_model_max, N_model_tail)
    I_true_model = make_lorentzian_ground_truth(Q_model, I0, xi_lorentzian, background)

    kernel_info = estimate_optimal_log_kernel_width(Q, I_true, n_counts=N)
    ell_log_multiplier = 1.0
    ell_log = ell_log_multiplier * kernel_info["ell_log"]
    kernel_family = "matern_log"
    K = make_log_matern_correlation_operator(Q, ell_log)
    target_uniform_counts_per_point = 100.0
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

    figure1_case_order = ["Base case", "Correlation-aware"]
    figure2_case_order = ["Base case", "Correlation-aware"]
    figure3_case_order = ["Base case"]

    T_tot_figure3_by_case = {
        case_name: target_uniform_counts_per_point * N / np.mean(cases[case_name]["I_ref"])
        for case_name in figure3_case_order
    }

    print("Demonstration count scale")
    print(f"  target uniform mean counts per point: {target_uniform_counts_per_point:.6e}")
    print(f"  T_tot_demo: {T_tot_fig1:.16e}")
    print(
        "  uniform mean counts per point: "
        f"{np.mean(I_true * uniform_allocation(N, T_tot_fig1)):.16e}"
    )
    print(f"  kernel variance estimate: {kernel_variance:.16e}")
    print(f"  kernel chi2: {kernel_chi2:.16e}")
    print(f"  kernel reduced chi2: {kernel_reduced_chi2:.16e}")
    print(f"  kernel family: {kernel_family}")
    print("Figure 3 count scales")
    for case_name in figure3_case_order:
        case = cases[case_name]
        T_case = T_tot_figure3_by_case[case_name]
        mean_counts = np.mean(case["I_ref"] * uniform_allocation(N, T_case))
        print(f"  {case_name}: T_tot={T_case:.16e}, n=0 mean counts={mean_counts:.16e}")

    allocations = {}
    score_profiles = {}
    for case_name in figure1_case_order:
        case = cases[case_name]
        I_ref = case["I_ref"]
        if case["kind"] == "base":
            t_opt = optimal_allocation_from_a(1.0 / np.maximum(I_ref, EPS), T_tot_fig1)
            E_opt, g_opt = case["score_func"](t_opt)
        else:
            t_initial = inverse_intensity_power_allocation(I_ref, 0.5, T_tot_fig1)
            t_opt, E_opt, g_opt = _optimize_allocation_by_scores(case["score_func"], t_initial, T_tot_fig1)
        allocations[case_name] = t_opt
        case["t_opt_demo"] = t_opt
        case["E_opt_demo"] = E_opt
        case["g_opt_demo"] = g_opt

    for case_name in figure3_case_order:
        case = cases[case_name]
        T_case = T_tot_figure3_by_case[case_name]
        I_ref = case["I_ref"]
        t_opt = optimal_allocation_from_a(1.0 / np.maximum(I_ref, EPS), T_case)
        E_opt, g_opt = case["score_func"](t_opt)
        case["t_opt_figure3"] = t_opt
        case["E_opt_figure3"] = E_opt
        case["g_opt_figure3"] = g_opt

    for case_name, t_opt in allocations.items():
        E_opt = cases[case_name]["E_opt_demo"]
        g = cases[case_name]["g_opt_demo"]
        score_profiles[case_name] = g
        print(case_name)
        print("  model: Lorentzian density-fluctuation test")
        print(f"  Q range: {Q_min:.3e} to {Q_max:.3e} A^-1")
        print(f"  xi: {xi_lorentzian:.6e} A")
        print(f"  sigma_x: {sigma_x:.6e} A^-1")
        print(f"  sigma_y: {sigma_y:.6e} A^-1")
        print(f"  sigma_x / sigma_y: {sigma_x / sigma_y:.6e}")
        print(f"  N_Q: {N}")
        print(f"  N_Q_model_resolution: {Q_model.size}")
        print(f"  N_Q_model_low_unknown: {np.count_nonzero(low_mask)}")
        print(f"  N_Q_model_high_known: {np.count_nonzero(high_mask)}")
        print(f"  Q_model_max_resolution: {Q_model[-1]:.3e} A^-1")
        print(f"  alpha_reg: {alpha_reg:.6e}")
        print(f"  ridge: {ridge:.6e}")
        print(f"  eta_H: {eta_H:.6e}")
        if "E_floor" in cases[case_name]:
            print(f"  E_floor: {cases[case_name]['E_floor']:.16e}")
        print(f"  ell_log: {ell_log:.6e}")
        print(f"  ell_log_multiplier: {ell_log_multiplier:.6e}")
        print(f"  h_instr_log: {kernel_info['h_instr_log']:.6e}")
        print(f"  ell_log / h_instr_log: {ell_log / kernel_info['h_instr_log']:.6e}")
        print(f"  kernel alpha: {kernel_info['alpha']:.6e}")
        print(f"  kernel beta: {kernel_info['beta']:.6e}")
        print(f"  kernel gamma: {kernel_info['gamma']:.6e}")
        print(f"  sum(t_i): {np.sum(t_opt):.16e}")
        print(f"  E(t_opt): {E_opt:.16e}")
        print(f"  min(g_i): {np.min(g):.16e}")
        print(f"  max(g_i): {np.max(g):.16e}")
        print(f"  max(g_i) / min(g_i): {np.max(g) / np.min(g):.16e}")

    T_values = np.logspace(-2, 2, 80)
    rows = []
    for case_name in figure2_case_order:
        case = cases[case_name]
        I_ref = case["I_ref"]
        for T_tot in T_values:
            if case["kind"] == "base":
                t_opt_strategy = optimal_allocation_from_a(1.0 / np.maximum(I_ref, EPS), T_tot)
            else:
                t_opt_strategy = T_tot * case["t_opt_demo"] / np.sum(case["t_opt_demo"])
            strategy_allocations = {
                "Uniform": uniform_allocation(N, T_tot),
                "Inverse intensity": inverse_intensity_allocation(I_ref, T_tot),
                "Optimal": t_opt_strategy,
            }
            for strategy, t in strategy_allocations.items():
                E_value, _ = case["score_func"](t)
                rows.append({"case": case_name, "strategy": strategy, "T_tot": T_tot, "E": E_value})

    power_values = np.linspace(-0.5, 1.0, 151)
    power_rows = []
    optimal_rows = []
    for case_name in figure2_case_order:
        case = cases[case_name]
        I_ref = case["I_ref"]
        optimal_rows.append({"case": case_name, "E_opt": case["E_opt_demo"]})
        for power_n in power_values:
            t_power = inverse_intensity_power_allocation(I_ref, power_n, T_tot_fig1)
            E_value, _ = case["score_func"](t_power)
            power_rows.append({"case": case_name, "power_n": power_n, "E": E_value})

    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "benchmark_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "strategy", "T_tot", "E"])
        writer.writeheader()
        writer.writerows(rows)

    plot_figure1(Q, I_true, I_smeared, allocations, output_dir / "figure1_profiles_and_allocations.png")
    plot_marginal_scores(Q, score_profiles, output_dir / "diagnostic_marginal_scores.png")
    plot_power_scan(power_rows, optimal_rows, output_dir / "figure2_power_scan.png", case_order=figure2_case_order)
    print_total_count_diagnostics(cases, T_tot_figure3_by_case, case_order=figure3_case_order)
    plot_errorbar_bands(
        Q,
        I_true,
        cases,
        output_dir / "figure3_errorbar_bands.png",
        T_tot_by_case=T_tot_figure3_by_case,
        case_order=figure3_case_order,
    )


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-folder", default="usans_benchmark_output")
    parser.add_argument("--show-correlation-aware", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark(output_folder=args.output_folder, show_correlation_aware=args.show_correlation_aware)


if __name__ == "__main__":
    main()
