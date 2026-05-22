"""Analyze measured USANS data with inverse-intensity time-allocation rules."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from usans_lib import EPS, inverse_intensity_power_allocation, uniform_allocation


def load_usans_data(path):
    """Load a reduced USANS text file with columns Q, I(Q), and dI(Q)."""
    data = np.genfromtxt(path, delimiter=",")
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"expected at least three columns in {path}")
    q = np.asarray(data[:, 0], dtype=float)
    intensity = np.asarray(data[:, 1], dtype=float)
    sigma = np.asarray(data[:, 2], dtype=float)
    mask = np.isfinite(q) & np.isfinite(intensity) & np.isfinite(sigma) & (intensity > 0)
    return q[mask], intensity[mask], sigma[mask]


def averaged_relative_counting_error(intensity, t):
    """Return mean relative Poisson variance, mean_i 1 / (I_i t_i)."""
    intensity = np.asarray(intensity, dtype=float).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)
    return float(np.mean(1.0 / np.maximum(intensity * t, EPS)))


def intensity_power_allocation(intensity, power, T_tot):
    """Return allocation proportional to I**power."""
    intensity = np.asarray(intensity, dtype=float).reshape(-1)
    weights = np.maximum(intensity, EPS) ** power
    return T_tot * weights / np.maximum(np.sum(weights), EPS)


def estimate_current_allocation_from_errors(intensity, sigma):
    """Estimate current exposure times from relative Poisson error bars."""
    intensity = np.asarray(intensity, dtype=float).reshape(-1)
    sigma = np.asarray(sigma, dtype=float).reshape(-1)
    return np.maximum(intensity, EPS) / np.maximum(sigma * sigma, EPS)


def smooth_intensity_log_gaussian(q, intensity, ell_log=0.22):
    """Return a smooth positive intensity profile using Gaussian smoothing in log Q."""
    q = np.asarray(q, dtype=float).reshape(-1)
    intensity = np.asarray(intensity, dtype=float).reshape(-1)
    logq = np.log(np.maximum(q, EPS))
    dlog = logq[:, None] - logq[None, :]
    weights = np.exp(-0.5 * (dlog / ell_log) ** 2)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), EPS)
    smooth_log_intensity = weights @ np.log(np.maximum(intensity, EPS))
    return np.exp(smooth_log_intensity)


def plot_iq_with_errorbars(q, intensity, sigma, output_path):
    """Plot measured I(Q) with error bars."""
    fig, ax = plt.subplots(figsize=(6.0, 4.8), constrained_layout=True)
    ax.errorbar(q, intensity, yerr=sigma, fmt="o", ms=4.0, lw=1.0, capsize=2.0, color="black")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$Q$ ($\mathrm{\AA}^{-1}$)")
    ax.set_ylabel(r"$I(Q)$")
    ax.grid(True, which="both", alpha=0.25)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_error_vs_power(power_rows, output_path):
    """Plot averaged relative error versus inverse-intensity power n."""
    fig, ax = plt.subplots(figsize=(5.6, 4.8), constrained_layout=True)
    ax.plot(
        [row["n"] for row in power_rows],
        [row["E"] for row in power_rows],
        color="black",
        lw=2.0,
    )
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$\mathcal{E}$")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_publication_strategy_figure(q, intensity, sigma, output_path):
    """Plot current and optimized USANS strategies plus E(n)."""
    rng = np.random.default_rng(12345)
    smooth_intensity = smooth_intensity_log_gaussian(q, intensity)
    t_current = estimate_current_allocation_from_errors(intensity, sigma)
    T_current = float(np.sum(t_current))
    exposure_scale = 0.1
    T_compare = exposure_scale * T_current
    t_current = exposure_scale * t_current
    t_opt = inverse_intensity_power_allocation(smooth_intensity, 0.5, T_compare)

    current_counts = rng.poisson(np.maximum(smooth_intensity * t_current, EPS))
    opt_counts = rng.poisson(np.maximum(smooth_intensity * t_opt, EPS))
    intensity_current_sample = current_counts / np.maximum(t_current, EPS)
    intensity_opt_sample = opt_counts / np.maximum(t_opt, EPS)

    sigma_current = np.sqrt(np.maximum(smooth_intensity, EPS) / np.maximum(t_current, EPS))
    sigma_opt = np.sqrt(np.maximum(smooth_intensity, EPS) / np.maximum(t_opt, EPS))
    rel_current_sigma = sigma_current / np.maximum(smooth_intensity, EPS)
    rel_opt_sigma = sigma_opt / np.maximum(smooth_intensity, EPS)
    rel_opt_sigma_band = smooth_intensity_log_gaussian(q, rel_opt_sigma, ell_log=0.16)
    sigma_opt_band = rel_opt_sigma_band * smooth_intensity
    E_current = averaged_relative_counting_error(smooth_intensity, t_current)

    n_values = np.linspace(0.0, 1.0, 201)
    E_values = []
    for n in n_values:
        t = inverse_intensity_power_allocation(smooth_intensity, n, T_compare)
        E_values.append(averaged_relative_counting_error(smooth_intensity, t))

    shift = 20.0
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.8), constrained_layout=True)
    ax_left, ax_right = axes

    ax_left.fill_between(
        q,
        np.maximum(smooth_intensity - sigma_current, EPS),
        smooth_intensity + sigma_current,
        color="black",
        alpha=0.18,
        linewidth=0,
    )
    ax_left.plot(q, smooth_intensity, color="black", lw=1.2, alpha=0.85)
    ax_left.errorbar(
        q,
        np.maximum(intensity_current_sample, EPS),
        yerr=sigma_current,
        fmt="o",
        ms=4.0,
        lw=1.0,
        capsize=2.0,
        color="black",
        alpha=0.75,
        label="Current",
    )
    ax_left.fill_between(
        q,
        np.maximum((smooth_intensity - sigma_opt_band) * shift, EPS),
        (smooth_intensity + sigma_opt_band) * shift,
        color="red",
        alpha=0.18,
        linewidth=0,
    )
    ax_left.plot(q, smooth_intensity * shift, color="red", lw=1.2, alpha=0.85)
    ax_left.errorbar(
        q,
        np.maximum(intensity_opt_sample, EPS) * shift,
        yerr=sigma_opt * shift,
        fmt="o",
        ms=4.0,
        lw=1.0,
        capsize=2.0,
        color="red",
        alpha=0.75,
        label="Optimized",
    )
    ax_left.set_xscale("log")
    ax_left.set_yscale("log")
    ax_left.set_ylim(1.0e2, 1.0e8)
    ax_left.set_xlabel(r"$Q$ ($\mathrm{\AA}^{-1}$)")
    ax_left.set_ylabel(r"$I(Q)$ ($\mathrm{cm}^{-1}$)")
    ax_left.set_box_aspect(1)
    ax_left.grid(True, which="both", alpha=0.25)
    ax_left.legend(frameon=False)
    inset = inset_axes(
        ax_left,
        width="50%",
        height="32%",
        bbox_to_anchor=(-0.38, -0.6, 1, 1),
        bbox_transform=ax_left.transAxes,
    )
    inset.fill_between(
        q,
        np.maximum(1.0 - rel_current_sigma, EPS),
        1.0 + rel_current_sigma,
        color="black",
        alpha=0.18,
        linewidth=0,
    )
    inset.fill_between(
        q,
        np.maximum(1.0 - rel_opt_sigma_band, EPS),
        1.0 + rel_opt_sigma_band,
        color="red",
        alpha=0.18,
        linewidth=0,
    )
    inset.axhline(1.0, color="0.35", lw=0.9, ls="--", alpha=0.75)
    inset.set_xscale("log")
    inset.set_ylim(0.5, 1.5)
    inset.set_yticks([0.5, 1.0, 1.5])
    inset.set_xlabel(r"$Q$", fontsize=10)
    inset.set_ylabel(r"$I/I_{\mathrm{gt}}$", fontsize=10, rotation=0)
    inset.xaxis.set_label_coords(1, -0.08)
    inset.yaxis.set_label_coords(-0.1, 1.05)
    inset.tick_params(labelsize=8)
    inset.grid(True, which="both", alpha=0.18)
    ax_left.text(
        0.02,
        0.98,
        "(a)",
        transform=ax_left.transAxes,
        ha="left",
        va="top",
        fontsize=20,
        fontweight="bold",
    )

    best_index = int(np.argmin(E_values))
    ax_right.plot(n_values, E_values, color="black", lw=2.0)
    ax_right.axhline(E_current, color="black", lw=1.6, ls="--", label="Current")
    ax_right.plot(n_values[best_index], E_values[best_index], "o", color="red", ms=7.0, label="Optimized")
    ax_right.set_xlabel(r"$n$")
    ax_right.set_ylabel(r"$\mathcal{E}_0$")
    ax_right.set_ylim(0.04, 0.09)
    ax_right.set_box_aspect(1)
    ax_right.grid(True, alpha=0.25)
    ax_right.legend(frameon=False)
    ax_right.text(
        0.02,
        0.98,
        "(b)",
        transform=ax_right.transAxes,
        ha="left",
        va="top",
        fontsize=20,
        fontweight="bold",
    )

    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return {
        "T_current": T_current,
        "T_compare": T_compare,
        "exposure_scale": exposure_scale,
        "E_current": E_current,
        "best_n": float(n_values[best_index]),
        "best_E": float(E_values[best_index]),
    }


def analyze_usans_case(
    input_path,
    output_folder,
    target_uniform_counts_per_point=100.0,
    publication_output=None,
):
    """Run the c1 USANS analysis and write figures plus a CSV summary."""
    input_path = Path(input_path)
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    q, intensity, sigma = load_usans_data(input_path)
    positive_q_mask = q > 0.0
    q = q[positive_q_mask]
    intensity = intensity[positive_q_mask]
    sigma = sigma[positive_q_mask]
    n_points = intensity.size
    T_tot = target_uniform_counts_per_point * n_points / np.mean(intensity)

    power_values = np.linspace(0.0, 1.0, 201)
    rows = []
    for power_n in power_values:
        t = inverse_intensity_power_allocation(intensity, power_n, T_tot)
        rows.append(
            {
                "n": float(power_n),
                "T_tot": float(T_tot),
                "E": averaged_relative_counting_error(intensity, t),
            }
        )

    t_uniform = uniform_allocation(n_points, T_tot)
    mean_uniform_counts = float(np.mean(intensity * t_uniform))

    plot_iq_with_errorbars(q, intensity, sigma, output_dir / "c1_usans_iq_errorbar.png")
    plot_error_vs_power(rows, output_dir / "c1_usans_relative_error_vs_n.png")
    publication_path = (
        Path(publication_output)
        if publication_output is not None
        else output_dir / "c1_usans_publication_strategy.png"
    )
    if publication_path.parent != Path(""):
        publication_path.parent.mkdir(parents=True, exist_ok=True)
    publication_stats = plot_publication_strategy_figure(q, intensity, sigma, publication_path)

    with open(output_dir / "c1_usans_power_scan.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "T_tot", "E"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Input: {input_path}")
    print(f"Points used: {n_points}")
    print(f"T_tot: {T_tot:.16e}")
    print(f"Uniform mean counts per point: {mean_uniform_counts:.16e}")
    print(f"Minimum E: {min(row['E'] for row in rows):.16e}")
    print(f"Best n: {rows[int(np.argmin([row['E'] for row in rows]))]['n']:.6f}")
    print(f"Current-strategy T_tot estimate: {publication_stats['T_current']:.16e}")
    print(f"Synthetic comparison T_tot: {publication_stats['T_compare']:.16e}")
    print(f"Synthetic exposure scale: {publication_stats['exposure_scale']:.16e}")
    print(f"Current-strategy E: {publication_stats['E_current']:.16e}")
    print(f"Best I^-n strategy n: {publication_stats['best_n']:.6f}")
    print(f"Best I^-n strategy E: {publication_stats['best_E']:.16e}")
    print(f"Wrote {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/usans/36537/reduced_c1/UN_C1_det_1_lbs.txt",
        help="Path to the reduced USANS file to analyze.",
    )
    parser.add_argument("--output-folder", default="usans_analysis_output")
    parser.add_argument(
        "--publication-output",
        default=None,
        help="Optional path for the two-panel publication figure.",
    )
    parser.add_argument("--target-uniform-counts", type=float, default=100.0)
    return parser.parse_args()


def main():
    args = parse_args()
    analyze_usans_case(
        args.input,
        args.output_folder,
        args.target_uniform_counts,
        publication_output=args.publication_output,
    )


if __name__ == "__main__":
    main()
