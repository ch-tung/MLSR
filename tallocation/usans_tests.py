"""Diagnostic self-checks for the USANS time-allocation benchmark."""

from __future__ import annotations

import shutil
from pathlib import Path

from run_usans_benchmark import run_benchmark


def run_output_smoke_check() -> dict[str, Path]:
    """Run the benchmark in a temporary folder and verify key outputs exist."""
    output_dir = Path(__file__).resolve().parent / "_usans_smoke_output"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_benchmark(output_folder=output_dir, show_correlation_aware=False)
        expected = [
            output_dir / "benchmark_summary.csv",
            output_dir / "figure1_profiles_and_allocations.png",
            output_dir / "figure2_power_scan.png",
            output_dir / "figure3_errorbar_bands.png",
            output_dir / "diagnostic_marginal_scores.png",
        ]
        missing = [path for path in expected if not path.exists()]
        if missing:
            raise AssertionError(f"Missing benchmark outputs: {missing}")
        return {path.name: path for path in expected}
    finally:
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    run_output_smoke_check()
