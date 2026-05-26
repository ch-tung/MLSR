# USANS Time-Allocation Benchmarks

This folder contains scripts for benchmarking and visualizing counting-time allocation strategies for one-dimensional USANS measurements.

The main examples compare allocations of the form

```text
t_i proportional to I(Q_i)^(-nu)
```

with special cases:

- `nu = 0`: uniform time per Q position
- `nu = 0.5`: optimal allocation for the base relative-counting-error model
- `nu = 1`: constant relative counting error per Q position

## Files

- `usans_lib.py`  
  Core numerical utilities: Bonse-Hart slit PSF, ground-truth profiles, allocation rules, uncertainty metrics, correlation kernels, and plotting helpers.

- `run_usans_benchmark.py`  
  Runs the synthetic USANS allocation benchmark and writes figures/CSV output to `usans_benchmark_output/`.

- `prepare_publication_figures.py`  
  Produces the synthetic publication Figure 1 in `publication_figures/publication_figure1.png`. The current version uses a four-panel layout:
  - `(a)` averaged relative uncertainty versus allocation power `nu`
  - `(b)` average counts per Q position versus `nu`
  - `(c)` absolute intensity bands for selected allocation strategies
  - `(d)` relative intensity bands for selected allocation strategies

- `analyze_usans_time_allocation.py`  
  Analyzes measured USANS data and estimates the uncertainty curve for allocation powers.

- `analyze_usans_time_allocation_uniform_current.py`  
  Publication Figure 2 workflow using a uniform-time "Current" comparison and an optimized allocation comparison. It supports optional log/linear Q resampling and optional sampled points.

- `usans_tests.py`  
  Diagnostic tests and checks for the USANS utility functions.

- `benchmark_usans_time_allocation.py`  
  Earlier self-contained benchmark script retained for reference.

## Data

The measured USANS example uses:

```text
data/usans/36537/reduced_c1/UN_C1_det_1_lbs.txt
```

The reduced file is expected to contain at least three columns:

```text
Q, I(Q), dI(Q)
```

## Environment

Use the local Python environment that has the scientific stack installed, for example the local `pyvista` environment used in this project.

Required Python packages:

```text
numpy
scipy
pandas
matplotlib
```

## Regenerate Synthetic Publication Figure 1

Default linear-Q calculation:

```powershell
python -B prepare_publication_figures.py --output-folder publication_figures
```

Log-Q calculation:

```powershell
python -B prepare_publication_figures.py --q-grid log --output-folder publication_figures_log
```

## Regenerate Measured-Data Publication Figure 2

Without sampled points:

```powershell
python -B analyze_usans_time_allocation_uniform_current.py --q-grid log --q-points 200 --output-folder usans_analysis_output_uniform_current --publication-output publication_figures\publication_figure2_uniform_current.png
```

With sampled points:

```powershell
python -B analyze_usans_time_allocation_uniform_current.py --q-grid log --q-points 200 --show-points --output-folder usans_analysis_output_uniform_current_points --publication-output publication_figures\publication_figure2_uniform_current_points.png
```

The `--q-grid` option accepts:

```text
lin
log
input
```

For `lin` and `log`, `--q-points` controls the number of resampled Q positions. For `input`, the original measured Q positions are preserved.

## Run Checks

Compile checks:

```powershell
python -B -m py_compile usans_lib.py usans_tests.py run_usans_benchmark.py prepare_publication_figures.py analyze_usans_time_allocation.py analyze_usans_time_allocation_uniform_current.py
```

Diagnostic tests:

```powershell
python -B usans_tests.py
```

