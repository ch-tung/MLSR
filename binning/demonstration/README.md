# MLSR Binning Demonstration

Minimal reproducible example of the bin-width and kernel-size estimates used in
the MLSR `binning` notebooks.

The reusable function is `estimate_binning_scales` in `binning.py`. It
accepts 1D scattering data `I(Q)` and returns:

- `h_fd`: estimated Freedman-Diaconis optimal bin width
- `lambda_opt`: estimated Gaussian/RBF kernel length
- `lambda_ab`: alpha-beta kernel-length heuristic, equal to `h_fd`
- `alpha`, `beta`, `gamma`, `chi`: intermediate quantities from the notebooks

The module also includes `rbf_gpr_predict`, a small Gaussian-process posterior
mean helper that uses `lambda_opt` as an RBF-equivalent kernel size. The
notebook plots ground truth, noisy observations, and the optimized GPR
reconstruction with a posterior +/- 1 sigma uncertainty band.

The GPR helper supports `kernel="rbf"`, `kernel="matern32"`, and
`kernel="matern52"`. The Matern kernels are parameterized so their normalized
second moment matches the RBF kernel with the same `kernel_size`, making the
curvature-bias comparison use the same effective smoothing moment.

The notebook also exposes a `lambda_scale` option. The default value `1.0` uses
the manuscript-derived `lambda_opt`, while other values let users intentionally
inspect nearby kernel sizes. Smaller kernels can follow noisy fluctuations,
whereas wider kernels increase curvature-induced bias from the underlying
signal. The diagnostic panel below the GPR reconstruction compares a local
Gaussian-weighted counting-error estimate with `4 x` squared curvature bias.
The counting-error curve varies with `Q` because each point has different
measurement uncertainty and a different set of nearby samples inside the kernel.
The curvature-bias curve is also estimated from the sampled observation using
the same derivative settings used for `lambda_opt`, so the lower-panel
diagnostic does not require ground truth.
Users can adjust `lambda_scale` to choose how strongly they want to trade noise
suppression against curvature bias for their data.

In the synthetic data generator, `total_counts` is distributed across the full
Q range in proportion to the underlying intensity. Low-intensity regions
therefore have fewer counts and larger relative error, even though their
absolute intensity error can be smaller. The notebook includes
`use_relative_error`; set it to `True` to plot relative squared contributions
instead of absolute MSE contributions.

At `lambda_scale = 1`, the integrated comparison is only approximate:
`lambda_opt` comes from the manuscript's asymptotic average MSE model, while the
plotted counting error is a finite-data local estimate. The synthetic ground
truth is shown only in the upper panel to validate the example.

The notebook also provides `kernel_choice = "alpha_beta"`, which uses
`lambda_ab = h_fd` as a practical alpha-beta kernel scale. This avoids the
explicit curvature estimate `gamma`, but it is a heuristic comparison scale, not
the same manuscript-derived GP optimum as the alpha-gamma `lambda_opt`.
Set `gpr_kernel` in the notebook to switch between the RBF and matched-moment
Matern kernels.

Run the script:

```powershell
python binning.py
```

Or open `binning.ipynb` and run the cells.

The implementation follows the formulas used in the original notebooks:

```text
beta = integral I'(Q)^2 dQ / (12 L)
gamma = integral I''(Q)^2 dQ / (4 L)
A0 = L * mean(I)^2
alpha = A0 / total_counts
h_FD = (alpha / (2 beta))^(1/3)
lambda_opt = (alpha / (8 sqrt(pi) gamma))^(1/5)
lambda_ab = h_FD
```

`scipy` is optional but recommended. When available, the function uses a
Savitzky-Golay filter for derivative estimation, matching the spirit of the
MLSR notebooks. Without `scipy`, it falls back to `numpy.gradient`.
