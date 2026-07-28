# MLSR Binning Demonstration

Minimal reproducible example of the bin-width and kernel-size estimates used in
the MLSR `binning` notebooks.

The reusable function is `estimate_binning_scales` in `binning.py`. It
accepts 1D scattering data `I(Q)` and returns:

- `h_fd`: estimated Freedman-Diaconis optimal bin width
- `lambda_opt`: estimated equivalent Gaussian/RBF smoothing width,
  \(\lambda_{\mathcal S}\)
- `lambda_ab`: alpha-beta kernel-length heuristic, equal to `h_fd`
- `alpha`, `beta`, `gamma`, `chi`: intermediate quantities from the notebooks

The module also includes `rbf_gpr_predict`, a small Gaussian-process posterior
mean helper, and `gp_covariance_length_from_smoothing_width`, which
automatically converts the equivalent smoothing width
\(\lambda_{\mathcal S}\) into the covariance length
\(\lambda_{\mathcal K}\) required by the exact GP backend. The notebook plots
ground truth, noisy observations, and the corrected GPR reconstruction with a
posterior +/- 1 sigma uncertainty band.

The GPR and correction helpers support `kernel="rbf"`, `kernel="matern32"`,
and `kernel="matern52"`. The Matern kernels use a matched second-moment
parameterization, and the correction is recomputed with the selected family.

Although the covariance matrix \(\mathcal K\) and equivalent smoothing
operator \(\mathcal S\) use the same kernel family, their widths are not
generally equal because the GP posterior acts through
\(\mathcal S=\mathcal K(\mathcal K+\Sigma)^{-1}\). The correction routine
constructs the normalized equivalent smoother \(W_{\lambda_{\mathcal S}}\)
on the actual sampling grid and solves

```text
trace(S Sigma S.T) = trace(W Sigma W.T)
```

for \(\lambda_{\mathcal K}\). The resulting deterministic operator correction
automatically includes finite boundaries, nonuniform sampling, heteroscedastic
observation errors, and the selected GP signal variance. The notebook reports
the correction factor and uses \(\lambda_{\mathcal K}\) only in the GP
covariance. Curvature and kernel-weighted error diagnostics continue to use
\(\lambda_{\mathcal S}=\mathtt{lambda_opt}\), since that is the width appearing
in the error analysis.

```python
signal_variance = float(np.var(intensity))
correction = gp_covariance_length_from_smoothing_width(
    q,
    intensity_error,
    smoothing_width=result.lambda_opt,
    signal_variance=signal_variance,
)
gpr_mean = rbf_gpr_predict(
    q,
    intensity,
    intensity_error,
    kernel_size=correction.covariance_length,
    signal_variance=signal_variance,
)
```

The diagnostic panel below the GPR reconstruction compares a local
kernel-weighted counting-error estimate with `4 x` squared curvature bias. Both
are estimated from the sampled observation, so the diagnostic does not require
ground truth.

In the synthetic data generator, `total_counts` is distributed across the full
Q range in proportion to the underlying intensity. Low-intensity regions
therefore have fewer counts and larger relative error, even though their
absolute intensity error can be smaller. The notebook includes
`use_relative_error`; set it to `True` to plot relative squared contributions
instead of absolute MSE contributions.

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
