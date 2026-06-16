# MLSR Binning Demonstration

Minimal reproducible example of the bin-width and kernel-size estimates used in
the MLSR `binning` notebooks.

The reusable function is `estimate_binning_scales` in `binning.py`. It
accepts 1D scattering data `I(Q)` and returns:

- `h_fd`: estimated Freedman-Diaconis optimal bin width
- `lambda_opt`: estimated Gaussian/RBF kernel length
- `alpha`, `beta`, `gamma`, `chi`: intermediate quantities from the notebooks

The module also includes `rbf_gpr_predict`, a small Gaussian-process posterior
mean helper that uses `lambda_opt` as the RBF kernel size. The notebook plots
ground truth, noisy observations, and the optimized GPR reconstruction with a
posterior +/- 1 sigma uncertainty band.

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
```

`scipy` is optional but recommended. When available, the function uses a
Savitzky-Golay filter for derivative estimation, matching the spirit of the
MLSR notebooks. Without `scipy`, it falls back to `numpy.gradient`.
