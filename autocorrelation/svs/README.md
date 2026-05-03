# SVS Benchmark Toolkit

This project contains a small Python toolkit for synthetic Speckle Visibility
Spectroscopy (SVS) tests. It generates intensity time series from a prescribed
normalized autocorrelation, applies finite-count Poisson observation noise,
estimates SVS visibility, and reconstructs the underlying autocorrelation with
Bayesian/Gaussian-process inference.

## Files

- `svs_lib.py`: core SVS library functions.
- `run_svs_benchmark.py`: command-line benchmark pipeline.
- `svs_tests.py`: testing and self-check scripts.
- `svs_benchmark_double_exp.ipynb`: notebook reproducing the double-exponential benchmark.
- `svs_benchmark_output_double_exp/`: current preferred double-exponential result.
- `svs_benchmark_output_single_exp/`: single-exponential benchmark output.
- `svs_benchmark_output_damped_oscillation/`: damped-oscillation benchmark output.

## Current Pipeline

The benchmark follows:

```text
known C(tau)
-> synthetic intensity I_true(t)
-> Poisson observed intensity I_obs(t)
-> segment visibility Khat^2(T)
-> uncertainty Sigma_K
-> Bayesian reconstruction of C(tau)
```

The Poisson observation model uses `kappa` as the conversion from integrated
intensity to expected detected counts:

```text
E[N_T | Ibar_T] = kappa * T * Ibar_T
Var[N_T | Ibar_T] = kappa * T * Ibar_T
```

The measured segment-average intensity is therefore noisy at low `kappa`, low
`mean_I`, and short exposure time `T`.

## Implemented Models

Supported target autocorrelation models include:

- `exponential`
- `double_exponential`
- `damped_oscillation`

The double-exponential model is:

```text
C(tau) = beta * [f exp(-tau/tau1) + (1 - f) exp(-tau/tau2)]
```

The current nice result uses `beta = 1`, `f = 0.6`, `tau1 = 0.5`, and
`tau2 = 2.5`.

## Visibility Estimation

For each exposure time `T`, the observed time series is segmented into windows.
The visibility estimator subtracts Poisson counting noise:

```text
Khat^2(T) = [Var(Ibar_T) - mu_T/(kappa T)] / mu_T^2
```

The code estimates a diagonal covariance for `Khat^2(T)` using moment
propagation from the observed segment averages. An optional full covariance
estimator is also available for block-aligned segment estimates.

## Bayesian Reconstruction

The inverse problem uses:

```text
Khat = R C + eta
eta ~ Normal(0, Sigma_K)
C ~ Normal(C0, Sigma_C)
```

The SVS forward operator is:

```text
R_ij = (2/T_i) * (1 - tau_j/T_i) * indicator(0 <= tau_j <= T_i) * w_j
```

The prior covariance is a squared-exponential Gaussian-process kernel. It can
be built in either linear delay time or log delay time:

```bash
--prior-covariance-space linear
--prior-covariance-space log
```

The log-time option is often useful because the reconstruction grid and exposure
times are commonly logarithmically spaced. The reconstruction grid skips
`tau = 0`, but starts below the smallest exposure time so the low-`T` visibility
rows are not degenerate.

Available prior mean choices:

- `zero`
- `exponential`
- `direct_smooth`

The `direct_smooth` option estimates a direct autocorrelation from `I_obs`,
smooths it, and uses it as a rough prior mean.

## Plot Options

The benchmark can toggle displayed autocorrelation estimates:

```bash
--show-autocorr-direct / --no-show-autocorr-direct
--show-autocorr-fft / --no-show-autocorr-fft
--show-autocorr-pinv / --no-show-autocorr-pinv
--show-autocorr-bayes / --no-show-autocorr-bayes
--show-prior-mean / --no-show-prior-mean
```

Autocorrelation axis controls:

```bash
--autocorr-time-scale linear
--autocorr-time-scale log
--autocorr-scale linear
--autocorr-scale log
--autocorr-time-range MIN MAX
--autocorr-range MIN MAX
```

## Reproduce Double-Exponential Result

The saved result in `svs_benchmark_output_double_exp` can be reproduced from the
notebook:

```text
svs_benchmark_double_exp.ipynb
```

or from the command line using the same parameter set:

```bash
python -B run_svs_benchmark.py ^
  --prior-mean direct_smooth ^
  --correlation-model double_exponential ^
  --f 0.6 ^
  --tau1 0.5 ^
  --tau2 2.5 ^
  --T-spacing log ^
  --N-T 31 ^
  --total-time 1200 ^
  --dt 0.01 ^
  --T-min 0.01 ^
  --T-max 5.0 ^
  --mean-I 1 ^
  --kappa 10 ^
  --sigma-C 0.1 ^
  --lambda 0.2 ^
  --prior-covariance-space log ^
  --no-low-tau-prior-anchor ^
  --autocorr-time-scale log ^
  --autocorr-time-range 0.05 5.0 ^
  --output-folder svs_benchmark_output_double_exp
```

The main output figure is:

```text
svs_benchmark_output_double_exp/reconstructed_c.png
```

It contains the signal, measured visibility against Bayesian forward prediction,
autocorrelation reconstruction, and autocorrelation residuals.

## Current Outputs

Each benchmark output folder contains:

- `synthetic_time_series.png`
- `target_vs_empirical_autocorrelation.png`
- `measured_visibility.png`
- `reconstructed_c.png`
- `visibility_residuals.png`
- `benchmark_results.json`

The JSON file stores the full configuration, metrics, figure paths, metadata,
and log-spacing diagnostics.

## To Do

### Efficiency Benchmark

Build a systematic benchmark over count rate/noise level using different values
of `kappa`. The goal is to quantify when the SVS Bayesian inversion outperforms
direct autocorrelation estimation from `I_obs`.

Suggested design:

1. Choose a fixed ground-truth model, for example the current double-exponential
   model.
2. Sweep `kappa`, for example:

   ```text
   kappa = [1, 2, 5, 10, 20, 50, 100]
   ```

3. For each `kappa`, run multiple random seeds.
4. For each run, compute:

   ```text
   MSE_Bayes = mean((C_Bayes(tau) - C_true(tau))^2)
   MSE_direct = mean((C_direct(tau) - C_true(tau))^2)
   ```

5. Interpolate `C_direct` onto the same delay grid used by the Bayesian
   reconstruction before comparing.
6. Report mean and standard deviation of the loss over random seeds.
7. Plot loss versus `kappa` for Bayes inversion and direct autocorrelation.

Possible loss definitions:

```text
unweighted MSE:
mean((C_est - C_true)^2)
```

```text
delay-windowed MSE:
mean((C_est - C_true)^2 over tau_min <= tau <= tau_max)
```

```text
log-time weighted MSE:
mean((C_est - C_true)^2 weighted uniformly in log(tau))
```

The log-time weighted loss may be the most relevant when the reconstruction grid
and exposure times are log-spaced.

### Suggested Efficiency-Benchmark Output

Create a new script, for example:

```text
run_svs_efficiency_benchmark.py
```

Expected outputs:

- `efficiency_metrics.json`
- `mse_vs_kappa.png`
- `example_reconstructions_by_kappa.png`
- optional runtime measurements for visibility estimation and inversion

The main comparison should be:

```text
Bayes/GPR reconstruction MSE vs direct autocorrelation MSE
```

as a function of count-rate quality `kappa`.
