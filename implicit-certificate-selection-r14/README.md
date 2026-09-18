# Reproducibility package — manuscript R14

This directory accompanies **Degenerate Endpoint Smoothness for Exponential Penalties and Entropic Optimal Transport**.

## Environment

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Core theorem diagnostics

```bash
python reproduce_r14.py
```

This checks the genuinely incommensurate LP expansion, the exact polynomial certificate behind the `> 3/2` local-misspecification risk theorem, the numerical optimum `1.7081773994`, and a fixed-misspecification illustration.

## Four-point block-exponent diagnostic

```bash
python ot_extrapolation.py
```

This uses 140-digit damped Newton solves on the fixed 4x4 empirical-OT stress test. It independently computes the first nonzero dual block exponent `0.3930843911` and fits the observed two-point secant errors; the last-two-point fit is approximately `0.38917`.

The coordinates are deterministic stress-test coordinates, not random draws.

## Real empirical OT tangent benchmark

```bash
python ot_tangent_digits.py
```

This uses `sklearn.datasets.load_digits`. For ten prespecified digit-class pairs, it takes the first 100 observations in each class, rescales pixels to `[0,1]`, normalizes mean squared Euclidean cost by its pairwise median, solves at `eps=0.05`, obtains the gauge-fixed potential derivative by implicit differentiation, and targets `eps=0.025`.

At marginal residual tolerances `1e-6` and `1e-9`, the tangent initialization reduces log-domain Sinkhorn correction iterations for all ten pairs. Median iteration counts are `26 -> 21` and `53 -> 46.5`.

The benchmark counts only correction iterations after the derivative is available. It does **not** claim that the dense derivative solve is free or that wall-clock time always improves.
