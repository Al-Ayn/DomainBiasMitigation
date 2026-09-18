# Reproducibility: smooth exponential-penalty limits and implicit certificate selection

This directory contains the reproducibility code accompanying the manuscript **"Smooth Exponential-Penalty Limits and the Statistical Price of Implicit Certificate Selection."**

## Environment

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Core analytic/numerical checks

```bash
python reproduce_r13.py
```

This reproduces the incommensurate two-scale LP table, the 2x3 Sinkhorn-divergence endpoint table, the replicated-path risk optimum, and the exact positivity checks behind the `> 3/2` best-tuned total-risk theorem.

## Degenerate empirical-OT continuation

```bash
python ot_extrapolation.py
```

This runs 140-digit damped Newton solves for the 4x4 empirical OT example and compares a standard warm start with the two-point predictor `(3/2) h_(2 eps) - (1/2) h_(4 eps)`.

## Premier League comparison-graph experiment

The script does **not** redistribute the underlying match table. By default it fetches the 2024-25 Premier League CSV from football-data.co.uk:

```bash
python epl_pairwise_experiment.py
```

For an offline run, pass either the original football-data CSV or a normalized CSV with columns `date,home,away,hg,ag`:

```bash
python epl_pairwise_experiment.py --csv /path/to/file.csv
```

The experiment uses a fixed chronological 75/20/20 train/validation/test split of the first 115 fixtures and fixed tuning grids. The bootstrap seed is `20260918`.

### Data provenance

Underlying match results: https://www.football-data.co.uk/

Season file used by the script: https://www.football-data.co.uk/mmz4281/2425/E0.csv

The upstream site publishes the historical CSVs freely but does not state explicit redistribution terms. For that reason this directory contains the fetch-and-analysis code but not a copy of the match table.

## Expected headline checks

- Robust replicated-path split: `(6/23, 17/23, 0)`.
- Robust reduced risk: `148/207`.
- Exact best-over-initialization risk lower bound: ratio `> 3/2` for every replication factor.
- Numerical minimum of the implicit reduced curve: approximately `1.7081773994` times robust risk.
- Premier League test RMSE: electrical `1.8613`, robust `1.8317`, implicit `1.9167`.
- At `eps=0.0125`, the 4x4 OT two-point potential error is approximately `9.62e-6`, versus `1.73e-2` for the ordinary warm start.
