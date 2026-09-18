#!/usr/bin/env python3
"""Chronological Premier League pairwise-certificate experiment for manuscript R13.

By default this script fetches the 2024-25 English Premier League CSV directly
from football-data.co.uk and keeps only the first 115 fixtures. To run fully
offline, pass --csv to a local normalized file with columns
    date, home, away, hg, ag
or to the original football-data CSV with columns
    Date, HomeTeam, AwayTeam, FTHG, FTAG.

No source match table is distributed with the manuscript package because the
upstream site states no explicit redistribution license.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from scipy.linalg import null_space
from scipy.optimize import brentq, minimize

SOURCE_URL = "https://www.football-data.co.uk/mmz4281/2425/E0.csv"
ROBUST_GRID = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
IMPLICIT_GRID = [0.01, 0.0215443469, 0.0464158883, 0.1, 0.215443469]
BOOTSTRAP_SEED = 20260918


def load_matches(path_or_url: str) -> pd.DataFrame:
    df = pd.read_csv(path_or_url)
    lower = {c.lower(): c for c in df.columns}
    if {"date", "home", "away", "hg", "ag"}.issubset(lower):
        out = pd.DataFrame(
            {
                "date": pd.to_datetime(df[lower["date"]]),
                "home": df[lower["home"]].astype(str),
                "away": df[lower["away"]].astype(str),
                "hg": pd.to_numeric(df[lower["hg"]]),
                "ag": pd.to_numeric(df[lower["ag"]]),
            }
        )
    else:
        required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"missing columns: {missing}")
        out = pd.DataFrame(
            {
                "date": pd.to_datetime(df["Date"], dayfirst=True, errors="raise"),
                "home": df["HomeTeam"].astype(str),
                "away": df["AwayTeam"].astype(str),
                "hg": pd.to_numeric(df["FTHG"]),
                "ag": pd.to_numeric(df["FTAG"]),
            }
        )
    out = out.sort_values("date", kind="stable").reset_index(drop=True).iloc[:115].copy()
    if len(out) != 115:
        raise ValueError(f"expected at least 115 fixtures, got {len(out)}")
    first = out.iloc[0]
    last = out.iloc[-1]
    if not (first["home"] == "Man United" and first["away"] == "Fulham"):
        raise ValueError("unexpected first fixture; source ordering/schema may have changed")
    if not (last["home"] == "Fulham" and last["away"] == "Wolves"):
        raise ValueError("unexpected 115th fixture; source ordering/schema may have changed")
    return out


def make_problem(df: pd.DataFrame):
    teams = sorted(set(df.home) | set(df.away))
    ix = {t: i for i, t in enumerate(teams)}
    train, valid, test = df.iloc[:75], df.iloc[75:95], df.iloc[95:115]
    home_adv = float((train.hg - train.ag).mean())
    y = (train.hg - train.ag).to_numpy(float) - home_adv
    B = np.zeros((len(train), len(teams)))
    for row_i, (_, row) in enumerate(train.iterrows()):
        B[row_i, ix[row.home]] = 1.0
        B[row_i, ix[row.away]] = -1.0
    Lp = np.linalg.pinv(B.T @ B)
    C = null_space(B.T)
    if np.linalg.matrix_rank(B) != len(teams) - 1:
        raise RuntimeError("training comparison graph is not connected")
    return teams, ix, train, valid, test, home_adv, y, B, Lp, C


def evaluator(df: pd.DataFrame):
    teams, ix, train, valid, test, home_adv, y, B, Lp, C = make_problem(df)
    n = len(teams)

    def target_vector(home, away):
        v = np.zeros(n)
        v[ix[home]] = 1.0
        v[ix[away]] = -1.0
        return v

    def electrical(home, away):
        return B @ (Lp @ target_vector(home, away))

    def flow_mu(home, away, mu, p0=None):
        v = target_vector(home, away)
        if p0 is None:
            phi0 = -2 * Lp @ v
            p0 = (phi0 - phi0[-1])[:-1]

        def fg(p):
            phi = np.r_[p, 0.0]
            q = B @ phi
            soft = np.sign(q) * np.maximum(np.abs(q) - mu, 0.0)
            val = 0.25 * np.dot(soft, soft) + np.dot(v, phi)
            grad = (0.5 * B.T @ soft + v)[:-1]
            return val, grad

        res = minimize(
            lambda p: fg(p)[0],
            p0,
            jac=lambda p: fg(p)[1],
            method="L-BFGS-B",
            options={"ftol": 1e-13, "gtol": 3e-9, "maxiter": 800, "maxls": 50},
        )
        if not res.success:
            raise RuntimeError(f"robust inner solve failed: {res.message}")
        phi = np.r_[res.x, 0.0]
        q = B @ phi
        flow = -0.5 * np.sign(q) * np.maximum(np.abs(q) - mu, 0.0)
        return flow, res.x

    def robust(home, away, lam):
        if lam == 0:
            return electrical(home, away)
        i0 = electrical(home, away)
        T0 = np.abs(i0).sum()
        last = [None]

        def fixed_point(mu):
            i, p = flow_mu(home, away, mu, last[0])
            last[0] = p
            return mu - 2 * lam * np.abs(i).sum()

        mu = brentq(fixed_point, 0.0, 2 * lam * T0, xtol=2e-9, rtol=2e-9, maxiter=40)
        return flow_mu(home, away, mu, last[0])[0]

    def implicit(home, away, alpha):
        i0 = electrical(home, away)
        a2 = alpha * alpha
        if C.shape[1] == 0:
            return i0

        def fun(z):
            i = i0 + C @ z
            t = i / a2
            q = t * np.arcsinh(t / 2) - np.sqrt(t * t + 4) + 2
            return a2 * q.sum()

        def grad(z):
            return C.T @ np.arcsinh((i0 + C @ z) / (2 * a2))

        res = minimize(
            fun,
            np.zeros(C.shape[1]),
            jac=grad,
            method="L-BFGS-B",
            options={"maxiter": 1500, "ftol": 1e-12, "gtol": 3e-8, "maxls": 60},
        )
        if not res.success:
            raise RuntimeError(f"implicit solve failed: {res.message}")
        return i0 + C @ res.x

    def eval_split(split, method, param=None, return_raw=False):
        preds, truth, weights = [], [], []
        for _, row in split.iterrows():
            if method == "electrical":
                w = electrical(row.home, row.away)
            elif method == "robust":
                w = robust(row.home, row.away, param)
            elif method == "implicit":
                w = implicit(row.home, row.away, param)
            else:
                raise ValueError(method)
            weights.append(w)
            preds.append(float(w @ y))
            truth.append(float(row.hg - row.ag) - home_adv)
        p = np.asarray(preds)
        t = np.asarray(truth)
        W = np.asarray(weights)
        err = p - t
        out = {
            "rmse": float(np.sqrt(np.mean(err**2))),
            "mae": float(np.mean(np.abs(err))),
            "l1": float(np.mean(np.sum(np.abs(W), axis=1))),
            "l2": float(np.mean(np.sum(W * W, axis=1))),
        }
        return (out, err, W) if return_raw else out

    return {
        "teams": teams,
        "train": train,
        "valid": valid,
        "test": test,
        "home_adv": home_adv,
        "B": B,
        "eval": eval_split,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="local normalized or football-data CSV")
    args = parser.parse_args()
    source = args.csv or SOURCE_URL
    df = load_matches(source)
    P = evaluator(df)

    print("source =", source)
    print("fixtures =", len(df), "teams =", len(P["teams"]))
    print("split =", len(P["train"]), len(P["valid"]), len(P["test"]))
    print("training incidence rank =", np.linalg.matrix_rank(P["B"]))
    print("training home advantage =", P["home_adv"])

    robust_scores = [(lam, P["eval"](P["valid"], "robust", lam)["rmse"]) for lam in ROBUST_GRID]
    implicit_scores = [(a, P["eval"](P["valid"], "implicit", a)["rmse"]) for a in IMPLICIT_GRID]
    best_lam = min(robust_scores, key=lambda x: x[1])[0]
    best_alpha = min(implicit_scores, key=lambda x: x[1])[0]
    print("best validation robust lambda =", best_lam)
    print("best validation implicit alpha =", best_alpha)

    methods = [
        ("Electrical", "electrical", None),
        ("Robust", "robust", best_lam),
        ("Implicit", "implicit", best_alpha),
    ]
    raw = {}
    print("\nTEST TABLE")
    print("method       RMSE       MAE        mean L1    mean L2^2")
    for label, method, param in methods:
        stat, err, W = P["eval"](P["test"], method, param, return_raw=True)
        raw[label] = (err, W)
        print(
            f"{label:10s}  {stat['rmse']:.7f}  {stat['mae']:.7f}  "
            f"{stat['l1']:.7f}  {stat['l2']:.7f}"
        )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for first, second in [("Robust", "Electrical"), ("Robust", "Implicit")]:
        ea = raw[first][0]
        eb = raw[second][0]
        diffs = []
        for _ in range(20000):
            ii = rng.integers(0, len(ea), len(ea))
            diffs.append(np.sqrt(np.mean(ea[ii] ** 2)) - np.sqrt(np.mean(eb[ii] ** 2)))
        lo, med, hi = np.quantile(diffs, [0.025, 0.5, 0.975])
        print(f"bootstrap {first}-{second}: {lo:.7f}, {med:.7f}, {hi:.7f}")

    # Calibrated perturbation diagnostics used in the manuscript. Estimate the
    # observation-noise scale with residual degrees of freedom M-rank(B), then
    # add its exact conditional variance sigma_hat^2 ||w||_2^2 to each held-out
    # squared error. The coordinatewise bias radius at eta=.25 is .25 ||w||_1.
    theta_ls = np.linalg.pinv(P["B"]) @ ((P["train"].hg - P["train"].ag).to_numpy(float) - P["home_adv"])
    train_resid = ((P["train"].hg - P["train"].ag).to_numpy(float) - P["home_adv"]) - P["B"] @ theta_ls
    sigma_hat = float(np.sqrt(np.sum(train_resid**2) / (len(train_resid) - np.linalg.matrix_rank(P["B"]))))
    print("residual sigma_hat =", sigma_hat)
    for label in ["Electrical", "Robust", "Implicit"]:
        err, W = raw[label]
        expected_rmse = float(np.sqrt(np.mean(err**2) + sigma_hat**2 * np.mean(np.sum(W*W, axis=1))))
        bias_radius = float(0.25 * np.mean(np.sum(np.abs(W), axis=1)))
        print(f"stress {label}: iid_expected_rmse={expected_rmse:.7f}, box_eta_.25_radius={bias_radius:.7f}")

    e = P["eval"](P["test"], "electrical")
    r = P["eval"](P["test"], "robust", best_lam)
    i = P["eval"](P["test"], "implicit", best_alpha)
    assert best_lam == 0.02
    assert abs(best_alpha - 0.0464158883) < 1e-12
    assert abs(e["rmse"] - 1.8612541517140195) < 5e-7
    assert abs(r["rmse"] - 1.8316728570398009) < 5e-7
    assert abs(i["rmse"] - 1.916713111058086) < 5e-7
    assert abs(sigma_hat - 1.349414287489426) < 5e-10
    print("EPL_EXPERIMENT_CHECKS_PASSED")


if __name__ == "__main__":
    main()
