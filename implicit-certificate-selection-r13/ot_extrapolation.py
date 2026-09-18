#!/usr/bin/env python3
"""High-precision two-point continuation test for degenerate 4x4 empirical OT.

The gauge-fixed potentials are solved by damped Newton in 140-digit arithmetic.
The table matches the continuation experiment in manuscript R13.
"""
from __future__ import annotations

import mpmath as mp

mp.mp.dps = 140

X = [mp.mpf("0"), mp.mpf("0.89016367"), mp.mpf("2.35894637"), mp.mpf("3.11632981")]
Y = [mp.mpf("0.20744063"), mp.mpf("0.92092571"), mp.mpf("2.36876553"), mp.mpf("2.88776868")]
N = 4
C = [[(X[i] - Y[j]) ** 2 for j in range(N)] for i in range(N)]
A = [mp.mpf(1) / N] * N


def residual_hessian(z, eps):
    f = z[:N]
    g = list(z[N:]) + [mp.mpf("0")]  # gauge g_4=0
    P = [[mp.e ** ((f[i] + g[j] - C[i][j]) / eps) for j in range(N)] for i in range(N)]

    grad = [sum(P[i]) - A[i] for i in range(N)]
    grad += [sum(P[i][j] for i in range(N)) - A[j] for j in range(N - 1)]

    d = 2 * N - 1
    H = mp.matrix(d, d)
    for i in range(N):
        H[i, i] = sum(P[i]) / eps
    for j in range(N - 1):
        H[N + j, N + j] = sum(P[i][j] for i in range(N)) / eps
    for i in range(N):
        for j in range(N - 1):
            H[i, N + j] = P[i][j] / eps
            H[N + j, i] = P[i][j] / eps
    return mp.matrix(grad), H


def solve(eps, z0=None, tol=mp.mpf("1e-110"), maxit=80):
    z = mp.matrix([0] * (2 * N - 1)) if z0 is None else mp.matrix(z0)
    for it in range(maxit):
        grad, H = residual_hessian(z, eps)
        gn = max(abs(q) for q in grad)
        if gn < tol:
            return z, it, gn
        step = mp.lu_solve(H, -grad)
        t = mp.mpf(1)
        accepted = False
        for _ in range(80):
            candidate = z + t * step
            gn2 = max(abs(q) for q in residual_hessian(candidate, eps)[0])
            if gn2 < gn:
                z = candidate
                accepted = True
                break
            t /= 2
        if not accepted:
            raise RuntimeError("Newton line search failed")
    raise RuntimeError("Newton failed to converge")


def inf_norm(a, b):
    return max(abs(a[i] - b[i]) for i in range(len(a)))


def main():
    eps_grid = [mp.mpf(x) for x in ["0.4", "0.2", "0.1", "0.05", "0.025", "0.0125"]]
    solutions = {}
    z = None
    for eps in eps_grid:
        z, it, gn = solve(eps, z)
        solutions[str(eps)] = z
        print("continuation", eps, "iterations", it, "residual", mp.nstr(gn, 5))

    print("\nTARGET TABLE")
    print("eps      warm error        extrapolated error   Newton warm  Newton extrap")
    rows = []
    for eps in eps_grid[2:]:
        true = solutions[str(eps)]
        warm = solutions[str(2 * eps)]
        old = solutions[str(4 * eps)]
        pred = mp.matrix([mp.mpf("1.5") * warm[i] - mp.mpf("0.5") * old[i] for i in range(len(warm))])
        warm_err = inf_norm(warm, true)
        pred_err = inf_norm(pred, true)
        _, iw, _ = solve(eps, warm)
        _, ip, _ = solve(eps, pred)
        rows.append((eps, warm_err, pred_err, iw, ip))
        print(
            f"{float(eps):0.4f}  {mp.nstr(warm_err, 11):>15}  "
            f"{mp.nstr(pred_err, 11):>18}  {iw:11d}  {ip:13d}"
        )

    expected = [
        (0.1000, 1.62910579469e-1, 2.65878053584e-2, 9, 7),
        (0.0500, 7.12391580580e-2, 1.02161316768e-2, 8, 7),
        (0.0250, 3.46766127658e-2, 9.42966263162e-4, 8, 6),
        (0.0125, 1.73286832239e-2, 9.62315898738e-6, 8, 6),
    ]
    for row, exp in zip(rows, expected):
        eps, we, pe, iw, ip = row
        assert abs(float(eps) - exp[0]) < 1e-15
        assert abs(float(we) - exp[1]) < 5e-12
        assert abs(float(pe) - exp[2]) < 5e-12
        assert (iw, ip) == (exp[3], exp[4])
    print("OT_EXTRAPOLATION_CHECKS_PASSED")


if __name__ == "__main__":
    main()
