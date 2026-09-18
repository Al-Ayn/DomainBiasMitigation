#!/usr/bin/env python3
"""Real empirical OT epsilon-halving benchmark for manuscript R14.

Uses sklearn's built-in handwritten-digits data.  For each prespecified class pair,
take the first 100 samples per class, scale pixels to [0,1], use mean squared
Euclidean cost divided by its pairwise median, solve the eps=0.05 problem, compute
the exact tangent of the gauge-fixed dual potentials by implicit differentiation,
and compare ordinary epsilon-scaling with the tangent predictor at eps=0.025.
"""
from __future__ import annotations
import numpy as np
from scipy.special import logsumexp
from sklearn.datasets import load_digits

PAIRS=[(0,1),(0,7),(1,7),(2,3),(3,8),(4,9),(5,6),(6,8),(7,9),(2,8)]
N=100
EPS0=0.05
TARGET=0.025
TOLS=(1e-6,1e-9)

def sinkhorn_potentials(C,eps,f0=None,g0=None,tol=1e-9,maxit=20000):
    n,m=C.shape
    a=np.full(n,1/n); b=np.full(m,1/m)
    f=np.zeros(n) if f0 is None else np.array(f0,float).copy()
    g=np.zeros(m) if g0 is None else np.array(g0,float).copy()
    s=g[-1]; g-=s; f+=s
    loga=np.log(a); logb=np.log(b)
    for it in range(maxit):
        f=eps*(loga-logsumexp((g[None,:]-C)/eps,axis=1))
        g=eps*(logb-logsumexp((f[:,None]-C)/eps,axis=0))
        s=g[-1]; g-=s; f+=s
        logP=(f[:,None]+g[None,:]-C)/eps
        P=np.exp(logP)
        res=max(np.max(np.abs(P.sum(1)-a)),np.max(np.abs(P.sum(0)-b)))
        if res<tol:
            return f,g,P,it+1,res
    raise RuntimeError(f"Sinkhorn did not converge: eps={eps}, tol={tol}, res={res}")

def tangent(C,eps,f,g,P):
    n,m=C.shape
    a=np.full(n,1/n); b=np.full(m,1/m)
    Pm=P[:,:m-1]
    J=np.block([[np.diag(a),Pm],[Pm.T,np.diag(b[:m-1])]])/eps
    logP=(f[:,None]+g[None,:]-C)/eps
    F_eps=np.r_[ -(P*logP).sum(axis=1)/eps,
                  -(P[:,:m-1]*logP[:,:m-1]).sum(axis=0)/eps ]
    hp=np.linalg.solve(J,-F_eps)
    return hp[:n],np.r_[hp[n:],0.0]

def main():
    data=load_digits()
    X=data.data/16.0; y=data.target
    rows=[]
    for c0,c1 in PAIRS:
        A=X[y==c0][:N]; B=X[y==c1][:N]
        C=((A[:,None,:]-B[None,:,:])**2).mean(axis=2)
        C=C/np.median(C)
        f,g,P,_,_=sinkhorn_potentials(C,EPS0,tol=1e-12)
        fp,gp=tangent(C,EPS0,f,g,P)
        f_tan=f-(EPS0-TARGET)*fp
        g_tan=g-(EPS0-TARGET)*gp
        rec=[f"{c0}-{c1}"]
        for tol in TOLS:
            iw=sinkhorn_potentials(C,TARGET,f,g,tol=tol)[3]
            it=sinkhorn_potentials(C,TARGET,f_tan,g_tan,tol=tol)[3]
            rec.extend([iw,it])
        rows.append(rec)
    print("pair  warm_1e-6 tan_1e-6 warm_1e-9 tan_1e-9")
    for r in rows: print(f"{r[0]:>4s} {r[1]:10d} {r[2]:8d} {r[3]:10d} {r[4]:8d}")
    arr=np.array([r[1:] for r in rows],int)
    print("median",np.median(arr,axis=0))
    print("mean savings",np.mean(arr[:,0]-arr[:,1]),np.mean(arr[:,2]-arr[:,3]))
    assert np.all(arr[:,0]>arr[:,1]) and np.all(arr[:,2]>arr[:,3])
    assert tuple(np.median(arr,axis=0))==(26.0,21.0,53.0,46.5)
    print("OT_TANGENT_DIGITS_CHECKS_PASSED")
if __name__=="__main__": main()
