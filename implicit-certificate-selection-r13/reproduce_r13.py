#!/usr/bin/env python3
"""Reproduce the main numerical and exact checks in manuscript R13."""
from __future__ import annotations
from fractions import Fraction
import mpmath as mp
import numpy as np
import sympy as sp
from scipy.optimize import minimize_scalar

mp.mp.dps = 90

def solve_multiscale_u(r: float):
    rmp = mp.mpf(str(r)); a = mp.sqrt(2)/2
    s1 = mp.e**(-a/rmp); q = mp.e**(-1/rmp)
    def f1(u1,u2,u3): return mp.sinh(u1) + s1*mp.e**(2*u1)
    def f2(u1,u2,u3): return mp.sinh(u2) + q*mp.e**(2*u2)
    def f3(u1,u2,u3):
        return mp.mpf("0.25")*mp.e**u3*(
            mp.e**u1+mp.e**(-u1)+mp.e**u2+mp.e**(-u2)
            +s1*mp.e**(2*u1)+q*mp.e**(2*u2))-1
    u1,u2,u3 = mp.findroot((f1,f2,f3),(-s1,-q,-(s1+q)/4),tol=mp.mpf("1e-70"))
    return u1,u2,u3,(u3+mp.mpf("0.25")*s1)/q

def sinkhorn_plan_mp(a,b,C,eps):
    aa=[mp.mpf(str(x)) for x in a]; bb=[mp.mpf(str(x)) for x in b]
    CC=[[mp.mpf(str(x)) for x in row] for row in C]; e=mp.mpf(str(eps))
    n,m=len(aa),len(bb)
    def equations(*z):
        f=list(z[:n]); g=list(z[n:])+[mp.mpf("0")]
        P=[[aa[i]*bb[j]*mp.e**((f[i]+g[j]-CC[i][j])/e) for j in range(m)] for i in range(n)]
        eq=[sum(P[i][j] for j in range(m))-aa[i] for i in range(n)]
        eq += [sum(P[i][j] for i in range(n))-bb[j] for j in range(m-1)]
        return tuple(eq)
    z0=tuple(mp.mpf("0") for _ in range(n+m-1))
    try:
        sol=mp.findroot(equations,z0,tol=mp.mpf("1e-65"),maxsteps=150)
    except Exception:
        sol=mp.findroot(equations,z0,tol=mp.mpf("1e-55"),maxsteps=400,solver="mdnewton")
    sol=list(sol); f=sol[:n]; g=sol[n:]+[mp.mpf("0")]
    return [[aa[i]*bb[j]*mp.e**((f[i]+g[j]-CC[i][j])/e) for j in range(m)] for i in range(n)]

def kl_entropic_cost(a,b,X,Y,eps):
    C=[[(mp.mpf(str(x))-mp.mpf(str(y)))**2 for y in Y] for x in X]
    P=sinkhorn_plan_mp(a,b,C,eps)
    aa=[mp.mpf(str(x)) for x in a]; bb=[mp.mpf(str(x)) for x in b]; e=mp.mpf(str(eps))
    val=mp.mpf("0")
    for i in range(len(a)):
        for j in range(len(b)):
            p=P[i][j]; val += C[i][j]*p
            if p != 0: val += e*p*mp.log(p/(aa[i]*bb[j]))
    return val

def sinkhorn_divergence_check(eps):
    a=[0.5,0.5]; b=[1/3,1/3,1/3]; X=[0,1]; Y=[0,0.5,1]
    cross=kl_entropic_cost(a,b,X,Y,eps)
    selfa=kl_entropic_cost(a,a,X,X,eps); selfb=kl_entropic_cost(b,b,Y,Y,eps)
    S=cross-(selfa+selfb)/2; OT0=mp.mpf(1)/12
    Delta=mp.log(3)/2-mp.log(2)/6; e=mp.mpf(str(eps))
    return (S-OT0)/e, S-OT0+e*Delta, Delta

LENGTHS=np.array([2.,4.,16.]); COUNTS=np.array([1.,9.,40.]); L=1/36
ROBUST_X=np.array([6/23,17/23,0.]); ROBUST_F_EXACT=Fraction(148,207)

def F_L(x):
    x=np.asarray(x,float)
    return float(np.sum((LENGTHS/COUNTS)*x*x)+L*(LENGTHS@x)**2)

def implicit_split_from_s(s):
    w=COUNTS*np.sinh(16*s/LENGTHS)
    return w/w.sum()

def implicit_A_from_s(s):
    return float(1/(2*np.sum(COUNTS*np.sinh(16*s/LENGTHS))))

def replicated_path_numerical_optimum():
    res=minimize_scalar(lambda s:F_L(implicit_split_from_s(s)),bounds=(1e-10,8),method="bounded",options={"xatol":1e-14})
    x=implicit_split_from_s(res.x); A=implicit_A_from_s(res.x)
    return res.x,A,x,res.fun/float(ROBUST_F_EXACT)

def exact_polynomial_certificate():
    z,y=sp.symbols("z y")
    P=(110080*z**14-330240*z**12+283904*z**10-17408*z**8-12160*z**7
       -81908*z**6+18240*z**5+35572*z**4+18680*z**3-4277*z**2-12380*z+66640)
    Q=sp.Rational(32,207)*P
    py=sp.Poly(sp.expand(Q.subs(z,1+y)),y); n=14
    power=[sp.Rational(py.nth(i)) for i in range(n+1)]
    bern=[sp.simplify(sum(power[i]*sp.binomial(k,i)/sp.binomial(n,i) for i in range(k+1))) for k in range(n+1)]
    c=[sp.Rational(py.nth(i)) for i in range(n+1)]
    return sp.factor(min(bern)), sp.factor(c[5]-(abs(c[1])+abs(c[2])+abs(c[3])))

def main():
    print("MULTISCALE LP")
    a=mp.sqrt(2)/2
    for r in [0.150,0.100,0.075,0.050]:
        u1,u2,u3,coeff=solve_multiscale_u(r); rr=mp.mpf(str(r))
        print(r, mp.nstr(u1/(-mp.e**(-a/rr)),9), mp.nstr(u2/(-mp.e**(-1/rr)),9),
              mp.nstr(u3,12), mp.nstr(coeff,10))
    print("\nSINKHORN DIVERGENCE")
    Delta=None
    for eps in [0.04,0.02,0.01]:
        slope,rem,Delta=sinkhorn_divergence_check(eps)
        print(eps, mp.nstr(slope,11), mp.nstr(rem,10))
    print("Delta =",mp.nstr(Delta,18))
    print("\nREPLICATED-PATH TOTAL RISK")
    print("robust split =",ROBUST_X)
    print("F_L robust =",F_L(ROBUST_X)," exact =",float(ROBUST_F_EXACT))
    s,A,x,ratio=replicated_path_numerical_optimum()
    print("best s =",s); print("best A=m*alpha^2 =",A)
    print("best implicit split =",x); print("risk ratio =",ratio)
    minb,margin=exact_polynomial_certificate()
    print("min Bernstein coefficient =",minb); print("tail positivity margin =",margin)
    assert minb==sp.Rational(18473472,23023)
    assert margin==sp.Rational(95656480,23)
    assert ratio>1.5
    print("ALL_R13_CORE_CHECKS_PASSED")

if __name__=="__main__":
    main()
