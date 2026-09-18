#!/usr/bin/env python3
"""Core exact/numerical checks for manuscript R14."""
from __future__ import annotations
from fractions import Fraction
import mpmath as mp
import numpy as np
import sympy as sp
from scipy.optimize import minimize_scalar

mp.mp.dps=90

def solve_multiscale_u(r):
    r=mp.mpf(str(r)); a=mp.sqrt(2)/2
    s1=mp.e**(-a/r); q=mp.e**(-1/r)
    def f1(u1,u2,u3): return mp.sinh(u1)+s1*mp.e**(2*u1)
    def f2(u1,u2,u3): return mp.sinh(u2)+q*mp.e**(2*u2)
    def f3(u1,u2,u3):
        return mp.mpf('0.25')*mp.e**u3*(mp.e**u1+mp.e**(-u1)+mp.e**u2+mp.e**(-u2)+s1*mp.e**(2*u1)+q*mp.e**(2*u2))-1
    return mp.findroot((f1,f2,f3),(-s1,-q,-(s1+q)/4),tol=mp.mpf('1e-70'))

LENGTHS=np.array([2.,4.,16.]); COUNTS=np.array([1.,9.,40.]); L=1/36
ROBUST=np.array([6/23,17/23,0.]); ROBUST_F=Fraction(148,207)

def F(x):
    x=np.asarray(x,float)
    return float(np.sum((LENGTHS/COUNTS)*x*x)+L*(LENGTHS@x)**2)

def split(s):
    w=COUNTS*np.sinh(16*s/LENGTHS)
    return w/w.sum()

def best_ratio():
    r=minimize_scalar(lambda s:F(split(s)),bounds=(1e-10,8),method='bounded',options={'xatol':1e-14})
    return r.x,split(r.x),r.fun/float(ROBUST_F)

def poly_certificate():
    z,y=sp.symbols('z y')
    P=(110080*z**14-330240*z**12+283904*z**10-17408*z**8-12160*z**7-81908*z**6+18240*z**5+35572*z**4+18680*z**3-4277*z**2-12380*z+66640)
    Q=sp.Rational(32,207)*P
    py=sp.Poly(sp.expand(Q.subs(z,1+y)),y); n=14
    power=[sp.Rational(py.nth(i)) for i in range(n+1)]
    bern=[sp.simplify(sum(power[i]*sp.binomial(k,i)/sp.binomial(n,i) for i in range(k+1))) for k in range(n+1)]
    c=[sp.Rational(py.nth(i)) for i in range(n+1)]
    return min(bern),sp.simplify(c[5]-(abs(c[1])+abs(c[2])+abs(c[3])))

def fixed_lambda_ratios(lam=0.1):
    from scipy.optimize import minimize
    out=[]
    for m in [1,4,16,64,256,1024]:
        def J(x): return np.sum((LENGTHS/COUNTS)*x*x)/m+lam*(LENGTHS@x)**2
        cons={'type':'eq','fun':lambda x:x.sum()-1}
        rr=minimize(J,np.array([.8,.15,.05]),bounds=[(0,1)]*3,constraints=cons,method='SLSQP',options={'ftol':1e-14,'maxiter':1000})
        ii=minimize_scalar(lambda s:J(split(s)),bounds=(1e-10,20),method='bounded',options={'xatol':1e-13})
        out.append((m,ii.fun/rr.fun))
    return out

def main():
    print('INCOMMENSURATE LP')
    a=mp.sqrt(2)/2
    for r in [.15,.1,.075,.05]:
        u1,u2,u3=solve_multiscale_u(r); rr=mp.mpf(str(r))
        print(r,mp.nstr(u1/(-mp.e**(-a/rr)),10),mp.nstr(u2/(-mp.e**(-1/rr)),10),mp.nstr(u3,14))
    s,x,ratio=best_ratio()
    print('\nLOCAL-MISSPECIFICATION RISK')
    print('robust',ROBUST,'F=',F(ROBUST),'best implicit',x,'ratio',ratio)
    b,m=poly_certificate(); print('min Bernstein',b,'tail margin',m)
    assert b==sp.Rational(18473472,23023)
    assert m==sp.Rational(95656480,23)
    assert ratio>1.5
    print('\nFIXED-LAMBDA ILLUSTRATION')
    vals=fixed_lambda_ratios()
    for mm,rr in vals: print(mm,rr)
    assert vals[-1][1] < vals[0][1]
    print('ALL_R14_CORE_CHECKS_PASSED')
if __name__=='__main__': main()
