import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import quad

# prelab
def eval_legendre(n, x):
    p = np.zeros(n+1)
    if n >= 0: p[0] = 1
    if n >= 1: p[1] = x
    for m in range(1, n):
        p[m+1] = ((2*m+1)*x*p[m] - m*p[m-1]) / (m+1)
    return p

# lab
def eval_legendre_expansion(f, a, b, w, n, x):
    p_at_x = eval_legendre(n, x)
    pval = 0
    for j in range(n+1):
        def phi_j(t): return eval_legendre(n, t)[j]
        def phi_j_sq(t): return (phi_j(t))**2 * w(t)
        norm_fac, _ = quad(phi_j_sq, a, b)
        def num_func(t): return phi_j(t)*f(t)*w(t)
        aj_num, _ = quad(num_func, a, b)
        aj = aj_num / norm_fac
        pval += aj * p_at_x[j]
    return pval

def driver():
    a, b = -1, 1
    w = lambda x: 1
    n = 2
    N = 500
    xeval = np.linspace(a, b, N+1)
    
    # f(x) = exp(x)
    f1 = lambda x: math.exp(x)
    pval1 = np.array([eval_legendre_expansion(f1, a, b, w, n, xx) for xx in xeval])
    fex1 = np.array([f1(xx) for xx in xeval])
    
    plt.figure()
    plt.plot(xeval, fex1, 'r-', label='f(x) = exp(x)')
    plt.plot(xeval, pval1, 'b--', label='Legendre Approx')
    plt.legend()
    plt.show()
    
    # f(x) = 1/(1 + x^2)
    f2 = lambda x: 1/(1 + x**2)
    pval2 = np.array([eval_legendre_expansion(f2, a, b, w, n, xx) for xx in xeval])
    fex2 = np.array([f2(xx) for xx in xeval])
    
    plt.figure()
    plt.plot(xeval, fex2, 'r-', label='f(x) = 1/(1+x^2)')
    plt.plot(xeval, pval2, 'b--', label='Legendre Approx')
    plt.legend()
    plt.show()

# additional exercise

def eval_chebyshev(n,x):
    p=np.zeros(n+1)
    if n>=0:p[0]=1
    if n>=1:p[1]=x
    for k in range(1,n):
        p[k+1]=2*x*p[k]-p[k-1]
    return p

def eval_chebyshev_expansion(f,a,b,w,n,x):
    t=eval_chebyshev(n,x)
    s=0
    for j in range(n+1):
        def T_j(u):return eval_chebyshev(n,u)[j]
        def T_j_sq(u):return (T_j(u))**2*w(u)
        nm,_=quad(T_j_sq,a,b)
        def numerator(u):return T_j(u)*f(u)*w(u)
        nj,_=quad(numerator,a,b)
        s+=nj/nm*t[j]
    return s

def driver_chebyshev():
    a,b=-1,1
    w=lambda x:1/np.sqrt(1-x**2)
    n=2
    xeval=np.linspace(a,b,501)
    f=lambda x:math.exp(x)
    pval=np.zeros(len(xeval))
    for i,xx in enumerate(xeval):
        pval[i]=eval_chebyshev_expansion(f,a,b,w,n,xx)
    fex=np.array([f(xx) for xx in xeval])
    
    plt.figure()
    plt.plot(xeval,fex,'r-',xeval,pval,'b--')
    plt.show()
    
    err=np.abs(pval-fex)
    plt.figure()
    plt.semilogy(xeval,err,'ro-')
    plt.show()
    
if __name__ == '__main__':
    driver()
    driver_chebyshev()
