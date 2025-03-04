import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve, norm

#prelab

def lineEval(a, x0, y0, x1, y1):
    return y0 + ((y1 - y0)/(x1 - x0))*(a - x0)

# 3.2 

def test_3_2():
    f = lambda x: 1.0/(1.0 + (10.0*x)**2)
    A, B = -1, 1
    N = 10
    nE = 200
    xE = np.linspace(A, B, nE)
    yE = evalLin(xE, nE, A, B, f, N)
    fEx = f(xE)
    e = yE - fEx
    print("3.2 (Linear Spline) L2 error =", norm(e))
    plt.plot(xE, fEx, 'r-', label='f(x)=1/(1+(10x)^2)')
    plt.plot(xE, yE, 'bo--', label='lin spline')
    plt.legend()
    plt.show()
    plt.plot(xE, np.abs(e), 'ro-')
    plt.show()

# 3.3 

def evalLin(xE, nE, A, B, f, N):
    xI = np.linspace(A, B, N+1)
    yE = np.zeros(nE)
    for j in range(N):
        xL = xI[j]
        xR = xI[j+1]
        ind = np.where((xE >= xL) & (xE <= xR))[0]
        fL = f(xL)
        fR = f(xR)
        for i in ind:
            xNow = xE[i]
            yE[i] = lineEval(xNow, xL, fL, xR, fR)
    return yE

def testLin():
    f = lambda x: np.exp(x)
    A, B, N = 0, 1, 10
    nE = 100
    xE = np.linspace(A, B, nE)
    yE = evalLin(xE, nE, A, B, f, N)
    fEx = f(xE)
    e = yE - fEx
    print("3.3 (Linear Spline) L2 error =", norm(e))
    plt.plot(xE, fEx, 'r-', label='exp(x)')
    plt.plot(xE, yE, 'bo--', label='lin spline')
    plt.legend()
    plt.show()
    plt.plot(xE, np.abs(e), 'ro-')
    plt.show()

def makeNat(xI, yI):
    N = len(xI) - 1
    h = np.zeros(N)
    for i in range(N):
        h[i] = xI[i+1] - xI[i]
    b = np.zeros(N-1)
    for i in range(1, N):
        diff = ((yI[i+1] - yI[i]) / h[i]) - ((yI[i] - yI[i-1]) / h[i-1])
        b[i-1] = diff / (h[i] + h[i-1])
    M = np.zeros((N-1, N-1))
    for i in range(N-1):
        M[i, i] = 2.0
        if i > 0:
            M[i, i-1] = h[i] / (h[i] + h[i-1])
        if i < (N-2):
            M[i, i+1] = h[i+1] / (h[i+1] + h[i])
    alpha = solve(M, 6.0 * b)
    A = np.zeros(N+1)
    for i in range(1, N):
        A[i] = alpha[i-1]
    B = np.zeros(N)
    C = np.zeros(N)
    for i in range(N):
        B[i] = yI[i] - (A[i]*(h[i]**2))/6.0
        C[i] = yI[i+1] - (A[i+1]*(h[i]**2))/6.0
    return A, B, C

def evalLoc(xV, xL, xR, AL, AR, BL, CR):
    h = xR - xL
    yV = np.zeros_like(xV)
    for i in range(len(xV)):
        xx = xV[i]
        dxL = xR - xx
        dxR = xx - xL
        t1 = (AL/(6.0*h))*(dxL**3)
        t2 = (AR/(6.0*h))*(dxR**3)
        t3 = BL*(dxL/h)
        t4 = CR*(dxR/h)
        yV[i] = t1 + t2 + t3 + t4
    return yV

def evalCub(xE, xI, A, B, C):
    N = len(xI) - 1
    yE = np.zeros_like(xE)
    for j in range(N):
        xL = xI[j]
        xR = xI[j+1]
        ind = np.where((xE >= xL) & (xE <= xR))[0]
        xV = xE[ind]
        yV = evalLoc(xV, xL, xR, A[j], A[j+1], B[j], C[j])
        yE[ind] = yV
    return yE

def testCub():
    f = lambda x: np.exp(x)
    Aint, Bint, N = 0, 1, 10
    xI = np.linspace(Aint, Bint, N+1)
    yI = f(xI)
    nE = 100
    xE = np.linspace(Aint, Bint, nE)
    a, b, c = makeNat(xI, yI)
    yE = evalCub(xE, xI, a, b, c)
    fEx = f(xE)
    e = yE - fEx
    print("3.4 (Cubic Spline) L2 error =", norm(e))
    plt.plot(xE, fEx, 'r-', label='exp(x)')
    plt.plot(xE, yE, 'bo--', label='cubic spline')
    plt.legend()
    plt.show()
    plt.semilogy(xE, np.abs(e), 'ro--')
    plt.show()

# 3.4

def test_3_4():
    f = lambda x: 1.0/(1.0 + (10.0*x)**2)
    A, B = -1, 1
    N = 10
    nE = 200
    xI = np.linspace(A, B, N+1)
    yI = f(xI)
    xE = np.linspace(A, B, nE)
    Acoef, Bcoef, Ccoef = makeNat(xI, yI)
    yE = evalCub(xE, xI, Acoef, Bcoef, Ccoef)
    fEx = f(xE)
    e = yE - fEx
    print("3.4 (Cubic Spline) for 1/(1+(10x)^2), L2 error =", norm(e))
    plt.plot(xE, fEx, 'r-', label='f(x)=1/(1+(10x)^2)')
    plt.plot(xE, yE, 'bo--', label='cubic spline')
    plt.legend()
    plt.show()
    plt.plot(xE, np.abs(e), 'ro--')
    plt.show()

if __name__ == "__main__":

    test_3_2()
    testLin()
    testCub()
    test_3_4()
