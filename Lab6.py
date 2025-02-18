import numpy as np
import math

# before lab: finite difference approximations
def bLab():
    x = math.pi/2
    e = -1.0
    hs = 0.01 * (2.0 ** (-np.arange(10)))
    print("h         FwdErr      CtrErr")
    for h in hs:
        fwd = (math.cos(x+h) - math.cos(x)) / h
        ctr = (math.cos(x+h) - math.cos(x-h)) / (2 * h)
        print(f"{h: .2e}   {abs(fwd-e): .2e}   {abs(ctr-e): .2e}")

# norm function
def norm(v):
    return np.linalg.norm(v)

# function f: system of equations
def f(x):
    return np.array([4*x[0]**2 + x[1]**2 - 4, x[0] + x[1] - math.sin(x[0]-x[1])])

# exact jacobian of f
def J(x):
    return np.array([[8*x[0], 2*x[1]],
                     [1 - math.cos(x[0]-x[1]), 1 + math.cos(x[0]-x[1])]])

# approximate jacobian using finite differences
def aJ(fun, x, c=1e-5):
    n = len(x)
    A = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = c if abs(x[j]) < 1e-14 else c * abs(x[j])
        A[:, j] = (fun(x+e) - fun(x)) / e[j]
    return A

# slacker newton: adaptive jacobian update
def slkNewton(fun, Jac, x0, tol=1e-10, m=50):
    x = x0.copy()
    Ai = np.linalg.inv(Jac(x))
    prev = None
    for i in range(m):
        s = -Ai.dot(fun(x))
        xn = x + s
        if norm(s) < tol:
            return xn, i+1
        if prev is not None and norm(s) > 0.8 * norm(prev):
            Ai = np.linalg.inv(Jac(xn))
        prev = s
        x = xn
    return x, m

# newton with approximate jacobian using finite differences
def newtonAJ(fun, x0, c=1e-5, tol=1e-10, m=50):
    x = x0.copy()
    for i in range(m):
        A = aJ(fun, x, c)
        s = np.linalg.solve(A, -fun(x))
        xn = x + s
        if norm(s) < tol:
            return xn, i+1
        x = xn
    return x, m

# hybrid newton: combine slacker and approximate jacobian, adapt constant c
def hybNewton(fun, x0, h0=1e-3, tol=1e-10, m=50):
    x = x0.copy()
    c = h0
    A = aJ(fun, x, c)
    Ai = np.linalg.inv(A)
    prev = None
    for i in range(m):
        s = -Ai.dot(fun(x))
        xn = x + s
        if norm(s) < tol:
            return xn, i+1
        if prev is not None and norm(s) > 0.8 * norm(prev):
            c /= 2
            A = aJ(fun, xn, c)
            Ai = np.linalg.inv(A)
        prev = s
        x = xn
    return x, m

if __name__ == "__main__":
    print("BEFORE LAB")
    bLab()
    
    x0 = np.array([1.0, 0.0])
    
    print("\nSLACKER NEWTON")
    sol, iters = slkNewton(f, J, x0)
    print("Sol:", sol, "f:", f(sol), "Iterations:", iters)
    
    print("\nNEWTON APPROX JAC (c = 1e-7)")
    sol, iters = newtonAJ(f, x0, c=1e-7)
    print("Sol:", sol, "f:", f(sol), "Iterations:", iters)
    
    print("\nNEWTON APPROX JAC (c = 1e-3)")
    sol, iters = newtonAJ(f, x0, c=1e-3)
    print("Sol:", sol, "f:", f(sol), "Iterations:", iters)
    
    print("\nHYBRID NEWTON")
    sol, iters = hybNewton(f, x0, h0=1e-3)
    print("Sol:", sol, "f:", f(sol), "Iterations:", iters)

