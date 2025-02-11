import numpy as np

# Define the function and derivatives.
def f(x):
    return np.exp(x**2 + 7*x - 30) - 1

def df(x):
    return np.exp(x**2 + 7*x - 30) * (2*x + 7)

def ddf(x):
    return np.exp(x**2 + 7*x - 30) * ((2*x + 7)**2 + 2)

# Bisection Method
def bisection_method(f, a, b, tol=1e-8, max_iter=100):
    fa = f(a)
    fb = f(b)
    if fa * fb >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    
    iterations = 0
    eval_count = 2  # f(a) and f(b) already evaluated
    while iterations < max_iter:
        c = (a + b) / 2.0
        fc = f(c)
        eval_count += 1
        iterations += 1
        if abs(fc) < tol or (b - a) / 2 < tol:
            return c, iterations, eval_count
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    raise Exception("Bisection did not converge")

# Newton's Method
def newton_method(f, df, x0, tol=1e-8, max_iter=100):
    x = x0
    iterations = 0
    eval_count = 0
    while iterations < max_iter:
        fx = f(x)
        eval_count += 1
        if abs(fx) < tol:
            return x, iterations, eval_count
        dfx = df(x)
        eval_count += 1
        if dfx == 0:
            raise ZeroDivisionError("Derivative zero encountered in Newton's method")
        x = x - fx / dfx
        iterations += 1
    raise Exception("Newton's method did not converge")

# Hybrid Method: Bisection + Newton
def is_in_newton_basin(x, f, df, ddf, basin_tol=1.0):
    dfx = df(x)
    if dfx == 0:
        return False
    return abs(f(x) * ddf(x)) / (dfx**2) < basin_tol

def hybrid_method(f, df, ddf, a, b, tol=1e-8, max_iter_bis=50, max_iter_newton=50):
    fa = f(a)
    fb = f(b)
    if fa * fb >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    
    iterations_bis = 0
    eval_count_bis = 2  # already evaluated f(a) and f(b)
    c = None
    while iterations_bis < max_iter_bis:
        c = (a + b) / 2.0
        fc = f(c)
        eval_count_bis += 1
        iterations_bis += 1
        if is_in_newton_basin(c, f, df, ddf):
            break
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    else:
        raise Exception("Bisection did not find a point in the basin of Newton's method")
    
    x_newton, iterations_newton, eval_count_newton = newton_method(f, df, c, tol, max_iter_newton)
    return x_newton, iterations_bis, iterations_newton, eval_count_bis, eval_count_newton

if __name__ == "__main__":
    tol = 1e-8
    print("Solving f(x)=exp(x^2+7x-30)-1 with root at x=3\n")
    
    # Pure Bisection Method
    a, b = 2, 4.5
    try:
        root_bis, iter_bis, eval_bis = bisection_method(f, a, b, tol=tol, max_iter=100)
        print("Bisection Method:")
        print(f"  Approximated root = {root_bis:.10f}")
        print(f"  Iterations = {iter_bis}")
        print(f"  Function evaluations = {eval_bis}\n")
    except Exception as e:
        print("Bisection method failed:", e)
    
    # Pure Newton's Method starting at x0 = 4.5
    x0 = 4.5
    try:
        root_newton, iter_newton, eval_newton = newton_method(f, df, x0, tol=tol, max_iter=100)
        print("Newton's Method:")
        print(f"  Approximated root = {root_newton:.10f}")
        print(f"  Iterations = {iter_newton}")
        print(f"  Function and derivative evaluations = {eval_newton}\n")
    except Exception as e:
        print("Newton's method failed:", e)
    
    # Hybrid Method
    try:
        (root_hybrid, iter_bis_hybrid, iter_newton_hybrid, 
         eval_bis_hybrid, eval_newton_hybrid) = hybrid_method(f, df, ddf, a, b, tol=tol)
        total_iter_hybrid = iter_bis_hybrid + iter_newton_hybrid
        total_eval_hybrid = eval_bis_hybrid + eval_newton_hybrid
        print("Hybrid Method (Bisection + Newton):")
        print(f"  Approximated root = {root_hybrid:.10f}")
        print(f"  Bisection iterations = {iter_bis_hybrid}, Newton iterations = {iter_newton_hybrid}")
        print(f"  Total iterations = {total_iter_hybrid}")
        print(f"  Function evaluations (bisection) = {eval_bis_hybrid}")
        print(f"  Function & derivative evaluations (Newton) = {eval_newton_hybrid}")
        print(f"  Total evaluations = {total_eval_hybrid}\n")
    except Exception as e:
        print("Hybrid method failed:", e)
