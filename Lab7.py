import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
import time

def f(x):
    return 1 / (1 + (10 * x) ** 2)

#vandermonde
def vandermonde_interp(xint, xtrg):
    n = len(xint) - 1
    fi = f(xint)
    V = np.vander(xint, increasing=True)
    c = np.linalg.solve(V, fi)
    return np.polyval(c[::-1], xtrg)

#lagrange
def lagrange_interp(xint, xtrg):
    poly = lagrange(xint, f(xint))
    return poly(xtrg)

#Newton
def newton_interp(xint, xtrg):
    n = len(xint)
    coef = np.copy(f(xint))
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j - 1]) / (xint[j:n] - xint[j - 1])
    
    def newton_poly(x):
        result = coef[-1]
        for j in range(n - 2, -1, -1):
            result = result * (x - xint[j]) + coef[j]
        return result
    
    return np.array([newton_poly(x) for x in xtrg])

# Compute and plot the interpolations with hopefully accurate timing
#(newton should be the fastest but its actually vandermonde in the outputs so idk)
def plot_error(N_values, method):
    x_plot = np.linspace(-1, 1, 1000)
    for N in N_values:
        xint = np.linspace(-1, 1, N + 1)
        
        #high precision timer
        num_trials = 100
        start_time = time.perf_counter()
        for _ in range(num_trials):
            if method == 'vandermonde':
                y_interp = vandermonde_interp(xint, x_plot)
            elif method == 'lagrange':
                y_interp = lagrange_interp(xint, x_plot)
            elif method == 'newton':
                y_interp = newton_interp(xint, x_plot)
        elapsed_time = (time.perf_counter() - start_time) / num_trials
        
        max_error = np.max(np.abs(f(x_plot) - y_interp))
        
        print(f'Method: {method.capitalize()}, N={N}, Max Error={max_error:.5e}, Time={elapsed_time:.8f} sec')
        
        plt.figure()
        plt.plot(x_plot, np.abs(f(x_plot) - y_interp), label=f'N={N}')
        plt.legend()
        plt.yscale('log')
        plt.xlabel('x')
        plt.ylabel('Absolute Error')
        plt.title(f'Interpolation Error ({method.capitalize()} Method, N={N})')
        plt.show()

#nodes
def chebyshev_nodes(N):
    return np.cos((2 * np.arange(1, N + 1) - 1) * np.pi / (2 * N))

N_values = [1, 5, 10]
plot_error(N_values, 'vandermonde')
plot_error(N_values, 'lagrange')
plot_error(N_values, 'newton')

#chebyshev nodes
N_cheb = [5, 10, 15, 20]
x_plot = np.linspace(-1, 1, 1000)
for N in N_cheb:
    xint = chebyshev_nodes(N)
    
    #timer
    num_trials = 100
    start_time = time.perf_counter()
    for _ in range(num_trials):
        y_interp = newton_interp(xint, x_plot)
    elapsed_time = (time.perf_counter() - start_time) / num_trials
    
    max_error = np.max(np.abs(f(x_plot) - y_interp))
    
    print(f'Chebyshev Nodes, N={N}, Max Error={max_error:.5e}, Time={elapsed_time:.8f} sec')
    
    plt.figure()
    plt.plot(x_plot, np.abs(f(x_plot) - y_interp), label=f'N={N}')
    plt.legend()
    plt.yscale('log')
    plt.xlabel('x')
    plt.ylabel('Absolute Error')
    plt.title(f'Interpolation Error with Chebyshev Nodes (Newton Method, N={N})')
    plt.show()