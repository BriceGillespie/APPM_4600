import numpy as np

# Exercise 2.2.1
def ooc(p_vals, p_fix):
    errs = np.abs(p_vals - p_fix)
    n = len(errs) - 2
    alphas = []
    lambdas = []

    for i in range(n):
        alpha = np.log(errs[i + 2] / errs[i + 1]) / np.log(errs[i + 1] / errs[i])
        alphas.append(alpha)

    alpha_avg = np.mean(alphas)
    for i in range(n):
        lambdas.append(errs[i + 1] / (errs[i] ** alpha_avg))

    lambda_avg = np.mean(lambdas)

    return alpha_avg, lambda_avg

# Exercise 2.2.2
def fpi(g, p0, tol=1e-10, max_iter=1000):
    p_vals = [p0]
    for _ in range(max_iter):
        p_next = g(p_vals[-1])
        p_vals.append(p_next)
        if abs(p_next - p_vals[-2]) < tol:
            break
    return np.array(p_vals)

def g(x):
    return (10 / (x + 4)) ** 0.5

p_fix = 1.3652300134140976
p0 = 1.5

p_fpi = fpi(g, p0)
num_iter_fpi = len(p_fpi) - 1
alpha_fpi, lambda_fpi = ooc(p_fpi, p_fix)

print(f"number of iterations to converge: {num_iter_fpi}")
print(f"FPI order of convergence: {alpha_fpi}")
print(f"FPI asymptotic error constant: {lambda_fpi}")

# Exercise 3.2
def aitken(p_vals, tol=1e-10, max_iter=1000):
    p_acc = []
    for i in range(len(p_vals) - 2):
        num = (p_vals[i+1] - p_vals[i])**2
        den = p_vals[i+2] - 2*p_vals[i+1] + p_vals[i]
        if abs(den) < tol:
            break
        p_next = p_vals[i] - num / den
        p_acc.append(p_next)
        if len(p_acc) > 1 and abs(p_acc[-1] - p_acc[-2]) < tol:
            break
    return np.array(p_acc)

p_aitken = aitken(p_fpi)
alpha_aitken, lambda_aitken = ooc(p_aitken, p_fix)

print(f"Aitkens method: {len(p_aitken)-1} iterations, order: {alpha_aitken}, Constant: {lambda_aitken}")

# Exercise 3.3
# For input need p_0, function, tolerance and max iterations
# Ouput computed fixed p and iterations
def steffensen(g, p0, tol=1e-10, max_iter=1000):
    p_vals = [p0]
    for _ in range(max_iter):
        a = p_vals[-1]
        b = g(a)
        c = g(b)
        den = c - 2 * b + a
        if abs(den) < tol:
            break
        p_next = a - ((b - a) ** 2) / den
        p_vals.append(p_next)
        if abs(p_next - p_vals[-2]) < tol:
            break
    return np.array(p_vals)

p_steff = steffensen(g, p0)
alpha_steff, lambda_steff = ooc(p_steff, p_fix)

print(f"Steffensens Method: {len(p_steff)-1} iterations, order: {alpha_steff}, Constant: {lambda_steff}")

