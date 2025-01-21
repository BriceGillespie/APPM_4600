import numpy as np

def driver():
    # Define orthogonal vectors
    x = np.array([1, 0])
    y = np.array([0, 1])
    n = len(x)  # Length of the vectors
    
    # Evaluate the dot product
    dp = dotProduct(x, y, n)
    
    # Print the result
    print('The dot product is:', dp)
    return

def dotProduct(x, y, n):
    # Compute the dot product of vectors x and y
    dp = 0.0
    for j in range(n):
        dp += x[j] * y[j]
    return dp

driver()