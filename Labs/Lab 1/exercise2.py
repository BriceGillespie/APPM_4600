import numpy as np
def driver():

    A = np.array([[2, 3], [4, 1]])  # 2x2 matrix
    v = np.array([5, 6])  # 2x1 vector
    n = len(v)  
    
    result = matrixVectorMultiplication(A, v, n)
    

    print('The matrix-vector multiplication result is:', result)
    return

def matrixVectorMultiplication(A, v, n):

    mv = np.zeros(n)  # Initialize result vector
    for i in range(n):
        for j in range(n):
            mv[i] += A[i][j] * v[j]
    return mv

driver()
