import numpy as np

def driver():
    A = np.array([[2, 3], [4, 1]])
    v = np.array([5, 6])
    
    mv_custom = matrixVectorMultiplication(A, v, len(v))
    
    mv_numpy = np.matmul(A, v)
    
    print('Custom implementation result:', mv_custom)
    print('NumPy implementation result:', mv_numpy)
    
    import time
    start_custom = time.time()
    for _ in range(10000):
        _ = matrixVectorMultiplication(A, v, len(v))
    end_custom = time.time()
    
    start_numpy = time.time()
    for _ in range(10000):
        _ = np.matmul(A, v)
    end_numpy = time.time()
    
    print(f"Custom code time: {end_custom - start_custom:.6f} seconds")
    print(f"NumPy time: {end_numpy - start_numpy:.6f} seconds")

def matrixVectorMultiplication(A, v, n):
    mv = np.zeros(n)
    for i in range(n):
        for j in range(n):
            mv[i] += A[i][j] * v[j]
    return mv

driver()