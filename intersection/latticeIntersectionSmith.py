import sympy as sp
import numpy as np
from kernelIntegerMatrix import findKernel


def intersectLatticeSmith(L1, L2):
    n = len(L1)
    L1 = sp.Matrix(np.array(L1).T)
    L2 = sp.Matrix(np.array(L2).T)
    L = L1.row_join(-L2)
    kernel = findKernel(L)
    intersection = np.dot(np.array(L1), np.array(kernel[:n]))
    return intersection.T

    
    
if __name__ == "__main__":
    L1 = [[1,0,0], [0,2,0]]
    L2 = [[0,4,0], [1,0,0]]
    L = intersectLatticeSmith(L1, L2)
    newBase = [l.tolist() for l in L]
    print(newBase)