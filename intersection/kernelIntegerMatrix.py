import sympy as sp
from hsnf import smith_normal_form
import numpy as np

"""
    The following code computes a basis of a kernel of a matrix equation Ax = 0, where A is integer and x also is integer.
"""

#`(D, L, R)` satisfy ``D = np.dot(L, np.dot(M, R))
def findKernel(A):
    # Ensure A is a numpy array
    A = np.array(A).astype(int)
    
    # Get the number of rows and columns
    rows, cols = A.shape
    assert rows <= cols
    # If the matrix is not square, add zero rows or columns to make it square
    if rows < cols:
        A = np.vstack([A, np.zeros((cols - rows, cols), dtype=int)])
    
    S, _, U = smith_normal_form(A)
    # V * A * U = S

    # Find the coordinates where S is zero
    zero_coords = [i for i in range(S.shape[0]) if S[i, i] == 0]
    k = len(zero_coords)
    
    # Get the last k columns of U
    kernel_basis = U[:, -k:]
    
    return kernel_basis



if __name__ == "__main__":
    A = sp.Matrix([[1,2],[2,4]])
    kernel = findKernel(A)
    print(kernel)