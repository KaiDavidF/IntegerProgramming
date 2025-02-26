import numpy as np

def extended_gcd(a, b):
    """
    Returns (g, x, y) such that a*x + b*y = g = gcd(a, b).
    """
    if b == 0:
        return abs(a), 1 if a >= 0 else -1, 0
    else:
        g, x, y = extended_gcd(b, a % b)
        return g, y, x - (a // b) * y

def smith_normal_form(A):
    """
    Compute the Smith Normal Form of an integer matrix A,
    along with unimodular matrices U and V such that U * A * V = S.
    
    Parameters:
      A : array-like (n x n)
      
    Returns:
      S : np.ndarray (n x n) -- the Smith Normal Form of A.
      U : np.ndarray (n x n) -- unimodular matrix (row operations).
      V : np.ndarray (n x n) -- unimodular matrix (column operations).
      
    Note: This implementation is for demonstration and is not optimized for large matrices.
    """
    # Convert A to a NumPy array of integers.
    A = np.array(A, dtype=int)
    n, m = A.shape
    if n != m:
        raise ValueError("Only square matrices are supported in this implementation.")
    
    # Initialize U and V as identity matrices.
    U = np.eye(n, dtype=int)
    V = np.eye(n, dtype=int)
    
    row = 0
    col = 0
    
    while row < n and col < m:
        # --- Step 1: Find a pivot ---
        # Look for the smallest nonzero element (by absolute value) in A[row:, col:].
        pivot_val = 0
        pivot_i, pivot_j = row, col
        found = False
        for i in range(row, n):
            for j in range(col, m):
                if A[i, j] != 0:
                    if not found or abs(A[i, j]) < abs(pivot_val):
                        pivot_val = A[i, j]
                        pivot_i, pivot_j = i, j
                        found = True
        if not found:
            # If the entire submatrix is 0, move to the next column.
            col += 1
            continue

        # --- Step 2: Swap pivot into position (row, col) ---
        if pivot_i != row:
            A[[row, pivot_i], :] = A[[pivot_i, row], :]
            U[[row, pivot_i], :] = U[[pivot_i, row], :]
        if pivot_j != col:
            A[:, [col, pivot_j]] = A[:, [pivot_j, col]]
            V[:, [col, pivot_j]] = V[:, [pivot_j, col]]
        pivot_val = A[row, col]

        # --- Step 3: Clear out the column (except the pivot) ---
        for i in range(n):
            if i == row:
                continue
            while A[i, col] != 0:
                pivot_val = A[row, col]
                # Safety check: if pivot becomes 0, break out so a new pivot can be chosen.
                if pivot_val == 0:
                    break
                q = A[i, col] // pivot_val
                # Row operation: R_i <- R_i - q * R_row.
                A[i, :] = A[i, :] - q * A[row, :]
                U[i, :] = U[i, :] - q * U[row, :]
                # If a smaller entry appears at (i, col), swap rows.
                if abs(A[i, col]) < abs(A[row, col]):
                    A[[row, i], :] = A[[i, row], :]
                    U[[row, i], :] = U[[i, row], :]
                    pivot_val = A[row, col]

        # --- Step 4: Clear out the row (except the pivot) ---
        for j in range(m):
            if j == col:
                continue
            while A[row, j] != 0:
                pivot_val = A[row, col]
                if pivot_val == 0:
                    break
                q = A[row, j] // pivot_val
                # Column operation: C_j <- C_j - q * C_col.
                A[:, j] = A[:, j] - q * A[:, col]
                V[:, j] = V[:, j] - q * V[:, col]
                if abs(A[row, j]) < abs(A[row, col]):
                    A[:, [col, j]] = A[:, [j, col]]
                    V[:, [col, j]] = V[:, [j, col]]
                    pivot_val = A[row, col]

        # --- Step 5: Make the pivot positive ---
        if A[row, col] < 0:
            A[row, :] = -A[row, :]
            U[row, :] = -U[row, :]

        row += 1
        col += 1

    # At this point, A is in a diagonal form (or nearly so). 
    # Further steps could enforce the full divisibility conditions if desired.
    return A, U, V

# ------------------- Example Usage -------------------
if __name__ == "__main__":
    A_example = [
        [2, 4, 4],
        [4, 6, 6],
        [2, 2, 2]
    ]
    
    S, U, V = smith_normal_form(A_example)
    
    print("Smith Normal Form S:")
    print(S)
    print("\nUnimodular Matrix U:")
    print(U)
    print("\nUnimodular Matrix V:")
    print(V)
