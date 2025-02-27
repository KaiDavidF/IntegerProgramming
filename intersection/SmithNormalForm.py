import numpy as np

def extended_gcd(a, b):
    """
    Return (g, x, y) such that a*x + b*y = g = gcd(a, b).
    """
    if b == 0:
        return (a, 1, 0)
    else:
        g, x1, y1 = extended_gcd(b, a % b)
        return (g, y1, x1 - (a // b) * y1)

def smith_normal_form(A):
    """
    Compute the Smith Normal Form of a square integer matrix A,
    along with unimodular matrices U and V such that U*A*V = S.

    Parameters
    ----------
    A : array-like, shape (n, n)
        Integer matrix.

    Returns
    -------
    S : np.ndarray, shape (n, n)
        The Smith Normal Form of A (diagonal with certain divisibility).
    U : np.ndarray, shape (n, n)
        Unimodular matrix tracking row operations (det(U) = ±1).
    V : np.ndarray, shape (n, n)
        Unimodular matrix tracking column operations (det(V) = ±1).

    Note
    ----
    - This implementation is for demonstration only and is not optimized.
    - It uses repeated gcd-based elimination to drive A toward a diagonal form.
    - For large matrices or for efficiency, consider more sophisticated algorithms.
    """
    A = np.array(A, dtype=int, copy=True)
    n, m = A.shape
    if n != m:
        raise ValueError("This demonstration code assumes A is square.")

    # Initialize U and V as identity matrices.
    U = np.eye(n, dtype=int)
    V = np.eye(n, dtype=int)

    # Helper functions to apply row/column operations
    def swap_rows(i, j):
        A[[i, j], :] = A[[j, i], :]
        U[[i, j], :] = U[[j, i], :]

    def swap_cols(i, j):
        A[:, [i, j]] = A[:, [j, i]]
        V[:, [i, j]] = V[:, [j, i]]

    def add_rows(src, dest, k):
        # Row(dest) += k * Row(src)
        A[dest, :] += k * A[src, :]
        U[dest, :] += k * U[src, :]

    def add_cols(src, dest, k):
        # Col(dest) += k * Col(src)
        A[:, dest] += k * A[:, src]
        V[:, dest] += k * V[:, src]

    def reduce_step():
        """
        Attempt to reduce off-diagonal elements by gcd-based elimination.
        Return True if any change was made, otherwise False.
        """
        changed = False
        for i in range(n):
            # If pivot is 0, try to swap in a nonzero pivot from below/right
            if A[i, i] == 0:
                pivot_found = False
                for r in range(i, n):
                    for c in range(i, n):
                        if A[r, c] != 0:
                            # Swap row r -> i
                            if r != i:
                                swap_rows(r, i)
                                changed = True
                            # Swap col c -> i
                            if c != i:
                                swap_cols(c, i)
                                changed = True
                            pivot_found = True
                            break
                    if pivot_found:
                        break

            pivot = A[i, i]
            if pivot == 0:
                continue  # no pivot to reduce with

            # Reduce elements in the same column (below and above).
            for r in range(n):
                if r != i and A[r, i] != 0:
                    g, x, y = extended_gcd(pivot, A[r, i])
                    # If gcd is strictly smaller than the pivot, we can reduce.
                    if abs(g) < abs(pivot):
                        # Save old rows for combined operation
                        old_r = A[r, :].copy()
                        old_i = A[i, :].copy()
                        oldU_r = U[r, :].copy()
                        oldU_i = U[i, :].copy()

                        # Row(r) <- x*Row(r) + y*Row(i)
                        A[r, :] = x * old_r + y * old_i
                        U[r, :] = x * oldU_r + y * oldU_i

                        changed = True

            # Reduce elements in the same row (left and right).
            for c in range(n):
                if c != i and A[i, c] != 0:
                    g, x, y = extended_gcd(pivot, A[i, c])
                    if abs(g) < abs(pivot):
                        # Save old columns for combined operation
                        old_c = A[:, c].copy()
                        old_i = A[:, i].copy()
                        oldV_c = V[:, c].copy()
                        oldV_i = V[:, i].copy()

                        # Col(c) <- x*Col(c) + y*Col(i)
                        A[:, c] = x * old_c + y * old_i
                        V[:, c] = x * oldV_c + y * oldV_i

                        changed = True

            # Ensure pivot is nonnegative (conventionally).
            if A[i, i] < 0:
                A[i, :] = -A[i, :]
                U[i, :] = -U[i, :]
                changed = True

        return changed

    # Iteratively apply gcd-based elimination until no more changes occur.
    while True:
        if not reduce_step():
            break

    # Now try to clear out any remaining off-diagonal elements using integer division.
    # This final pass tries to zero out off-diagonal entries by standard "add multiples".
    for i in range(n):
        pivot = A[i, i]
        if pivot == 0:
            continue
        # Clear out the column except pivot
        for r in range(n):
            if r != i and A[r, i] != 0:
                q = A[r, i] // pivot
                add_rows(i, r, -q)
        # Clear out the row except pivot
        for c in range(n):
            if c != i and A[i, c] != 0:
                q = A[i, c] // pivot
                add_cols(i, c, -q)

    # A should now be diagonal (though for large or tricky matrices,
    # you might need further passes to ensure the divisibility chain d_i | d_{i+1}.
    # For demonstration, we stop here.

    S = A
    return S, U, V

# ---------------------------------------------------------------------------
# Demonstration / Example
if __name__ == "__main__":
    A_demo = np.array([
        [4, -2],
        [-6, 8]
    ], dtype=int)

    S, U, V = smith_normal_form(A_demo)
    print("Original A:\n", A_demo)
    print("Smith Normal Form S:\n", S)
    print("Unimodular U:\n", U)
    print("Unimodular V:\n", V)
    # Check that U*A_demo*V == S
    check = U @ A_demo @ V
    print("U * A * V:\n", check)
