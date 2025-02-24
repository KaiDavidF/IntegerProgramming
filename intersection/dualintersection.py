import sympy
from sympy import Matrix, Rational, ilcm
from sympy.matrices.normalforms import hermite_normal_form

def lattice_basis_from_generators(vectors):
    """
    Given a list of integer vectors (possibly linearly dependent),
    compute an integral basis for the lattice they generate using the HNF.
    
    Parameters
    ----------
    vectors : list of lists/tuples
        A list of d-dimensional integer vectors.
    
    Returns
    -------
    basis : sympy.Matrix
        A d x r matrix whose columns form a basis for L(vectors).
    """
    # Create a matrix whose columns are the input vectors.
    M = Matrix(vectors).T
    # Compute the HNF of M.
    H = hermite_normal_form(M)
    basis_cols = []
    for j in range(H.shape[1]):
        col = H.col(j)
        if not col.is_zero_matrix:  # corrected: no parentheses
            basis_cols.append(col)
    if basis_cols:
        return Matrix.hstack(*basis_cols)
    else:
        return Matrix.zeros(M.rows, 0)

def lattice_intersection_generators(P, Q):
    """
    Computes a generating set (basis) for the intersection of two lattices L(P) and L(Q)
    using the dual lattice method from "LatticeAlgorithms-Dual-Intersection.pdf".
    
    This version works even when the input generating sets are not bases.
    
    Parameters
    ----------
    P : list of lists/tuples
        A list of d-dimensional integer vectors generating L(P).
    Q : list of lists/tuples
        A list of d-dimensional integer vectors generating L(Q).
        
    Returns
    -------
    gens : list of sympy.Matrix
        A list of column vectors forming a basis for L(P) ∩ L(Q).
    """
    # First, extract a basis for each lattice.
    P_basis = lattice_basis_from_generators(P)
    Q_basis = lattice_basis_from_generators(Q)
    
    if P_basis.shape[1] == 0 or Q_basis.shape[1] == 0:
        return []

    # Compute dual bases for L(P) and L(Q)
    D_P = P_basis * (P_basis.T * P_basis).inv()
    D_Q = Q_basis * (Q_basis.T * Q_basis).inv()

    # Concatenate dual bases horizontally
    D_all = D_P.row_join(D_Q)
    
    # Clear denominators by computing the LCM of all denominators
    denominators = [Rational(entry).q for entry in D_all]
    common_denom = ilcm(*denominators)
    
    D_all_int = D_all * common_denom
    D_all_int = D_all_int.applyfunc(lambda x: int(x))
    
    # Compute the HNF of the integer matrix
    H_int = hermite_normal_form(D_all_int)
    # Convert back to a rational matrix
    H = H_int.applyfunc(lambda x: Rational(x, common_denom))
    
    # Compute the dual of H to obtain a basis for L(P) ∩ L(Q)
    dual_H = H * (H.T * H).inv()
    
    # Return the columns of dual_H as a list.
    gens = [dual_H[:, i] for i in range(dual_H.shape[1])]
    return gens

# Example usage:
if __name__ == '__main__':
    P = [
        [1,1],
        [1,2],
        [1,3]
    ]
    
    gens = lattice_intersection_generators(P, P)
    for g in gens:
        print(g.flat())
