import sympy as sp
from dualLattice import lattice_intersection_generators
from vectorSpaces import intersection_basis, generators_to_H
from hsnf import smith_normal_form

def intersectLatticeVectorspace(L, vectorSpace):
    H = generators_to_H(vectorSpace)
    # the vectorSpace is Q^n.
    if not H:
        return L

    cols = [sp.Matrix(b) for b in L]
    B = sp.Matrix.hstack(*cols)
    
    n = B.shape[1]
    HB = H * B
    zero_matrix = sp.zeros(n - HB.shape[0], n)
    combined_matrix = sp.Matrix.vstack(HB, zero_matrix)

    S, _, U = smith_normal_form(combined_matrix)
    S = sp.Matrix(S)
    U = sp.Matrix(U)

    BU = B*U
    BU = BU.T
    
    # take the last k entries which correspond to zeros in S.
    k = S.shape[0] - S.rank()
    if k == 0:
        return []
    last_k_vectors = BU[-k:, :]
  
    return [last_k_vectors]


def intersectLattices(L1, L2):
    vectorSpace = intersection_basis(L1, L2)
    L = lattice_intersection_generators(L1, L2)
    return intersectLatticeVectorspace(L=L, vectorSpace=vectorSpace)

if __name__ == "__main__":
    L1 = [[2,1],[0,2]]
    L2 = [[1,0],[0,1]]
    print(intersectLattices(L1, L2))
    
    
    
    