#!/usr/bin/env python3
from sympy import Matrix

def intersection_basis(basis1, basis2):
    d = len(basis1[0])
    V_mat = Matrix.hstack(*[Matrix(vec) for vec in basis1])
    W_mat = Matrix.hstack(*[Matrix(vec) for vec in basis2])
    A = V_mat.row_join(-W_mat)
    ns = A.nullspace()
    r = V_mat.shape[1]
    intersection_vectors = []
    for vec in ns:
        a = vec[:r, 0]
        x = V_mat * a
        intersection_vectors.append(x)
    if not intersection_vectors:
        return []
    X = Matrix.hstack(*intersection_vectors)
    col_basis = X.columnspace()
    return [list(vec) for vec in col_basis]

def generators_to_H(generators):
    G = Matrix.hstack(*[Matrix(vec) for vec in generators])
    H_basis = G.T.nullspace()
    if not H_basis:
        return Matrix.zeros(0, G.rows)
    H = Matrix.vstack(*[v.T for v in H_basis])
    return H

if __name__ == "__main__":
    V_basis = [[1, 0, 0], [0, 1, 0]]
    W_basis = [[1, 1, 0], [0, 0, 1]]
    inter_basis = intersection_basis(V_basis, W_basis)
    print("Basis for V ∩ W:")
    for vec in inter_basis:
        print(vec)
    generators = [[1,0,0]]
    H = generators_to_H(generators)
    print("\nMatrix H such that V = { x : H * x = 0 }:")
    print(H)
    x = Matrix([1, 2, 3])
    print("\nH * [1, 2, 3] =")
    print(H * x)
