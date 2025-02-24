import numpy as np
from sympy import Matrix, lcm_list

# We use pycddlib for converting between V- and H-representations of cones.
# (You can replace this with any double description method.)
try:
    import cdd
except ImportError:
    cdd = None

###############################################################################
# Classes representing the two types of objects.
###############################################################################

class Lattice:
    def __init__(self, A, proj):
        """
        We represent a lattice L ⊆ ℤ^d in the form
            L = { x in ℤ^d : A x = 0 }
        together with a projection (really a choice of basis for ker A)
        so that L ≅ ℤ^r.
        
        Parameters
        ----------
        A : array_like
            A matrix (with integer entries) so that L = {x : A x = 0}.
        proj : array_like
            A matrix whose columns give an integer basis for ker(A).
        """
        self.A = np.array(A)
        self.proj = np.array(proj)
        
    def intersect(self, other):
        """
        Intersection of two lattices:
        
            L1 = { x : A1 x = 0 } and L2 = { x : A2 x = 0 } 
            ⇒ L1 ∩ L2 = { x : [A1; A2] x = 0 }.
            
        (Here we leave the projection as the identity on the essential space.)
        """
        A_new = np.vstack([self.A, other.A])
        proj_new = np.eye(self.proj.shape[1])
        return Lattice(A_new, proj_new)
    
    def __str__(self):
        return f"Lattice(A={self.A}, proj={self.proj})"

class Cone:
    def __init__(self, A):
        """
        We represent a cone in ℝ^d as
            C = { x in ℝ^d : A x >= 0 }.
            
        Parameters
        ----------
        A : array_like
            A matrix such that x in C if and only if A x >= 0.
        """
        self.A = np.array(A)
        
    def intersect(self, other):
        """
        Intersection of two cones:
            C1 = {x: A1 x >= 0} and C2 = {x: A2 x >= 0} 
            ⇒ C1 ∩ C2 = {x: [A1; A2] x >= 0}.
        """
        A_new = np.vstack([self.A, other.A])
        return Cone(A_new)
    
    def __str__(self):
        return f"Cone(A={self.A})"

class PeriodicSet:
    def __init__(self, lattice: Lattice, cone: Cone):
        """
        A periodic set is an intersection of a lattice L and a cone C:
            S = L ∩ C.
        """
        self.lattice = lattice
        self.cone = cone
        
    def intersect(self, other):
        """
        Intersection of two periodic sets.
        """
        new_lat = self.lattice.intersect(other.lattice)
        new_cone = self.cone.intersect(other.cone)
        return PeriodicSet(new_lat, new_cone)
    
    def __str__(self):
        return f"PeriodicSet(lattice={self.lattice}, cone={self.cone})"

###############################################################################
# Conversion routines between generating sets (the V–representation) and
# the matrix representation (the H–representation).
###############################################################################

def lattice_from_generators(generators):
    """
    Given generators p1,...,pk in ℕ^d (each as a list or 1D array), 
    we want to produce an H–representation for the lattice
        L = {x in ℤ^d : A x = 0}
    together with a projection (i.e. a basis for ker A).

    Here is one way:
      1. Form the matrix P whose columns are the p_i.
      2. Then any linear relation that holds on all generators
         is a row vector in the left nullspace of P. That is, if
             A P = 0,
         then every x in the lattice spanned by the p_i satisfies A x = 0.
      3. Compute A as a (rational) basis of the left nullspace, clear denominators,
         and compute a basis for ker(A) as the projection.
    """
    P = Matrix(np.array(generators).T)
    # Compute a basis for the left nullspace (i.e. nullspace of P^T).
    A_rows = P.T.nullspace()
    if not A_rows:
        A = Matrix([[0]*P.shape[0]])
    else:
        A = Matrix.hstack(*A_rows).T
    # Clear denominators so that A is integer.
    if A:
        denoms = [abs(elem.q) for elem in A]
        lcm_val = lcm_list(denoms) if denoms else 1
        A = A * lcm_val
    # Next, compute a basis for the kernel of A.
    proj_basis = A.nullspace()
    if not proj_basis:
        proj = Matrix.eye(P.shape[0])
    else:
        proj = Matrix.hstack(*proj_basis)
    return Lattice(np.array(A.tolist(), dtype=int),
                   np.array(proj.tolist(), dtype=int))

def generators_from_lattice(lat: Lattice):
    """
    Given a lattice in H–representation (A, proj), return a generating set,
    namely the columns of the projection matrix.
    """
    proj = Matrix(lat.proj)
    gens = []
    for i in range(proj.shape[1]):
        gens.append(list(proj.col(i)))
    return gens

def cone_from_generators(generators):
    """
    Given generators p1,...,pk in ℕ^d, we may define the cone as
        C = { x in ℝ^d : x = λ1 p1 + ... + λk pk, λi >= 0 }.
    This is the V–representation.
    
    To get an H–representation (inequalities), one standard method is to use the
    double description method. Here we use the pycddlib package.
    """
    if cdd is None:
        raise ImportError("Please install pycddlib to perform cone conversions.")
    d = len(generators[0])
    # cdd requires that each generator be given in homogeneous form.
    # For a ray we use the form [0, p1, ..., pd] (the first coordinate is 0 for rays).
    rows = []
    for g in generators:
        rows.append([0] + list(g))
    mat = cdd.Matrix(rows, number_type='fraction')
    mat.rep_type = cdd.RepType.GENERATOR
    poly = cdd.Polyhedron(mat)
    H = poly.get_inequalities()
    # H is a matrix whose rows are [b, a1, ..., ad] and the inequality is
    #    b + a1 x1 + ... + ad xd >= 0.
    # In our case we want homogeneous inequalities so we only keep rows with b = 0.
    A = []
    for row in H:
        if row[0] == 0:
            A.append(list(row[1:]))
    return Cone(np.array(A, dtype=int))

def generators_from_cone(cone: Cone):
    """
    Converts an H–representation of a cone to a V–representation (its extreme rays)
    using pycddlib.
    """
    if cdd is None:
        raise ImportError("Please install pycddlib to perform cone conversions.")
    A = cone.A
    # cdd expects an inequality matrix with rows [b, a1, ..., ad] for
    #   b + a1 x1 + ... + ad xd >= 0.
    # Here we have A x >= 0 so we form [0, -a1, ..., -ad].
    rows = []
    for a in A:
        rows.append([0] + list(-np.array(a)))
    H = cdd.Matrix(rows, number_type='fraction')
    H.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.Polyhedron(H)
    G = poly.get_generators()
    gens = []
    for row in G:
        if row[0] == 0:  # only include rays
            gens.append(list(row[1:]))
    return gens

def periodic_set_from_generators(generators):
    """
    If the same generators p1,...,pk ∈ ℕ^d are used to define both the lattice
    (by integer combinations) and the cone (by nonnegative combinations), then the
    periodic set is defined as
         S = L ∩ C.
    This function converts a list of generators into our (A,proj) and (B,≥0) matrices.
    """
    L = lattice_from_generators(generators)
    C = cone_from_generators(generators)
    return PeriodicSet(L, C)

###############################################################################
# Example usage
###############################################################################

if __name__=='__main__':
    # Let us say we have generators in ℕ^3:
    gens = [[1, 0, 0],
            [0, 1, 0],
            [1, 1, 1]]
    
    # Build the periodic set S = L ∩ C (with L and C both generated by these p_i)
    PS = periodic_set_from_generators(gens)
    print("Periodic Set:")
    print(PS)
    
    # The lattice L is stored via an equality matrix A and a projection.
    print("\nLattice representation:")
    print("A =", PS.lattice.A)
    print("Projection (basis for L):", PS.lattice.proj)
    
    # The cone C is stored by an inequality matrix.
    print("\nCone representation:")
    print("Inequalities (A x >= 0):")
    print(PS.cone.A)
    
    # One may convert back to the original generating set:
    lat_gens = generators_from_lattice(PS.lattice)
    print("\nRecovered lattice generators:", lat_gens)
    
    cone_gens = generators_from_cone(PS.cone)
    print("Recovered cone generators:", cone_gens)
