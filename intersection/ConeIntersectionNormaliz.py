from PyNormaliz import * 

# This solves all of the problems, LOL

def intersection(cone1, cone2):
    intersection_ineq = cone1.SupportHyperplanes()+cone2.SupportHyperplanes()
    C = Cone(inequalities = intersection_ineq)
    return C

C1 = Cone(cone_and_lattice=[[1,1],[1,3]])
C2 = Cone(cone_and_lattice=[[1,1],[1,3]])

C = intersection(C1, C2)

print(C.HilbertBasis())
