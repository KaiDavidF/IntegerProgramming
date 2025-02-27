from PyNormaliz import * 
from latticeVectorspace import intersectLattices

def intersectCones(cone1, cone2):
    intersection_ineq = cone1.SupportHyperplanes()+cone2.SupportHyperplanes()
    L = intersectLattices(cone1.HilbertBasis(), cone2.HilbertBasis())
    if L is None:
        return None
    C = Cone(inequalities = intersection_ineq, lattice=L)
    return C


C1 = Cone(cone_and_lattice=[[1,1],[1,2],[1,3],[1,4]])
# C2 is NOT full linear!
C2 = Cone(cone_and_lattice=[[1,1],[1,2],[1,4]])

C = intersectCones(C1,C2)

print(C.HilbertBasis())









