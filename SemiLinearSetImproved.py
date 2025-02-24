import numpy as np
from minimalPoints import getMinimalPoints

def buildVector(coefficients, vectors):
    """
    Given:
      - coefficients: a list or 1D np.array of length k,
      - vectors: a list/array of row vectors (each a 1D array of length n).
    Returns their linear combination (a 1D array of length n).
    """

    # (Old code did an explicit loop; we preserve that logic.)
    vector = np.zeros(vectors[0].shape[0])
    for i in range(len(coefficients)):
        vector += coefficients[i] * vectors[i]
    return vector

class LinearSet:
    def __init__(self, offset, periodSet):
        """
        offset: a row vector (a 1D array of length n)
        periodSet: a 2D array of shape (k, n) where each row is a period generator.
        """
        self.offset = np.array(offset).ravel()  # ensure 1D row vector
        self.periodSet = np.array(periodSet)
        if self.periodSet.ndim == 1:
            self.periodSet = self.periodSet.reshape(1, -1)
        if self.periodSet.shape[1] != self.offset.size:
            raise ValueError("offset dimension (%d) does not match periodSet generator dimension (%d)" %
                             (self.offset.size, self.periodSet.shape[1]))
        self.dimension = self.offset.size
        print(f"offset = {self.offset}")
        print(f"periodSet = {self.periodSet}")
        
    def addVector(self, v):
        """
        Computes the Minkowski sum: L(b, P) + ℕ₀·v.
        
        (Internally, the new period set is constructed by appending v as a new generator
         and then applying an intersection with the natural numbers set.)
         
        v must be given as a row vector. (E.g. [3, -1], not a column vector.)
        """
        # Ensure v is a row vector (as a 2D array with one row)
        v = np.atleast_2d(v)
        if v.shape[0] != 1:
            raise ValueError("v must be a row vector.")
            
        # The natural numbers set: offset zero and period set = identity.
        # (np.eye returns an array whose rows are the standard basis vectors.)
        NaturalNumbers = LinearSet(offset=np.zeros(self.dimension, dtype=int),
                                   periodSet=np.eye(self.dimension))
        # Add v as an extra generator.
        # Since self.periodSet is stored with generators in rows, simply stack v:
        P = np.concatenate((self.periodSet, v), axis=0)  # new P has shape ((k+1), n)
        b = self.offset  # a 1D array
        
        print("NaturalNumbers.periodSet:\n", NaturalNumbers.periodSet)
        # In the old code the intersection routine expected period sets in column‐form.
        # Here we pass LinearSet(b, P) but inside intersect we will transpose the period sets.
        resultC, resultP = intersect(NaturalNumbers, LinearSet(b, P))
        
        # In the old code the number of generators was taken as P[0].size
        # (since P was stored with generators as columns). Now, with row vectors, the number
        # of generators is P.shape[0].
        num_generators = P.shape[0]
        coefficientsC = [point[:num_generators] for point in resultC]
        coefficientsP = [point[:num_generators] for point in resultP]
        
        newPeriodSet = []
        newOffsetSet = []
        # In the old code, P.transpose() was passed to buildVector because the generators were columns.
        # Now that our generators are rows, we simply pass P.
        for coeff in coefficientsP:
            newPeriodSet.append(buildVector(coeff, P))
        for coeff in coefficientsC:
            newOffsetSet.append(buildVector(coeff, P) + b)
        
        return np.vstack(newOffsetSet), np.vstack(newPeriodSet)

def intersect(l1: LinearSet, l2: LinearSet):
    """
    Computes the intersection used in the addVector method.
    
    In the original code, period sets were stored as columns so that
      A = [ l2.periodSet   |  -l1.periodSet ]
    In our unified representation the period sets are stored as rows.
    We therefore use their transposes so that the generators become columns.
    """

    A = np.concatenate((l2.periodSet.T, -l1.periodSet.T), axis=1)
    b = l1.offset - l2.offset
    C = getMinimalPoints(A=A, b=b, noZero=False)
    P = getMinimalPoints(A=A, b=np.zeros_like(b), noZero=True)
    return C, P

def project(points, dimension):
    """Projects a list/array of points (each a row vector) to its first 'dimension' coordinates."""
    return [point[:dimension] for point in points]

class SemiLinearSet:
    def __init__(self, baseSets=None, periodSets=None, linearSet=None):
        """
        Either initialize from a single linear set or supply lists of base sets and period sets.
        In our unified representation:
          - Each base offset is a row vector (1D array).
          - Each period set is a 2D array whose rows are generators.
        """
        if linearSet is not None:
            self.baseSets = [linearSet.offset]  # list of 1D arrays
            self.periodSets = [linearSet.periodSet]  # list of 2D arrays
        elif baseSets is not None and periodSets is not None:
            self.baseSets = baseSets
            self.periodSets = periodSets
        else:
            self.baseSets = []
            self.periodSets = []
            
    def add(self, C, P):
        """Add new components (lists of offsets and period sets) to the semilinear set."""
        self.baseSets += C
        self.periodSets += P
    
    def intersect(self, other):
        """Computes (and prints) the intersection between two semilinear sets."""
        intersectionSet = SemiLinearSet([], [])
        for b, P in zip(self.baseSets, self.periodSets):
            for c, Q in zip(other.baseSets, other.periodSets):
                C_prime, P_prime = intersect(LinearSet(b, P), LinearSet(c, Q))
                print("------")
                print("Intersection offsets (C_prime):")
                print(C_prime)
                print("Intersection periods (P_prime):")
                print(P_prime)
                print("------")
                # (One could add these new components to intersectionSet here.)
        return intersectionSet
    
    def addVector(self, v):
        """
        Adds ℕ₀·v to the semilinear set by applying addVector to each linear component.
        Returns a new semilinear set built from the new components.
        """
        new_base_list = []
        new_period_list = []
        for b, P in zip(self.baseSets, self.periodSets):
            linearSet = LinearSet(b, P)
            C, Q = linearSet.addVector(v=v)
            # C and Q are returned as 2D arrays (each row a vector);
            # convert them to lists of 1D arrays (for offsets) and 2D arrays (for period sets).
            for off, per in zip(C, Q):
                new_base_list.append(off)         # off is a 1D array
                new_period_list.append(per)         # per is a row vector (2D array with one row)
        return SemiLinearSet(baseSets=new_base_list, periodSets=new_period_list)
        
    def union(self, other):
        newBase = self.baseSets + other.baseSets
        newPeriod = self.periodSets + other.periodSets
        return SemiLinearSet(baseSets=newBase, periodSets=newPeriod)

if __name__ == "__main__":
    # === Old functionality example, now with unified representation ===
    # In the old code, the period set was given as columns:
    #     P = np.array([[0],
    #                   [1]])
    # Now we represent a period generator as a row vector.
    P = np.array([[0,1]])   # 1 generator of length 2 (row vector)
    print("P shape:", P.shape)  # Expect (1,2)
    
    # In the old code, the offset was given as:
    #     b = np.array([[0,3]])
    # We now use a row vector (or a 1D array)
    b = np.array([0,3])
    
    # Create the linear set with unified representations.
    lSet = LinearSet(offset=b, periodSet=P)
    intersect(lSet, lSet)
    
    
    # # v must be a row vector; here we use [3, -1] (not a column vector).
    v = np.array([3, -1])
    
    print("Original period set:")
    print(lSet.periodSet)
    
    # Compute the new offsets and period set after adding v.
    B, PeriodSet = lSet.addVector(v)
    print("New offsets after adding v:")
    for row in B:
        print(row)
    print("New period set after adding v:")
    for row in PeriodSet:
        print(row)
    
    lSet = LinearSet(offset = [3,2], periodSet=[[0,1],[3,0]])
    print(lSet.addVector([5,-1]))
    
    # newBList = []
    # newPList = []
    # for b in B:
    #     newLSet = LinearSet(b, PeriodSet)
    #     print(b)
    #     print(P)
    #     print()
    #     newB, newP = newLSet.addVector(np.array([5, -1]))
    #     bList = [row for row in newB]
    #     pList = [row for row in newP]
    #     for row in bList:
    #         print(row)
    #         newBList.append(row)
    #     for row in pList:
    #         newPList.append(row)
        
    # print(newBList)
    # print()
    # print(newPList)
    
