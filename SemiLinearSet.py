import numpy as np
from minimalPoints import getMinimalPoints

def buildVector(coefficients, vectors):
    vector = np.zeros(vectors[0].size)
    for i in range(len(coefficients)):
        vector += coefficients[i] * vectors[i]
    return vector

    
class LinearSet:
    def __init__(self, offset, periodSet):
        self.offset = offset
        self.periodSet = periodSet
        self.dimension = offset.size
        
    def addVector(self, v):
        NaturalNumbers = LinearSet(offset=np.zeros(self.dimension, dtype=int), periodSet=np.eye(self.dimension))
        
        # add v to the period set.
        P = np.concatenate((self.periodSet.T, v), axis=0).T
        b = self.offset
        print(NaturalNumbers.periodSet)
        resultC, resultP = intersect(NaturalNumbers, LinearSet(b, P))
        
        coefficientsC = [point[:(P[0].size)] for point in resultC]
        coefficientsP = [point[:(P[0].size)] for point in resultP]
        
        newPeriodSet = []
        newOffsetSet = []
        for coeff in coefficientsP:
            newPeriodSet.append(buildVector(coeff, P.transpose()))
        for coeff in coefficientsC:
            newOffsetSet.append(buildVector(coeff, P.transpose()) + b)
        
        return np.vstack(newOffsetSet), np.vstack(newPeriodSet)
    
    

  
# We represent Semi-Linear Sets as L = \bigcup_{i\in I} L(b_i,P_i), where P_i periodic set and b_i base vector.

def project(points, dimension):
    return [point[:dimension] for point in points]

def intersect(l1 : LinearSet, l2 : LinearSet):
        
    A = np.concatenate((l2.periodSet, -l1.periodSet), axis=1)
    b = l1.offset - l2.offset
    
    C = getMinimalPoints(A=A, b=b, noZero=False, greaterEqual=False)
    
    P = getMinimalPoints(A=A, b=b-b, noZero=True, greaterEqual=False)
    
    return C,P

class SemiLinearSet:
    def __init__(self, baseSets, periodSets):
        self.baseSets = baseSets
        self.periodSets = periodSets
        
    def __init__(self, linearSet : LinearSet):
        self.baseSets = [linearSet.offset]
        self.periodSets = [linearSet.periodSet]
        
    def add(self, C, P):
        self.baseSets += C
        self.periodSets += P
    
        
    def intersect(self, other):
        intersectionSet = SemiLinearSet([],[])
        for b, P in zip(self.baseSets, self.periodSets):
            for c, Q in zip(other.baseSets, other.periodSets):
                C_prime, P_prime = intersect(LinearSet(b, P), LinearSet(c, Q))
                print("------")
                print(C_prime)
                print(P_prime)
                print("------")

        return intersectionSet
    
    # adds IN_0 * x to the set. Non-trivial operation.
    def addVector(self, v):
        newSemiLinearSet = SemiLinearSet([],[])
        for b, P in zip(self.baseSets, self.periodSets):
            linearSet = LinearSet(offset=b, periodSet=P)
            C, Q = linearSet.addVector(v=v)
            newSemiLinearSet.add(C=C, P=Q)
        return newSemiLinearSet
        
    # trivially, we can just union the sets.
    # TODO: There could be some logic implemented to reduce redundancy i.e. remove linearly dependent columns etc.
    def union(self, other):
        semilinearUnion = SemiLinearSet()
        semilinearUnion.baseSets = self.baseSets + other.baseSets
        semilinearUnion.periodSets = self.periodSets + other.periodSets
        return semilinearUnion
    
    



if __name__ == "__main__":
    P = np.array([[0],
                  [1]])
    print((P[0].size))
    b = np.array([[0,3]])
    
    # lSet = LinearSet(offset=b, periodSet=P)
    # N = LinearSet(offset=np.zeros(b.shape, dtype=int), periodSet=np.array([[0,1],[1,0]]))
    # resultC, resultP = intersect(N,lSet)

    # # project (the first part of the tau projection):
    # coefficientsC = [point[:(P[0].size)] for point in resultC]
    # coefficientsP = [point[:(P[0].size)] for point in resultP]
    # print(resultC)
    # print(resultP)
    # print()
    # for coefficients in coefficientsP:
    #     print(buildVector(coefficients=coefficients, vectors=P.transpose()))
    # print()
    # for coefficients in coefficientsC:
    #     print(buildVector(coefficients=coefficients, vectors=P.transpose()) + b)
    # p1 = [0,1]
    # b = [0,3]
    # L = SemiLinearSet(LinearSet(b, p1))
    lSet = LinearSet(offset=b, periodSet=P)
    v = np.array([[3,-1]])
    print(lSet.periodSet)
    B, PeriodSet = lSet.addVector(v)
    print(B)
    print()
    print(PeriodSet)
    

    
    
