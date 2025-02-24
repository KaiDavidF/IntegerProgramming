import gurobipy as gp
from gurobipy import GRB
import numpy as np



def listLeq(a, b):
    assert len(a) == len(b)
    for i in range(len(a)):
        if a[i] > b[i]:
            return False
    return True

# checks if every point in the list is actually minimal.
def areMinPoints(points):
    for point in points:
        for otherPoint in points:
            if listLeq(point, otherPoint) and otherPoint != point:
                return False
    return True
    


def getMinimalPoint(model, x):
    model.setObjective(gp.quicksum(x[i] for i in range(len(x))), GRB.MINIMIZE)
    model.optimize()
    result = None
    if model.Status == GRB.OPTIMAL:
        result = [x[i].X for i in range(len(x))]
    model.setObjective(0)
    return result
        
# checks if there is a feasible point which is leq than point.
def isMinimalPoint(model, x, minPointCandidate):
    # We should only temporarily add the constraints to the model.
    constraints = []
    isMinimal = True
    
    for i in range(len(x)):
        constraints.append(model.addConstr(x[i] <= minPointCandidate[i]))
    
    # the minimal point should also be strictly less in some entry:
    constraints.append(model.addConstr(gp.quicksum(minPointCandidate[i] - x[i] for i in range(len(x))) >= 1))
    
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        # We found a new minimal point?
        isMinimal = False
        minPointCandidate = [x[i].X for i in range(len(x))]
    
    for constraint in constraints:
        model.remove(constraint)
        
    return isMinimal, minPointCandidate

def getMinimalPointsFullLinear(P, b, noZero=False):
    minimalPoints = []
    model = gp.Model()
    # model.setParam('OutputFlag', 0)
    
    # +1 because we add n.
    x = [model.addVar(lb=0, vtype=GRB.INTEGER) for _ in range(len(b)+1)]
    
    if noZero:
        model.addConstr(gp.quicksum(x[i] for i in range(len(x))) >= 1)
    
    latticeCoeff = [model.addVar(lb=0, vtype=GRB.INTEGER) for _ in range(len(P))]
    coneCoeff = [model.addVar(lb=0, vtype=GRB.CONTINUOUS) for _ in range(len(P))]
    
    for i in range(len(x)):
        model.addConstr(gp.quicksum(latticeCoeff[j] * P[j][i] for j in range(len(latticeCoeff))) == x[i])
        model.addConstr(gp.quicksum(coneCoeff[j] * P[j][i] for j in range(len(coneCoeff))) == x[i])
        
    while True:
        model.optimize()
        
        if model.Status != GRB.OPTIMAL:
            break
        
        newSolution = [sol.X for sol in x]
        
        newSolution = getMinimalPoint(model=model, x=newSolution)
        
        if newSolution is None:
            break
        
        minimalPoints.append(newSolution)
        
        M = 1000
        
        binary = [model.addVar(vtype=GRB.BINARY) for _ in range(len(x))]
        
        for i in range(len(x)):
            model.addConstr(x[i] <= (newSolution[i] - 1)  + M * binary[i])
            
        model.addConstr(gp.quicksum(binary[i] for i in range(len(binary))) <= len(binary)-1)
        
        model.optimize()
    
    return minimalPoints

def getMinimalPoints(A, b, noZero=False):
    minimalPoints = []
    
    assert len(A) == b.size
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    x = [model.addVar(lb=0, vtype=GRB.INTEGER) for _ in range(A[0].size)]
    
    if noZero:
        model.addConstr(gp.quicksum(x[i] for i in range(len(x))) >= 1)

    for row in A:
        assert row.size == len(x)
    
    for idr, row in enumerate(A):
        model.addConstr(gp.quicksum(row.flat[i] * x[i] for i in range(row.size)) == b.flat[idr])
        
    ctr = 0
    while True:
    # First, find some feasible solution:
        model.optimize()

        if model.Status != GRB.OPTIMAL:
            break # No more minimal points.
    
        x_prime = [x[i].X for i in range(len(x))]

        x_prime = getMinimalPoint(model=model, x=x)
        print(f"x_p = {x_prime}")
        print(f"n = {ctr}")
        print(len(x))
        ctr+=1
        
        if x_prime is None:
            break
        minimalPoints.append(x_prime)
        
        M = 1000 # Some big M, has to be larger than feasible points.

        binary = [model.addVar(vtype=GRB.BINARY) for _ in range(len(x))]
        
        for i in range(len(x)):
            model.addConstr(x[i] <= (x_prime[i] - 1)  + M * binary[i])
            
        model.addConstr(gp.quicksum(binary[i] for i in range(len(binary))) <= len(binary)-1)
        
        model.optimize()
        
        
    return minimalPoints

if __name__ == "__main__":
    P = np.array([[1,2,3],
                  [4,5,6]])
    c = np.array([[1,2]])
    
    A = np.concatenate((P, -P), axis=1)
    b = c-c
    
    print(f"A={A}")
    print(f"b={b}")
    
    minPoints = getMinimalPoints(A=A, b=b-b, noZero=True)
    
    print(minPoints)
    print(areMinPoints(minPoints))
