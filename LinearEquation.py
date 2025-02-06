import gurobipy as gp
from gurobipy import GRB

def solveIntegerEquation(A, rhs):
    model = gp.Model("IntegerEquationSolver")
    model.setParam('OutputFlag', 0)
    # Optionally suppress output:
    model.Params.OutputFlag = 0

    k = len(A)  # number of generators (unknowns y_i)
    # Create nonnegative integer variables y_0, y_1, ..., y_{k-1}.
    y = [model.addVar(lb=0, vtype=GRB.INTEGER, name=f"y_{i}") for i in range(k)]
    
    d = len(rhs)  # ambient dimension
    # For each coordinate j, add the constraint:
    #    sum_{i=0}^{k-1} A[i][j] * y[i] == rhs[j]
    for j in range(d):
        model.addConstr(gp.quicksum(A[i][j] * y[i] for i in range(k)) == rhs[j],
                        name=f"Constraint_{j}")
    
    model.optimize()
    
    if model.Status == GRB.OPTIMAL:
        # Retrieve the solution vector from the variables.
        solution = [var.X for var in y]
        return solution
    else:
        return None