import pulp

def is_natural_combination(target, others):
    """
    Given a target vector and a list of other vectors, decide if
    target can be written as a linear combination
       target = sum_j (x_j * others[j])
    with nonnegative integers x_j.
    """
    dim = len(target)
    # Create a (dummy) LP problem. We do not really care about the objective.
    prob = pulp.LpProblem("NaturalCombination", pulp.LpMinimize)
    
    # For each vector in 'others', create a variable x_j (integer, ≥ 0)
    xs = []
    for j, vec in enumerate(others):
        # It is often useful to restrict the variable's upper bound.
        # For any coordinate i where vec[i] > 0 we must have x_j * vec[i] ≤ target[i],
        # so a safe upper bound is floor(target[i]/vec[i]). We take the minimum over such i.
        ub_candidates = [target[i] // vec[i] for i in range(dim) if vec[i] > 0]
        if ub_candidates:
            ub = min(ub_candidates)
        else:
            # If vec is all zeros, then unless target is also zero,
            # its coefficient cannot help. We set ub=0.
            ub = 0
        x = pulp.LpVariable(f"x_{j}", lowBound=0, upBound=ub, cat="Integer")
        xs.append(x)
    
    # We do not really care about the value; we just want feasibility.
    prob += 0
    
    # Add one constraint for each coordinate so that:
    # sum_j (xs[j] * others[j][i]) == target[i]
    for i in range(dim):
        prob += pulp.lpSum(xs[j] * others[j][i] for j in range(len(others))) == target[i]
    
    # Solve the ILP (using CBC solver quietly)
    result = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[result] == "Optimal":
        coeffs = [int(x.varValue) for x in xs]
        return True, coeffs
    else:
        return False, None

def find_natural_linear_combination(vectors):
    """
    For each vector in 'vectors', check if it is a natural linear combination
    of the others. If one is found, print the result and stop.
    """
    for i, vec in enumerate(vectors):
        # Create the list of other vectors (all except the i-th one)
        others = vectors[:i] + vectors[i+1:]
        possible, coeffs = is_natural_combination(vec, others)
        if possible:
            print(f"Vector {vec} is a natural linear combination of the others.")
            print(f"One such representation is:")
            # To show the combination, print the coefficients and the corresponding vectors.
            for coeff, other in zip(coeffs, others):
                print(f"  {coeff} * {other}")
            return
    print("No vector is a natural linear combination of the others.")

if __name__ == "__main__":
    # Example: you can change this list to test with different vectors.
    import pulp

def is_natural_combination(target, others):
    """
    Given a target vector and a list of other vectors, decide if
    target can be written as a linear combination
       target = sum_j (x_j * others[j])
    with nonnegative integers x_j.
    """
    dim = len(target)
    # Create a (dummy) LP problem. We do not really care about the objective.
    prob = pulp.LpProblem("NaturalCombination", pulp.LpMinimize)
    
    # For each vector in 'others', create a variable x_j (integer, ≥ 0)
    xs = []
    for j, vec in enumerate(others):
        # It is often useful to restrict the variable's upper bound.
        # For any coordinate i where vec[i] > 0 we must have x_j * vec[i] ≤ target[i],
        # so a safe upper bound is floor(target[i]/vec[i]). We take the minimum over such i.
        ub_candidates = [target[i] // vec[i] for i in range(dim) if vec[i] > 0]
        if ub_candidates:
            ub = min(ub_candidates)
        else:
            # If vec is all zeros, then unless target is also zero,
            # its coefficient cannot help. We set ub=0.
            ub = 0
        x = pulp.LpVariable(f"x_{j}", lowBound=0, upBound=ub, cat="Integer")
        xs.append(x)
    
    # We do not really care about the value; we just want feasibility.
    prob += 0
    
    # Add one constraint for each coordinate so that:
    # sum_j (xs[j] * others[j][i]) == target[i]
    for i in range(dim):
        prob += pulp.lpSum(xs[j] * others[j][i] for j in range(len(others))) == target[i]
    
    # Solve the ILP (using CBC solver quietly)
    result = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[result] == "Optimal":
        coeffs = [int(x.varValue) for x in xs]
        return True, coeffs
    else:
        return False, None

def find_natural_linear_combination(vectors):
    """
    For each vector in 'vectors', check if it is a natural linear combination
    of the others. If one is found, print the result and stop.
    """
    for i, vec in enumerate(vectors):
        # Create the list of other vectors (all except the i-th one)
        others = vectors[:i] + vectors[i+1:]
        possible, coeffs = is_natural_combination(vec, others)
        if possible:
            print(f"Vector {vec} is a natural linear combination of the others.")
            print(f"One such representation is:")
            # To show the combination, print the coefficients and the corresponding vectors.
            for coeff, other in zip(coeffs, others):
                print(f"  {coeff} * {other}")
            return
    print("No vector is a natural linear combination of the others.")

if __name__ == "__main__":
    # Example: you can change this list to test with different vectors.
    vectors = [[37.0, 4.0, 6.0], [29.0, 8.0, 50.0], [39.0, 3.0, 37.0], [43.0, 1.0, 21.0], [31.0, 7.0, 6.0], [13.0, 16.0, -0.0], [33.0, 6.0, 31.0], [-0.0, 45.0, -0.0], [25.0, 10.0, -0.0], [37.0, 4.0, 15.0], [13.0, 16.0, 9.0], [33.0, 6.0, 40.0], [-0.0, 45.0, 9.0], [37.0, 4.0, 24.0], [39.0, 3.0, 55.0], [13.0, 16.0, 18.0], [33.0, 6.0, 49.0], [37.0, 4.0, 33.0], [35.0, 5.0, 5.0], [39.0, 3.0, 64.0], [39.0, 3.0, 49.0], [45.0, 0.0, 45.0], [45.0, 0.0, 54.0], [43.0, 1.0, 16.0], [45.0, 0.0, 63.0], [45.0, 0.0, 72.0], [1.0, 22.0, 7.0], [17.0, 14.0, 7.0], [27.0, 9.0, 3.0], [27.0, 9.0, 12.0], [41.0, 2.0, 55.0], [21.0, 12.0, 6.0], [7.0, 19.0, 6.0], [7.0, 19.0, 15.0], [45.0, 0.0, 6.0], [29.0, 8.0, 18.0], [29.0, 8.0, 27.0], [11.0, 17.0, 5.0], [41.0, 2.0, 51.0], [23.0, 11.0, 29.0], [29.0, 8.0, 36.0], [5.0, 20.0, 5.0], [11.0, 17.0, 14.0], [39.0, 3.0, 23.0], [23.0, 11.0, 38.0], [29.0, 8.0, 45.0], [5.0, 20.0, 14.0], [11.0, 17.0, 23.0], [31.0, 7.0, 1.0], [33.0, 6.0, 17.0], [33.0, 6.0, 26.0], [37.0, 4.0, 1.0], [37.0, 4.0, 10.0], [39.0, 3.0, 32.0], [13.0, 16.0, 4.0], [33.0, 6.0, 35.0], [37.0, 4.0, 19.0], [39.0, 3.0, 50.0], [39.0, 3.0, 41.0], [13.0, 16.0, 13.0], [33.0, 6.0, 44.0], [39.0, 3.0, 59.0], [33.0, 6.0, 53.0], [45.0, 0.0, 40.0], [45.0, 0.0, 49.0], [45.0, 0.0, 58.0], [45.0, 0.0, 67.0], [1.0, 22.0, 2.0], [17.0, 14.0, 2.0], [7.0, 19.0, 1.0], [43.0, 1.0, 47.0], [7.0, 19.0, 10.0], [29.0, 8.0, 13.0], [41.0, 2.0, 37.0], [29.0, 8.0, 22.0], [39.0, 3.0, 9.0], [11.0, 17.0, -0.0], [41.0, 2.0, 46.0], [23.0, 11.0, 24.0], [29.0, 8.0, 31.0], [5.0, 20.0, -0.0], [11.0, 17.0, 9.0], [33.0, 6.0, 3.0], [23.0, 11.0, 33.0], [29.0, 8.0, 40.0], [5.0, 20.0, 9.0], [11.0, 17.0, 18.0], [33.0, 6.0, 12.0], [33.0, 6.0, 21.0], [29.0, 8.0, 49.0], [37.0, 4.0, 5.0], [39.0, 3.0, 27.0], [39.0, 3.0, 36.0], [33.0, 6.0, 30.0], [39.0, 3.0, 45.0], [41.0, 2.0, 64.0], [33.0, 6.0, 39.0], [45.0, 0.0, 26.0], [45.0, 0.0, 35.0], [45.0, 0.0, 44.0], [3.0, 21.0, 7.0], [45.0, 0.0, 53.0], [45.0, 0.0, 62.0], [1.0, 22.0, 6.0], [39.0, 3.0, 61.0], [7.0, 19.0, 5.0], [29.0, 8.0, 8.0], [41.0, 2.0, 32.0], [7.0, 19.0, 14.0], [23.0, 11.0, 10.0], [29.0, 8.0, 17.0], [39.0, 3.0, 4.0], [41.0, 2.0, 41.0], [23.0, 11.0, 19.0], [29.0, 8.0, 26.0], [39.0, 3.0, 13.0], [11.0, 17.0, 4.0], [41.0, 2.0, 50.0], [23.0, 11.0, 28.0], [29.0, 8.0, 35.0], [33.0, 6.0, 7.0], [11.0, 17.0, 13.0], [39.0, 3.0, 22.0], [23.0, 11.0, 37.0], [33.0, 6.0, 16.0], [37.0, 4.0, -0.0], [41.0, 2.0, 68.0], [11.0, 17.0, 22.0], [33.0, 6.0, 25.0], [39.0, 3.0, 31.0], [39.0, 3.0, 40.0], [45.0, 0.0, 12.0], [33.0, 6.0, 34.0], [19.0, 13.0, 34.0], [45.0, 0.0, 21.0], [45.0, 0.0, 30.0], [45.0, 0.0, 39.0], [3.0, 21.0, 2.0], [45.0, 0.0, 48.0], [3.0, 21.0, 11.0], [1.0, 22.0, 1.0], [41.0, 2.0, 18.0], [7.0, 19.0, -0.0], [43.0, 1.0, 10.0], [29.0, 8.0, 3.0], [41.0, 2.0, 27.0], [7.0, 19.0, 9.0], [23.0, 11.0, 5.0], [29.0, 8.0, 12.0], [43.0, 1.0, 9.0], [41.0, 2.0, 36.0], [23.0, 11.0, 14.0], [29.0, 8.0, 21.0], [39.0, 3.0, 8.0], [41.0, 2.0, 45.0], [23.0, 11.0, 23.0], [29.0, 8.0, 30.0], [33.0, 6.0, 2.0], [11.0, 17.0, 8.0], [39.0, 3.0, 17.0], [23.0, 11.0, 32.0], [33.0, 6.0, 11.0], [39.0, 3.0, 26.0], [41.0, 2.0, 54.0], [23.0, 11.0, 41.0], [33.0, 6.0, 20.0], [41.0, 2.0, 60.0], [45.0, 0.0, 7.0], [19.0, 13.0, 29.0], [31.0, 7.0, 53.0], [41.0, 2.0, 59.0], [45.0, 0.0, 16.0], [43.0, 1.0, 59.0], [45.0, 0.0, 25.0], [45.0, 0.0, 34.0], [35.0, 5.0, 52.0], [45.0, 0.0, 43.0], [3.0, 21.0, 6.0], [37.0, 4.0, 37.0], [41.0, 2.0, 13.0], [41.0, 2.0, 22.0], [7.0, 19.0, 4.0], [23.0, 11.0, -0.0], [29.0, 8.0, 7.0], [41.0, 2.0, 31.0], [23.0, 11.0, 9.0], [29.0, 8.0, 16.0], [39.0, 3.0, 3.0], [41.0, 2.0, 40.0], [23.0, 11.0, 18.0], [39.0, 3.0, 12.0], [11.0, 17.0, 3.0], [41.0, 2.0, 49.0], [23.0, 11.0, 27.0], [33.0, 6.0, 6.0], [39.0, 3.0, 21.0], [33.0, 6.0, 15.0], [19.0, 13.0, 15.0], [45.0, 0.0, 2.0], [19.0, 13.0, 24.0], [31.0, 7.0, 48.0], [45.0, 0.0, 11.0], [19.0, 13.0, 33.0], [25.0, 10.0, 42.0], [45.0, 0.0, 20.0], [35.0, 5.0, 38.0], [43.0, 1.0, 40.0], [45.0, 0.0, 29.0], [35.0, 5.0, 47.0], [3.0, 21.0, 1.0], [43.0, 1.0, 39.0], [35.0, 5.0, 56.0], [3.0, 21.0, 10.0], [15.0, 15.0, 21.0], [27.0, 9.0, 45.0], [41.0, 2.0, 8.0], [9.0, 18.0, 15.0], [41.0, 2.0, 17.0], [29.0, 8.0, 2.0], [41.0, 2.0, 26.0], [23.0, 11.0, 4.0], [29.0, 8.0, 11.0], [-0.0, 23.0, 4.0], [41.0, 2.0, 35.0], [23.0, 11.0, 13.0], [39.0, 3.0, 7.0], [23.0, 11.0, 22.0], [33.0, 6.0, 1.0], [19.0, 13.0, 10.0], [31.0, 7.0, 34.0], [19.0, 13.0, 19.0], [25.0, 10.0, 28.0], [31.0, 7.0, 43.0], [39.0, 3.0, 54.0], [43.0, 1.0, 67.0], [19.0, 13.0, 28.0], [25.0, 10.0, 37.0], [31.0, 7.0, 52.0], [37.0, 4.0, 61.0], [45.0, 0.0, 15.0], [39.0, 3.0, 53.0], [35.0, 5.0, 33.0], [45.0, 0.0, 24.0], [35.0, 5.0, 42.0], [35.0, 5.0, 51.0], [15.0, 15.0, 7.0], [9.0, 18.0, 1.0], [15.0, 15.0, 16.0], [27.0, 9.0, 40.0], [41.0, 2.0, 3.0], [9.0, 18.0, 10.0], [15.0, 15.0, 25.0], [21.0, 12.0, 34.0], [41.0, 2.0, 12.0], [9.0, 18.0, 19.0], [41.0, 2.0, 21.0], [23.0, 11.0, 8.0], [39.0, 3.0, 2.0], [19.0, 13.0, 5.0], [31.0, 7.0, 29.0], [43.0, 1.0, 53.0], [19.0, 13.0, 14.0], [25.0, 10.0, 23.0], [31.0, 7.0, 38.0], [37.0, 4.0, 47.0], [43.0, 1.0, 62.0], [19.0, 13.0, 23.0], [25.0, 10.0, 32.0], [31.0, 7.0, 47.0], [35.0, 5.0, 19.0], [43.0, 1.0, 71.0], [19.0, 13.0, 32.0], [25.0, 10.0, 41.0], [35.0, 5.0, 28.0], [37.0, 4.0, 56.0], [43.0, 1.0, 3.0], [45.0, 0.0, 10.0], [35.0, 5.0, 37.0], [43.0, 1.0, 2.0], [35.0, 5.0, 46.0], [35.0, 5.0, 55.0], [17.0, 14.0, 30.0], [15.0, 15.0, 2.0], [27.0, 9.0, 26.0], [15.0, 15.0, 11.0], [21.0, 12.0, 20.0], [27.0, 9.0, 35.0], [41.0, 2.0, 52.0], [9.0, 18.0, 5.0], [15.0, 15.0, 20.0], [21.0, 12.0, 29.0], [27.0, 9.0, 44.0], [41.0, 2.0, 7.0], [9.0, 18.0, 14.0], [15.0, 15.0, 29.0], [21.0, 12.0, 38.0], [41.0, 2.0, 16.0], [43.0, 1.0, 52.0], [23.0, 11.0, 3.0], [-0.0, 23.0, 3.0], [31.0, 7.0, 15.0], [19.0, 13.0, -0.0], [25.0, 10.0, 9.0], [31.0, 7.0, 24.0], [43.0, 1.0, 48.0], [19.0, 13.0, 9.0], [25.0, 10.0, 18.0], [31.0, 7.0, 33.0], [37.0, 4.0, 42.0], [43.0, 1.0, 57.0], [19.0, 13.0, 18.0], [25.0, 10.0, 27.0], [31.0, 7.0, 42.0], [35.0, 5.0, 14.0], [37.0, 4.0, 51.0], [19.0, 13.0, 27.0], [25.0, 10.0, 36.0], [31.0, 7.0, 51.0], [35.0, 5.0, 23.0], [37.0, 4.0, 60.0], [43.0, 1.0, 66.0], [35.0, 5.0, 32.0], [35.0, 5.0, 41.0], [35.0, 5.0, 50.0], [17.0, 14.0, 16.0], [17.0, 14.0, 25.0], [27.0, 9.0, 21.0], [15.0, 15.0, 6.0], [21.0, 12.0, 15.0], [27.0, 9.0, 30.0], [9.0, 18.0, -0.0], [15.0, 15.0, 15.0], [21.0, 12.0, 24.0], [27.0, 9.0, 39.0], [41.0, 2.0, 2.0], [9.0, 18.0, 9.0], [15.0, 15.0, 24.0], [21.0, 12.0, 33.0], [43.0, 1.0, 33.0], [9.0, 18.0, 18.0], [31.0, 7.0, 10.0], [43.0, 1.0, 34.0], [-0.0, 45.0, 4.0], [25.0, 10.0, 4.0], [31.0, 7.0, 19.0], [37.0, 4.0, 28.0], [43.0, 1.0, 43.0], [0.0, 45.0, 13.0], [19.0, 13.0, 4.0], [25.0, 10.0, 13.0], [13.0, 16.0, 22.0], [31.0, 7.0, 28.0], [19.0, 13.0, 13.0], [25.0, 10.0, 22.0], [31.0, 7.0, 37.0], [35.0, 5.0, -0.0], [35.0, 5.0, 9.0], [37.0, 4.0, 46.0], [25.0, 10.0, 31.0], [31.0, 7.0, 46.0], [35.0, 5.0, 18.0], [37.0, 4.0, 55.0], [43.0, 1.0, 70.0], [25.0, 10.0, 40.0], [35.0, 5.0, 27.0], [43.0, 1.0, 61.0], [35.0, 5.0, 36.0], [17.0, 14.0, 11.0], [17.0, 14.0, 20.0], [27.0, 9.0, 7.0], [21.0, 12.0, 1.0], [17.0, 14.0, 29.0], [27.0, 9.0, 16.0], [15.0, 15.0, 1.0], [21.0, 12.0, 10.0], [27.0, 9.0, 25.0], [15.0, 15.0, 10.0], [21.0, 12.0, 19.0], [27.0, 9.0, 34.0], [39.0, 3.0, 46.0], [9.0, 18.0, 4.0], [15.0, 15.0, 19.0], [21.0, 12.0, 28.0], [27.0, 9.0, 43.0], [9.0, 18.0, 13.0], [43.0, 1.0, 14.0], [21.0, 12.0, 37.0], [43.0, 1.0, 13.0], [43.0, 1.0, 20.0], [31.0, 7.0, 5.0], [37.0, 4.0, 14.0], [43.0, 1.0, 29.0], [41.0, 2.0, 65.0], [31.0, 7.0, 14.0], [13.0, 16.0, 8.0], [37.0, 4.0, 23.0], [-0.0, 45.0, 8.0], [25.0, 10.0, 8.0], [31.0, 7.0, 23.0], [13.0, 16.0, 17.0], [33.0, 6.0, 48.0], [19.0, 13.0, 8.0], [25.0, 10.0, 17.0], [31.0, 7.0, 32.0], [13.0, 16.0, 26.0], [35.0, 5.0, 4.0], [37.0, 4.0, 41.0], [25.0, 10.0, 26.0], [35.0, 5.0, 13.0], [37.0, 4.0, 50.0], [43.0, 1.0, 56.0], [35.0, 5.0, 22.0], [35.0, 5.0, 31.0], [1.0, 0.0, 0.0], [45.0, 0.0, 71.0], [17.0, 14.0, 6.0], [17.0, 14.0, 15.0], [27.0, 9.0, 2.0], [17.0, 14.0, 24.0], [27.0, 9.0, 11.0], [21.0, 12.0, 5.0], [27.0, 9.0, 20.0], [15.0, 15.0, 5.0], [21.0, 12.0, 14.0], [27.0, 9.0, 29.0], [21.0, 12.0, 23.0], [27.0, 9.0, 38.0], [21.0, 12.0, 32.0], [5.0, 20.0, 4.0], [29.0, 8.0, 44.0], [5.0, 20.0, 13.0], [43.0, 1.0, 15.0], [31.0, 7.0, -0.0], [37.0, 4.0, 9.0], [43.0, 1.0, 24.0], [45.0, 0.0, 1.0], [31.0, 7.0, 9.0], [13.0, 16.0, 3.0], [37.0, 4.0, 18.0], [-0.0, 45.0, 3.0], [25.0, 10.0, 3.0], [31.0, 7.0, 18.0], [13.0, 16.0, 12.0], [33.0, 6.0, 43.0], [-0.0, 45.0, 12.0], [25.0, 10.0, 12.0], [37.0, 4.0, 27.0], [13.0, 16.0, 21.0], [33.0, 6.0, 52.0], [37.0, 4.0, 36.0], [35.0, 5.0, 8.0], [39.0, 3.0, 58.0], [37.0, 4.0, 45.0], [43.0, 1.0, 51.0], [43.0, 1.0, 45.0], [35.0, 5.0, 17.0], [43.0, 1.0, 44.0], [45.0, 0.0, 57.0], [45.0, 0.0, 66.0], [17.0, 14.0, 1.0], [17.0, 14.0, 10.0], [17.0, 14.0, 19.0], [27.0, 9.0, 6.0], [21.0, 12.0, 0.0], [27.0, 9.0, 15.0], [15.0, 15.0, -0.0], [21.0, 12.0, 9.0], [27.0, 9.0, 24.0], [21.0, 12.0, 18.0], [43.0, 1.0, 1.0], [29.0, 8.0, 39.0], [5.0, 20.0, 8.0], [11.0, 17.0, 17.0], [41.0, 2.0, 63.0], [37.0, 4.0, 4.0], [29.0, 8.0, 48.0], [39.0, 3.0, 35.0], [43.0, 1.0, 19.0], [31.0, 7.0, 4.0], [33.0, 6.0, 29.0], [37.0, 4.0, 13.0], [39.0, 3.0, 44.0], [43.0, 1.0, 28.0], [31.0, 7.0, 13.0], [13.0, 16.0, 7.0], [33.0, 6.0, 38.0], [-0.0, 45.0, 7.0], [25.0, 10.0, 7.0], [37.0, 4.0, 22.0], [13.0, 16.0, 16.0], [33.0, 6.0, 47.0], [37.0, 4.0, 31.0], [35.0, 5.0, 3.0], [39.0, 3.0, 62.0], [33.0, 6.0, 56.0], [13.0, 16.0, 25.0], [35.0, 5.0, 12.0], [45.0, 0.0, 52.0], [43.0, 1.0, 25.0], [45.0, 0.0, 61.0], [45.0, 0.0, 70.0], [1.0, 22.0, 5.0], [17.0, 14.0, 5.0], [17.0, 14.0, 14.0], [27.0, 9.0, 1.0], [27.0, 9.0, 10.0], [21.0, 12.0, 4.0], [27.0, 9.0, 19.0], [21.0, 12.0, 13.0], [7.0, 19.0, 13.0], [29.0, 8.0, 25.0], [29.0, 8.0, 34.0], [5.0, 20.0, 3.0], [11.0, 17.0, 12.0], [41.0, 2.0, 58.0], [23.0, 11.0, 36.0], [29.0, 8.0, 43.0], [5.0, 20.0, 12.0], [11.0, 17.0, 21.0], [39.0, 3.0, 30.0], [33.0, 6.0, 24.0], [37.0, 4.0, 8.0], [39.0, 3.0, 39.0], [41.0, 2.0, 67.0], [43.0, 1.0, 23.0], [13.0, 16.0, 2.0], [33.0, 6.0, 33.0], [-0.0, 45.0, 2.0], [37.0, 4.0, 17.0], [39.0, 3.0, 48.0], [13.0, 16.0, 11.0], [33.0, 6.0, 42.0], [37.0, 4.0, 26.0], [39.0, 3.0, 57.0], [43.0, 1.0, 32.0], [13.0, 16.0, 20.0], [33.0, 6.0, 51.0], [45.0, 0.0, 38.0], [45.0, 0.0, 47.0], [43.0, 1.0, 7.0], [45.0, 0.0, 56.0], [43.0, 1.0, 6.0], [45.0, 0.0, 65.0], [1.0, 22.0, -0.0], [45.0, -0.0, 74.0], [17.0, 14.0, -0.0], [27.0, 9.0, 5.0], [41.0, 2.0, 57.0], [7.0, 19.0, 8.0], [7.0, 19.0, 17.0], [29.0, 8.0, 20.0], [41.0, 2.0, 44.0], [29.0, 8.0, 29.0], [39.0, 3.0, 16.0], [11.0, 17.0, 7.0], [41.0, 2.0, 53.0], [23.0, 11.0, 31.0], [29.0, 8.0, 38.0], [5.0, 20.0, 7.0], [11.0, 17.0, 16.0], [33.0, 6.0, 10.0], [23.0, 11.0, 40.0], [29.0, 8.0, 47.0], [33.0, 6.0, 19.0], [37.0, 4.0, 3.0], [39.0, 3.0, 25.0], [33.0, 6.0, 28.0], [37.0, 4.0, 12.0], [39.0, 3.0, 34.0], [39.0, 3.0, 43.0], [43.0, 1.0, 18.0], [13.0, 16.0, 6.0], [33.0, 6.0, 37.0], [39.0, 3.0, 52.0], [33.0, 6.0, 46.0], [45.0, 0.0, 33.0], [45.0, 0.0, 42.0], [3.0, 21.0, 5.0], [45.0, 0.0, 51.0], [45.0, 0.0, 60.0], [45.0, 0.0, 69.0], [1.0, 22.0, 4.0], [43.0, 1.0, 38.0], [7.0, 19.0, 3.0], [29.0, 8.0, 6.0], [41.0, 2.0, 30.0], [7.0, 19.0, 12.0], [29.0, 8.0, 15.0], [43.0, 1.0, 37.0], [41.0, 2.0, 39.0], [23.0, 11.0, 17.0], [29.0, 8.0, 24.0], [39.0, 3.0, 11.0], [11.0, 17.0, 2.0], [41.0, 2.0, 48.0], [23.0, 11.0, 26.0], [29.0, 8.0, 33.0], [5.0, 20.0, 2.0], [11.0, 17.0, 11.0], [33.0, 6.0, 5.0], [23.0, 11.0, 35.0], [29.0, 8.0, 42.0], [33.0, 6.0, 14.0], [11.0, 17.0, 20.0], [39.0, 3.0, 20.0], [33.0, 6.0, 23.0], [37.0, 4.0, 7.0], [39.0, 3.0, 29.0], [39.0, 3.0, 38.0], [41.0, 2.0, 66.0], [33.0, 6.0, 32.0], [39.0, 3.0, 47.0], [45.0, 0.0, 19.0], [33.0, 6.0, 41.0], [45.0, 0.0, 28.0], [45.0, 0.0, 37.0], [3.0, 21.0, -0.0], [45.0, 0.0, 46.0], [3.0, 21.0, 9.0], [45.0, 0.0, 55.0], [1.0, 22.0, 8.0], [39.0, 3.0, 51.0], [29.0, 8.0, 1.0], [41.0, 2.0, 25.0], [7.0, 19.0, 7.0], [29.0, 8.0, 10.0], [41.0, 2.0, 34.0], [7.0, 19.0, 16.0], [23.0, 11.0, 12.0], [29.0, 8.0, 19.0], [39.0, 3.0, 6.0], [41.0, 2.0, 43.0], [23.0, 11.0, 21.0], [29.0, 8.0, 28.0], [33.0, 6.0, -0.0], [11.0, 17.0, 6.0], [39.0, 3.0, 15.0], [23.0, 11.0, 30.0], [33.0, 6.0, 9.0], [39.0, 3.0, 24.0], [11.0, 17.0, 15.0], [41.0, 2.0, 61.0], [23.0, 11.0, 39.0], [33.0, 6.0, 18.0], [39.0, 3.0, 33.0], [33.0, 6.0, 27.0], [45.0, 0.0, 14.0], [45.0, 0.0, 23.0], [45.0, 0.0, 32.0], [45.0, 0.0, 41.0], [3.0, 21.0, 4.0], [35.0, 5.0, 59.0], [45.0, 0.0, 50.0], [41.0, 2.0, 11.0], [41.0, 2.0, 20.0], [7.0, 19.0, 2.0], [29.0, 8.0, 5.0], [41.0, 2.0, 29.0], [7.0, 19.0, 11.0], [23.0, 11.0, 7.0], [29.0, 8.0, 14.0], [39.0, 3.0, 1.0], [41.0, 2.0, 38.0], [43.0, 1.0, -0.0], [23.0, 11.0, 16.0], [29.0, 8.0, 23.0], [39.0, 3.0, 10.0], [11.0, 17.0, 1.0], [41.0, 2.0, 47.0], [23.0, 11.0, 25.0], [33.0, 6.0, 4.0], [39.0, 3.0, 19.0], [11.0, 17.0, 10.0], [45.0, 0.0, 5.0], [23.0, 11.0, 34.0], [33.0, 6.0, 13.0], [39.0, 3.0, 28.0], [33.0, 6.0, 22.0], [19.0, 13.0, 22.0], [45.0, 0.0, 9.0], [19.0, 13.0, 31.0], [43.0, 1.0, 50.0], [45.0, 0.0, 18.0], [45.0, 0.0, 27.0], [43.0, 1.0, 49.0], [35.0, 5.0, 45.0], [45.0, 0.0, 36.0], [35.0, 5.0, 54.0], [3.0, 21.0, 8.0], [41.0, 2.0, 6.0], [15.0, 15.0, 28.0], [41.0, 2.0, 15.0], [29.0, 8.0, -0.0], [41.0, 2.0, 24.0], [23.0, 11.0, 2.0], [29.0, 8.0, 9.0], [-0.0, 23.0, 2.0], [41.0, 2.0, 33.0], [23.0, 11.0, 11.0], [39.0, 3.0, 5.0], [41.0, 2.0, 42.0], [23.0, 11.0, 20.0], [39.0, 3.0, 14.0], [33.0, 6.0, 8.0], [19.0, 13.0, 17.0], [31.0, 7.0, 41.0], [45.0, 0.0, 4.0], [43.0, 1.0, 65.0], [19.0, 13.0, 26.0], [25.0, 10.0, 35.0], [31.0, 7.0, 50.0], [37.0, 4.0, 59.0], [39.0, 3.0, 63.0], [19.0, 13.0, 35.0], [25.0, 10.0, 44.0], [43.0, 1.0, 31.0], [45.0, 0.0, 22.0], [45.0, 0.0, 13.0], [35.0, 5.0, 40.0], [45.0, 0.0, 31.0], [43.0, 1.0, 30.0], [35.0, 5.0, 49.0], [3.0, 21.0, 3.0], [35.0, 5.0, 58.0], [15.0, 15.0, 14.0], [41.0, 2.0, 1.0], [9.0, 18.0, 8.0], [15.0, 15.0, 23.0], [27.0, 9.0, 47.0], [41.0, 2.0, 10.0], [9.0, 18.0, 17.0], [41.0, 2.0, 19.0], [29.0, 8.0, 4.0], [41.0, 2.0, 28.0], [23.0, 11.0, 6.0], [39.0, 3.0, -0.0], [23.0, 11.0, 15.0], [19.0, 13.0, 3.0], [31.0, 7.0, 27.0], [19.0, 13.0, 12.0], [25.0, 10.0, 21.0], [31.0, 7.0, 36.0], [43.0, 1.0, 60.0], [19.0, 13.0, 21.0], [25.0, 10.0, 30.0], [31.0, 7.0, 45.0], [37.0, 4.0, 54.0], [43.0, 1.0, 69.0], [19.0, 13.0, 30.0], [25.0, 10.0, 39.0], [35.0, 5.0, 26.0], [45.0, 0.0, 17.0], [45.0, 0.0, 8.0], [35.0, 5.0, 35.0], [43.0, 1.0, 12.0], [35.0, 5.0, 44.0], [43.0, 1.0, 11.0], [35.0, 5.0, 53.0], [17.0, 14.0, 28.0], [41.0, 2.0, 62.0], [15.0, 15.0, 9.0], [27.0, 9.0, 33.0], [9.0, 18.0, 3.0], [15.0, 15.0, 18.0], [21.0, 12.0, 27.0], [27.0, 9.0, 42.0], [41.0, 2.0, 5.0], [9.0, 18.0, 12.0], [15.0, 15.0, 27.0], [21.0, 12.0, 36.0], [41.0, 2.0, 14.0], [41.0, 2.0, 23.0], [23.0, 11.0, 1.0], [-0.0, 23.0, 1.0], [31.0, 7.0, 22.0], [43.0, 1.0, 46.0], [19.0, 13.0, 7.0], [25.0, 10.0, 16.0], [31.0, 7.0, 31.0], [37.0, 4.0, 40.0], [43.0, 1.0, 55.0], [19.0, 13.0, 16.0], [25.0, 10.0, 25.0], [31.0, 7.0, 40.0], [37.0, 4.0, 49.0], [43.0, 1.0, 64.0], [19.0, 13.0, 25.0], [25.0, 10.0, 34.0], [31.0, 7.0, 49.0], [35.0, 5.0, 21.0], [37.0, 4.0, 58.0], [45.0, 0.0, 3.0], [25.0, 10.0, 43.0], [35.0, 5.0, 30.0], [35.0, 5.0, 39.0], [35.0, 5.0, 48.0], [35.0, 5.0, 57.0], [17.0, 14.0, 23.0], [17.0, 14.0, 32.0], [15.0, 15.0, 4.0], [27.0, 9.0, 28.0], [15.0, 15.0, 13.0], [21.0, 12.0, 22.0], [27.0, 9.0, 37.0], [41.0, 2.0, -0.0], [9.0, 18.0, 7.0], [15.0, 15.0, 22.0], [21.0, 12.0, 31.0], [27.0, 9.0, 46.0], [41.0, 2.0, 9.0], [9.0, 18.0, 16.0], [43.0, 1.0, 42.0], [0.0, 23.0, 5.0], [31.0, 7.0, 8.0], [25.0, 10.0, 2.0], [31.0, 7.0, 17.0], [43.0, 1.0, 41.0], [-0.0, 45.0, 11.0], [19.0, 13.0, 2.0], [25.0, 10.0, 11.0], [31.0, 7.0, 26.0], [37.0, 4.0, 35.0], [19.0, 13.0, 11.0], [25.0, 10.0, 20.0], [31.0, 7.0, 35.0], [35.0, 5.0, 7.0], [37.0, 4.0, 44.0], [19.0, 13.0, 20.0], [25.0, 10.0, 29.0], [31.0, 7.0, 44.0], [35.0, 5.0, 16.0], [37.0, 4.0, 53.0], [43.0, 1.0, 68.0], [25.0, 10.0, 38.0], [35.0, 5.0, 25.0], [37.0, 4.0, 62.0], [35.0, 5.0, 34.0], [35.0, 5.0, 43.0], [17.0, 14.0, 9.0], [17.0, 14.0, 18.0], [17.0, 14.0, 27.0], [27.0, 9.0, 14.0], [21.0, 12.0, 8.0], [27.0, 9.0, 23.0], [39.0, 3.0, 56.0], [15.0, 15.0, 8.0], [21.0, 12.0, 17.0], [27.0, 9.0, 32.0], [9.0, 18.0, 2.0], [15.0, 15.0, 17.0], [21.0, 12.0, 26.0], [27.0, 9.0, 41.0], [41.0, 2.0, 4.0], [9.0, 18.0, 11.0], [15.0, 15.0, 26.0], [21.0, 12.0, 35.0], [9.0, 18.0, 20.0], [-0.0, 23.0, -0.0], [31.0, 7.0, 3.0], [43.0, 1.0, 27.0], [31.0, 7.0, 12.0], [37.0, 4.0, 21.0], [43.0, 1.0, 36.0], [-0.0, 45.0, 6.0], [25.0, 10.0, 6.0], [31.0, 7.0, 21.0], [13.0, 16.0, 15.0], [37.0, 4.0, 30.0], [19.0, 13.0, 6.0], [25.0, 10.0, 15.0], [31.0, 7.0, 30.0], [13.0, 16.0, 24.0], [33.0, 6.0, 55.0], [35.0, 5.0, 2.0], [25.0, 10.0, 24.0], [31.0, 7.0, 39.0], [35.0, 5.0, 11.0], [37.0, 4.0, 48.0], [37.0, 4.0, 39.0], [25.0, 10.0, 33.0], [35.0, 5.0, 20.0], [37.0, 4.0, 57.0], [43.0, 1.0, 63.0], [35.0, 5.0, 29.0], [17.0, 14.0, 4.0], [17.0, 14.0, 13.0], [27.0, 9.0, -0.0], [17.0, 14.0, 22.0], [27.0, 9.0, 9.0], [21.0, 12.0, 3.0], [17.0, 14.0, 31.0], [27.0, 9.0, 18.0], [15.0, 15.0, 3.0], [21.0, 12.0, 12.0], [27.0, 9.0, 27.0], [15.0, 15.0, 12.0], [21.0, 12.0, 21.0], [27.0, 9.0, 36.0], [9.0, 18.0, 6.0], [43.0, 1.0, 5.0], [21.0, 12.0, 30.0], [43.0, 1.0, 4.0], [5.0, 20.0, 11.0], [43.0, 1.0, 22.0], [31.0, 7.0, 7.0], [13.0, 16.0, 1.0], [37.0, 4.0, 16.0], [-0.0, 45.0, 1.0], [25.0, 10.0, 1.0], [31.0, 7.0, 16.0], [13.0, 16.0, 10.0], [37.0, 4.0, 25.0], [-0.0, 45.0, 10.0], [19.0, 13.0, 1.0], [25.0, 10.0, 10.0], [13.0, 16.0, 19.0], [39.0, 3.0, 65.0], [31.0, 7.0, 25.0], [25.0, 10.0, 19.0], [33.0, 6.0, 50.0], [35.0, 5.0, 6.0], [37.0, 4.0, 43.0], [37.0, 4.0, 34.0], [35.0, 5.0, 15.0], [43.0, 1.0, 58.0], [37.0, 4.0, 52.0], [43.0, 1.0, 54.0], [35.0, 5.0, 24.0], [45.0, 0.0, 64.0], [45.0, 0.0, 73.0], [17.0, 14.0, 8.0], [17.0, 14.0, 17.0], [27.0, 9.0, 4.0], [37.0, 4.0, 32.0], [17.0, 14.0, 26.0], [27.0, 9.0, 13.0], [21.0, 12.0, 7.0], [27.0, 9.0, 22.0], [21.0, 12.0, 16.0], [27.0, 9.0, 31.0], [39.0, 3.0, 18.0], [21.0, 12.0, 25.0], [29.0, 8.0, 37.0], [5.0, 20.0, 6.0], [43.0, 1.0, 8.0], [37.0, 4.0, 2.0], [29.0, 8.0, 46.0], [43.0, 1.0, 17.0], [31.0, 7.0, 2.0], [37.0, 4.0, 11.0], [39.0, 3.0, 42.0], [43.0, 1.0, 26.0], [31.0, 7.0, 11.0], [13.0, 16.0, 5.0], [33.0, 6.0, 36.0], [-0.0, 45.0, 5.0], [25.0, 10.0, 5.0], [31.0, 7.0, 20.0], [13.0, 16.0, 14.0], [33.0, 6.0, 45.0], [37.0, 4.0, 20.0], [25.0, 10.0, 14.0], [35.0, 5.0, 1.0], [13.0, 16.0, 23.0], [33.0, 6.0, 54.0], [37.0, 4.0, 29.0], [35.0, 5.0, 10.0], [37.0, 4.0, 38.0], [39.0, 3.0, 60.0], [43.0, 1.0, 35.0], [45.0, 0.0, 59.0], [45.0, 0.0, 68.0], [1.0, 22.0, 3.0], [17.0, 14.0, 3.0], [17.0, 14.0, 12.0], [17.0, 14.0, 21.0], [27.0, 9.0, 8.0], [21.0, 12.0, 2.0], [27.0, 9.0, 17.0], [21.0, 12.0, 11.0], [29.0, 8.0, 32.0], [5.0, 20.0, 1.0], [41.0, 2.0, 56.0], [29.0, 8.0, 41.0], [5.0, 20.0, 10.0], [11.0, 17.0, 19.0]]
    
    vectors2 = [[21.0, 12.0, 38.0], [27.0, 9.0, 47.0], [29.0, 8.0, 50.0], [3.0, 21.0, 11.0], [17.0, 14.0, 32.0], [0.0, 0.0, 1.0], [9.0, 18.0, 20.0], [0.0, 23.0, 5.0], [35.0, 5.0, 59.0], [13.0, 16.0, 26.0], [23.0, 11.0, 41.0], [1.0, 0.0, 0.0], [37.0, 4.0, 62.0], [43.0, 1.0, 71.0], [5.0, 20.0, 14.0], [11.0, 17.0, 23.0], [19.0, 13.0, 35.0], [25.0, 10.0, 44.0], [31.0, 7.0, 53.0], [0.0, 45.0, 13.0], [33.0, 6.0, 56.0], [39.0, 3.0, 65.0], [7.0, 19.0, 17.0], [41.0, 2.0, 68.0], [1.0, 22.0, 8.0], [15.0, 15.0, 29.0]]
    print(len(vectors))
    find_natural_linear_combination(vectors2)

    find_natural_linear_combination(vectors2)
