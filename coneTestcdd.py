import cdd

print("Representing a cone by building the Matrix row-by-row.")

# Define the rays for the cone in homogeneous form.
# Each ray is represented as [0, p1, p2].
rays = [
    [0, 1, 0],  # Ray in the direction (1, 0)
    [0, 0, 1]   # Ray in the direction (0, 1)
]

# Create an empty cdd Matrix.
ray_matrix = cdd.Matrix()  # No arguments allowed in your version.
ray_matrix.number_type = 'fraction'  # Use exact (fractional) arithmetic.

# Append each ray to the matrix.
for ray in rays:
    ray_matrix.append(ray)

# Specify that the rows in the matrix are generators (i.e. rays).
ray_matrix.rep_type = cdd.RepType.GENERATOR

# Build the polyhedron (which, in this case, is a cone) from the V–representation.
cone = cdd.Polyhedron(ray_matrix)

# Get the H–representation (inequalities) for the cone.
inequalities = cone.get_inequalities()

print("Inequalities representation of the cone:")
for row in inequalities:
    # Each row is in homogeneous form: [b, a1, a2],
    # representing the inequality: b + a1*x + a2*y >= 0.
    print(list(row))
