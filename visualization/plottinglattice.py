#!/usr/bin/env python3
import matplotlib.pyplot as plt
from collections import deque
import itertools

def compute_generated_set(b, periodic_vectors, max_val=30, set_type='semilinear'):
    """
    Computes the set S = b + (combination of periodic vectors) intersected with [0, max_val-1]^2.
    
    If set_type == 'semilinear', S is defined as:
        S = { b + n1*p1 + ... + nk*pk  |  n_i in ℕ }  
    (only nonnegative combinations).
    
    If set_type == 'lattice', S is defined as:
        S = { b + n1*p1 + ... + nk*pk  |  n_i in ℤ }  
    (all integer combinations).
    
    Parameters:
      b: tuple (x, y) for the base point.
      periodic_vectors: list of tuples, each a periodic vector.
      max_val: size of the grid (default 30 gives points 0,...,29 in each coordinate).
      set_type: either 'semilinear' or 'lattice'
      
    Returns:
      A set of tuples (x, y) that lie in the grid.
    """
    points = set()
    q = deque()
    
    # Determine allowed moves.
    if set_type == 'semilinear':
        moves = periodic_vectors
    elif set_type == 'lattice':
        # Allow both the vectors and their negatives.
        moves = periodic_vectors + [(-p[0], -p[1]) for p in periodic_vectors]
    else:
        raise ValueError("set_type must be either 'semilinear' or 'lattice'")
    
    # Start from the base point if it's in the grid.
    if 0 <= b[0] < max_val and 0 <= b[1] < max_val:
        points.add(b)
        q.append(b)
    
    # Breadth-first search over the grid.
    while q:
        current = q.popleft()
        for move in moves:
            new_point = (current[0] + move[0], current[1] + move[1])
            if (0 <= new_point[0] < max_val and 0 <= new_point[1] < max_val
                    and new_point not in points):
                points.add(new_point)
                q.append(new_point)
                
    return points

def visualize_multiple_sets(sets, max_val=30):
    """
    Visualizes multiple sets on the same 30x30 grid.
    
    Each set in 'sets' is defined as a dictionary with:
      - 'b': the base point (tuple)
      - 'periodic_vectors': list of periodic vectors (list of tuples)
      - 'set_type': either 'semilinear' or 'lattice' (default is 'semilinear' if omitted)
      - Optional 'color': a color string
      - Optional 'label': a label for the set (appears in the legend)
      
    Points are plotted with round markers.
    """
    plt.figure(figsize=(8,8))
    ax = plt.gca()
    
    # Use the default color cycle for sets without a specified color.
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_cycle = itertools.cycle(default_colors)
    
    for s in sets:
        b = s['b']
        periodic_vectors = s['periodic_vectors']
        set_type = s.get('set_type', 'semilinear')
        label = s.get('label', None)
        color = s.get('color', next(color_cycle))
        
        points = compute_generated_set(b, periodic_vectors, max_val=max_val, set_type=set_type)
        xs = [pt[0] for pt in points]
        ys = [pt[1] for pt in points]
        plt.scatter(xs, ys, color=color, s=60, marker='o', label=label)
    
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.xticks(range(0, max_val))
    plt.yticks(range(0, max_val))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    if any(s.get('label') for s in sets):
        plt.legend()
    plt.title("Visualization of Semilinear Sets and Lattices")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def main():
    # Define a collection of sets. Each dictionary can specify the type as either
    # 'semilinear' (only nonnegative combinations) or 'lattice' (all integer combinations).
    sets = [
    {'b': (0, 0), 'periodic_vectors': [(1,2),(1,3),(1,4),(1,5)], 'color': 'red', 'label': 'Set 1'},
]
    
    visualize_multiple_sets(sets, max_val=20)

if __name__ == '__main__':
    main()
