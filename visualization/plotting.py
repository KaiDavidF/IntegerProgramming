#!/usr/bin/env python3
import matplotlib.pyplot as plt
from collections import deque
import itertools

def compute_semilinear_set(b, periodic_vectors, max_val=30):
    """
    Computes the set S = b + (N*p1 + ... + N*pk) for points in the [0, max_val-1]^2 quadrant.
    
    Parameters:
      b: tuple (x, y) representing the base point.
      periodic_vectors: list of tuples, each representing a periodic vector.
      max_val: grid size (default 30 for coordinates 0–29).
      
    Returns:
      A set of tuples (x, y) within the quadrant.
    """
    points = set()
    q = deque()

    # Add the base point if it lies within the grid.
    if 0 <= b[0] < max_val and 0 <= b[1] < max_val:
        points.add(b)
        q.append(b)

    # Breadth-first search: from each point, add every periodic vector.
    while q:
        current = q.popleft()
        for p in periodic_vectors:
            new_point = (current[0] + p[0], current[1] + p[1])
            if (new_point[0] < max_val and new_point[1] < max_val and new_point not in points):
                points.add(new_point)
                q.append(new_point)
    return points

def visualize_multiple_sets(sets, max_val=30):
    """
    Visualizes multiple semilinear sets on the same grid.
    
    Each set is defined as a dictionary with:
      - 'b': the base point (tuple)
      - 'periodic_vectors': list of periodic vectors (list of tuples)
      - Optional 'color': a color string
      - Optional 'label': a label for the set (will appear in the legend)
      
    Points are plotted with round markers.
    """
    plt.figure(figsize=(8,8))
    ax = plt.gca()
    
    # Use the default color cycle for sets with no specified color.
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_cycle = itertools.cycle(default_colors)
    
    for s in sets:
        b = s['b']
        periodic_vectors = s['periodic_vectors']
        label = s.get('label', None)
        color = s.get('color', next(color_cycle))
        
        points = compute_semilinear_set(b, periodic_vectors, max_val=max_val)
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
    plt.title("Visualization of Semilinear Sets: b + (N*p_1 + ... + N*p_k)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def main():
    # Here we pull out two of the periodic elements and thus have to increase the number of linear sets.
    sets = [
    {'b': (0, 0), 'periodic_vectors': [(1,2),(1,5)], 'color': 'purple', 'label': 'Set 0'},
    {'b': (1, 3), 'periodic_vectors': [(1,2),(1,5)], 'color': 'blue', 'label': 'Set 1'},
    {'b': (1, 4), 'periodic_vectors': [(1,2),(1,5)], 'color': 'yellow', 'label': 'Set 2'}
]
    
    sets2 = [
    {'b': (0, 0), 'periodic_vectors': [(1,2),(1,3),(1,4),(1,5)], 'color': 'purple', 'label': 'Set 0'}
]

    
    visualize_multiple_sets(sets, max_val=20)

if __name__ == '__main__':
    main()
