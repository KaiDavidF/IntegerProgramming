import gurobipy as gp
from gurobipy import GRB
import numpy as np

class Cone:
    pass


class Lattice:
    pass


class FullLinear:
    def __init__(self, C : Cone, L : Lattice):
        self.C = C
        self.L = L