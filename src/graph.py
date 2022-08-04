import numpy as np


class vertice:
    def __init__(self,p,theta) -> None:
        self.p = p
        self.theta = theta
        
    def __eq__(self, __o: object) -> bool:
        return self.p == __o.p and self.theta == __o.theta
    def match(self,other):
        return self.p == other.p and (self.theta + other.theta) % (2*np.pi) == 0
    def __hash__(self):
        return hash(self.p)

class edge:
    def __init__(self,v1,v2,pressure_drop,fluxes) -> None:
        self.v1 = v1
        self.v2 = v2
        self.pressure_drop = pressure_drop
        self.fluxes = fluxes
        

class graph:
    def __init__(self) -> None:
        pass
    def add_points(self,points):
        pass
    def add_edges(self,edges):
        pass
    def join_graphs(self,other):
        pass
    def set_initial_conditions(self):
        pass
    def build_equations(self):
        pass
    def solve(self):
        pass
    
    
    