import numpy as np

class let:

    pos:  np.ndarray
    dir:  np.float64
    diam: np.float64
    
    def __init__(self,pos,dir,diam) -> None:

        assert pos.shape == (2,)

        self.pos  = pos
        self.dir  = dir
        self.diam = diam
        
    def __eq__(self, other) -> bool:    return self.pos == other.pos and self.dir == other.dir and self.diam == other.diam
    def match(self,other):              return max(np.linalg.norm(self.pos - other.pos), np.abs((self.theta + other.theta) % (2*np.pi)), np.abs(self.diam - other.diam)) < 1e-10

    # def __hash__(self):
    #     return hash(self.p)

class flow:
    inlet: let
    outlet: let
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
    
    
    