from tube import *
import numpy as np


class Let:
    
    pos: np.ndarray
    dir: float
    dia: float
    
    def __init__(self, position, direction, diameter):
        self.pos = position
        self.dir = direction
        self.dia = diameter
    
    def match(self, other: 'Let'):
        return max(
                np.linalg.norm(self.pos - other.pos),
                np.abs((self.dir-other.dir) % (2*np.pi)), 
                np.abs(self.dia - other.dia)) \
            < 1e-10


class GPipe:
    
    lets: List[Let]
    
    def __init__(self,p:Pipe) -> None:
        
        self.lets = []
        
        for l  in p.lets:
            c = p.curves[l]
            self.lets.append(Let(c.matching_pt, c.dir, c.diameter))
        
        self.flows = [(self.lets[0],o) for o in self.lets[1:]]
        self.boundary = p.boundary.copy()
        self.pressure_drops = p.pressure_drops.copy()
        
    def fluxes_at_lets(fluxes):
        return np.concatenate([ -np.sum(fluxes), fluxes])
    
    def pressure_drops_at_lets(self, fluxes):
        return np.dot(fluxes, self.pressure_drops)