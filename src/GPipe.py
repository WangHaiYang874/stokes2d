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
                np.abs((self.dir-other.dir) % (2*np.pi) - np.pi), 
                np.abs(self.dia - other.dia)) \
            < 1e-10

class BoundaryLet(Let):
    
    def __init__(self, position, direction, diameter,flux):
        super().__init__(position, direction, diameter)
        self.flux = flux


class GraphComponent:
    
    lets: List[Let]
    def __init__(self) -> None:
        pass

    def fluxes_at_lets(self,fluxes):
        assert len(fluxes) == len(self.lets)-1
        return np.concatenate([ [-np.sum(fluxes)], fluxes])
    
class PipeGraphComponent(GraphComponent):
    lets: List[Let]
    boundary: np.ndarray
    pressure_drops: np.ndarray
    def __init__(self,p:Pipe,shift_x=0,shift_y=0) -> None:
        
        self.lets = []
        shift = np.array([shift_x,shift_y])
        
        for l  in p.lets:
            c = p.curves[l]
            self.lets.append(Let(c.matching_pt + shift, c.dir, c.diameter))
        self.boundary = p.boundary.copy() + shift
        self.pressure_drops = p.pressure_drops
        
    def pressure_drops_at_lets(self, fluxes):
        return np.dot(fluxes, self.pressure_drops)
    
class BoundaryGraphComponent(GraphComponent):
    def __init__(self,lets:List[BoundaryLet]) -> None:
        self.lets = lets
    def fluxes_at_lets(self):
        return np.array([l.flux for l in self.lets])