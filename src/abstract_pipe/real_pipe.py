from .abstract_pipe import AbstractPipe
from .let import Let

from multiply_connected_pipe import *
from utils import *


class RealPipe(AbstractPipe):

    # basic data
    prototye: multiply_connected_pipe
    n_lets: int
    n_flows: int
    
    # affine data
    shift: np.ndarray
    # rotation: float = 0     # TODO
    
    # solver data
    pressure_drops: np.ndarray  # shape = (n_fluxes = n_lets-1, n_lets-1)

    # plotting data
    
    # List[np.ndarray with shape = (*, 2) and dtype = float64]
    # open_bdr: np.ndarray
    # closed_boundary: np.ndarray    # shape=(*, 2), dtype=float64.
    # extent: Tuple[float, float, float, float]  # (xmin, xmax, ymin, ymax)
        
    # xs: np.ndarray
    # ys: np.ndarray
    # interior: np.ndarray 
    # u_fields: np.ndarray  # shape=(n_flows, x, y)
    # v_fields: np.ndarray  # shape=(n_flows, x, y)
    # p_fields: np.ndarray  # shape=(n_flows, x, y)
    # o_fields: np.ndarray  # shape=(n_flows, x, y)
    
    def __init__(self, p: MultiplyConnectedPipe, shift_x=0, shift_y=0, rotation=0) -> None:

        self.prototye = p
        self.n_lets = len(p.lets)
        self.n_flows = self.n_lets - 1
        self.shift = np.array([shift_x, shift_y])
        
        if rotation != 0:
            raise NotImplementedError('Rotation not implemented yet')

    @property
    def lets(self):
        return [Let(*l.matching_pt+self.shift, l.dir, l.diameter)for l in self.prototye.lets]

    # @property
    # def closed_boundary(self):
    #     return self.prototye.closed_plyg_bdr + self.shift

    # @property
    # def open_bdr(self):
        # return [i+self.shift for i in self.prototye.open_bdr]
    # @property
    # def extent(self):
    #     return self.prototye.extent + self.shift

    @property
    def pressure_drops(self):
        return self.prototye.pressure_drops

    # @property
    # def xs(self):
    #     return self.prototye.xs + self.shift[0]
    
    # @property
    # def ys(self):
    #     return self.prototye.ys + self.shift[1]
    
    # @property
    # def interior(self):
    #     return self.prototye.interior
    
    # @property
    # def u_fields(self):
    #     return self.prototye.u_fields
    # @property
    # def v_fields(self):
    #     return self.prototye.v_fields
    # @property
    # def p_fields(self):
    #     return self.prototye.p_fields
    # @property
    # def o_fields(self):
    #     return self.prototye.o_fields
    
    # def move(self,shift_x,shift_y):
    #     self.shift += np.array([shift_x, shift_y])
        
    def flux_indices_at_let(self, i: int):

        assert 0 <= i < len(self.lets)

        if i == 0:
            return [(flow_index, 1) for flow_index in range(len(self.lets)-1)]
        return [(i-1, -1)]

    def pressure_diff_coef_at_let(self, i: int):

        assert 0 <= i < len(self.lets)
        if i == 0:
            return np.zeros(len(self.lets)-1)
        else:
            return self.pressure_drops[:, i-1]

    def pressure_at_let(self, fluxes:np.ndarray, eval_let:int, base_let:int=0, base_pressure:float=0):
        
        assert eval_let in range(len(self.lets))
        assert base_let in range(len(self.lets))
        assert fluxes.shape == (len(self.lets)-1,)
        assert fluxes.dtype == np.float64        
        
        if eval_let == base_let:
            return base_pressure
        
        pressure_diff = np.concatenate([[0],fluxes@self.pressure_drops])
        pressure_diff = pressure_diff[eval_let] - pressure_diff[base_let]
        return base_pressure + pressure_diff
    
    # def fields_with_fluxes(self, fluxes, base_let_index, base_pressure):
        
    #     assert isinstance(fluxes, np.ndarray)
    #     assert fluxes.ndim == 1
    #     assert len(fluxes) == self.n_flows

    #     u = fluxes@self.u_fields
    #     v = fluxes@self.v_fields
    #     p = fluxes@self.p_fields
    #     o = fluxes@self.o_fields

    #     if base_let_index == 0:
    #         curr_pressure = 0
    #     else:
    #         curr_pressure = (fluxes@self.pressure_drops)[base_let_index-1]

    #     p = p - curr_pressure + base_pressure

    #     return u, v, p, o
    
    # def clean_prototype(self):
    #     # TODO: clean the unimportant fields of this prototype.
    #     pass
    