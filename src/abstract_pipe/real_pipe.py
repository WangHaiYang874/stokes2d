from .abstract_pipe import AbstractPipe
from let import Let

from pipe import *
from utils import *


class RealPipe(AbstractPipe):

    prototye: Pipe
    shift: np.ndarray
    rotation: float = 0  # TODO, probably I need affine transformation here...
    boundary: np.ndarray  # shape = (_,2)
    pressure_drops: np.ndarray  # shape = (n_fluxes = n_lets-1, n_lets-1)

    def __init__(self, p: Pipe, shift_x=0, shift_y=0, rotation=0) -> None:

        self.prototye = p
        self.shift = np.array([shift_x, shift_y])

        if rotation != 0:
            raise NotImplementedError('Rotation not implemented yet')

    @property
    def lets(self):
        return [Let(*l.matching_pt+self.shift, l.dir, l.diameter)for l in self.prototye.lets]

    @property
    def boundary(self):
        return self.prototye.boundary + self.shift

    @property
    def pressure_drops(self):
        return self.prototye.pressure_drops

    @property
    def move(self,shift_x,shift_y,rotation):
        if rotation != 0:
            raise NotImplementedError('Rotation not implemented yet')
        self.shift += np.array([shift_x, shift_y])
        
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

    def clean_prototype(self):
        # TODO: clean the unimportant fields of this prototype.
        pass
        