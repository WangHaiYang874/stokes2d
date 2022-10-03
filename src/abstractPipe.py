from tube import *
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True, repr=True, order=True)
class Let:

    x: float
    y: float
    dir: float
    dia: float

    @property
    def pos(self):
        return np.array([self.x, self.y])

    def almost_match(self, other: 'Let', threshold=1e-10):
        return np.max(np.abs([
            np.linalg.norm(self.pos - other.pos),       # same position
            self.dia - other.dia,                       # same diameter
            (self.dir-other.dir) % (2*np.pi) - np.pi    # opposite direction
        ])) <= threshold


@dataclass(frozen=True, repr=True, order=True)
class BoundaryLet(Let):
    flux: float


class AbstractPipe:
    lets: List[Let]

    def __init__(self, lets: List[Let]):
        self.lets = lets


class RealPipe(AbstractPipe):

    boundary: np.ndarray  # shape = (_,2)
    pressure_drops: np.ndarray  # shape = (n_fluxes = n_lets-1, n_lets-1)

    def __init__(self, p: Pipe, shift_x=0, shift_y=0) -> None:

        shift = np.array([shift_x, shift_y])
        self.lets = [Let(*l.matching_pt+shift, l.dir, l.diameter)
                     for l in p.lets]
        self.boundary = p.boundary + shift
        self.pressure_drops = p.pressure_drops

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


class BoundaryPipe(AbstractPipe):
    lets: List[BoundaryLet]

    def __init__(self, lets: List[BoundaryLet]) -> None:
        assert len(lets) > 1
        assert np.sum([l.flux for l in lets]) == 0
        self.lets = lets

    def flux_at_let(self, i: int):
        assert 0 <= i < len(self.lets)
        return self.lets[i].flux
