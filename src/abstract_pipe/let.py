from dataclasses import dataclass
import numpy as np
from utils import THRESHOLD

@dataclass(frozen=True, repr=True, order=True)
class Let:

    x: float
    y: float
    dir: float
    dia: float

    @property
    def pos(self):
        return np.array([self.x, self.y])

    def almost_match(self, other: 'Let', threshold=THRESHOLD):
        return np.max(np.abs([
            np.linalg.norm(self.pos - other.pos),       # same position
            self.dia - other.dia,                       # same diameter
            (self.dir-other.dir) % (2*np.pi) - np.pi    # opposite direction
        ])) <= threshold

@dataclass(frozen=True, repr=True, order=True)
class BoundaryLet(Let):
    flux: float
