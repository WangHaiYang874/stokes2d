from .abstract_pipe import AbstractPipe
from utils import *
from .let import BoundaryLet

class BoundaryPipe(AbstractPipe):
    lets: List[BoundaryLet]

    def __init__(self, lets: List[BoundaryLet]) -> None:
        assert len(lets) > 1
        assert np.sum([l.flux for l in lets]) == 0
        self.lets = lets

    def flux_at_let(self, i: int):
        assert 0 <= i < len(self.lets)
        return self.lets[i].flux
