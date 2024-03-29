from .multiply_connected_pipe import MultiplyConnectedPipe
from curve import Boundary
from utils import *

class MultiplyConnectedPipeFromBoundaries(MultiplyConnectedPipe):
    def __init__(self, boundaries:List[Boundary]):
        super().__init__()
        self.boundaries = sorted(boundaries, key=lambda boundary: np.min(boundary.t.real))
        