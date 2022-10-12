import imp
from .multiply_connected_pipe import MultiplyConnectedPipe
from .boundary import Boundary
from utils import *

class MultiplyConnectedPipeFromBoundaries(MultiplyConnectedPipe):
    def __init__(self, boundaries:List[Boundary]):
        super().__init__()
        self.boundaries = boundaries
        