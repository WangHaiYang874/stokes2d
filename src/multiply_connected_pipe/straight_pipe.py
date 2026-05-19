from .smooth_pipe import SmoothPipe
from utils import pt
from curve import Line, Cap

class StraightPipe(SmoothPipe):
    def __init__(self,l,r,) -> None:
        
        assert r>0
        assert l>0
        
        
        pts = [
            pt(-l/2,-r),
            pt(-l/2,r),
            pt(l/2,r),
            pt(l/2,-r),
        ]
        
        curves = [
            Cap,
            Line,
            Cap,
            Line,
        ]
        
        super().__init__(pts,curves,corner_size=1)
    