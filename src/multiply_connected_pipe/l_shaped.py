from .smooth_pipe import SmoothPipe
from curve import Line, Cap
from utils import pt


class LShaped(SmoothPipe):
    """L-shaped pipe with two perpendicular branches forming a 90-degree angle.
    
    Creates an L-shaped pipe with customizable branch lengths and pipe radius.
    The L-shape has one horizontal branch and one vertical branch meeting at a corner.
    """

    def __init__(self, l1, l2, r, corner_size=1e-1):
        """Initialize L-shaped pipe.
        
        Args:
            l1: Length of horizontal branch  
            l2: Length of vertical branch
            r: Pipe radius
            corner_size: Corner smoothing parameter
        """
        assert r > 0
        assert l1 > 0
        assert l2 > 0

        # Define points for L-shape: horizontal branch + vertical branch
        # Create a proper L-shape that forms a closed boundary
        pts = [
            pt(-l1, -r),       # Bottom-left (inlet cap start)
            pt(-l1, r),        # Top-left (inlet cap end)
            pt(-r, r),         # Inner corner top-left
            pt(-r, l2),        # Top of vertical branch (outlet cap start)
            pt(r, l2),         # Top-right (outlet cap end)
            pt(r, -r),         # Bottom-right
        ]
        
        # Define curve types for each segment
        curves = [Cap,    # Left cap (inlet)
                 Line,    # Left vertical segment
                 Line,    # Top horizontal segment (short)
                 Cap,     # Top cap (outlet)
                 Line,    # Right vertical segment  
                 Line]    # Bottom horizontal segment

        super().__init__(pts, curves, corner_size)
