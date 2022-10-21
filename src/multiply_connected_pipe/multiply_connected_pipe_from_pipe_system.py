from copy import copy, deepcopy
from email.errors import BoundaryError
from curve.cap import Cap
# from pipe_system.pipe_system import PipeSystem
from .multiply_connected_pipe_from_curves import MultiplyConnectedPipeFromCurves

class MultiplyConnectedPipeFromPipeSystem(MultiplyConnectedPipeFromCurves):

    def __init__(self, pipe_sys: "PipeSystem"):

        caps_to_keep = []

        for v in pipe_sys.vertices:
            if v.atBdr:
                l = v.l1 if isinstance(v.l1, BoundaryError) else v.l2
                caps_to_keep.append((l.pipeIndex,l.letIndex))

        curves = []

        for pipe_index, pipe in enumerate(pipe_sys.pipes):
            
            shift = pipe.shift
            pipe = pipe.prototye
            c_index2l_index = {c:l for l,c in enumerate(pipe.let_index2curve_index)}
            
            for curve_index, curve in enumerate(pipe.exterior_boundary.curves):

                if isinstance(curve, Cap):
                    let_index = c_index2l_index[curve_index]
                    if (pipe_index, let_index) not in caps_to_keep:
                        continue
                    
                curves.append(curve.transformed(shift))
            
            for b in pipe.interior_boundaries:
                curves = curves + b.curves

        super().__init__(curves)