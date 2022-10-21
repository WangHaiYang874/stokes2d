from curve import Boundary
from .multiply_connected_pipe import MultiplyConnectedPipe
import networkx as nx
import numpy as np

class MultiplyConnectedPipeFromCurves(MultiplyConnectedPipe):
    def __init__(self, curves):
        super().__init__()
        
        G = nx.DiGraph()

        edges = [(pt2tuple(c.start_pt),pt2tuple(c.end_pt), {'curve_index':i}) for i,c in enumerate(curves)]
        G.add_edges_from(edges)

        pts = np.array(list(G.nodes))
        pts_cplx = pts[:,0] + 1j*pts[:,1]
        distance = np.abs(pts_cplx[:,None] - pts_cplx[None,:])
        need_to_merge = (distance < 1e-10) & (distance > 0)

        while np.any(need_to_merge):
            i,j = np.array(np.where(need_to_merge)).T[0]
            
            node1 = list(G.nodes)[i]
            node2 = list(G.nodes)[j]

            nx.contracted_nodes(G,node1,node2, self_loops=False, copy=False)
            
            pts = np.array(list(G.nodes))
            pts_cplx = pts[:,0] + 1j*pts[:,1]
            distance = np.abs(pts_cplx[:,None] - pts_cplx[None,:])
            need_to_merge = (distance < 1e-10) & (distance > 0)
            
        assert len(G.nodes) == len(set(G.edges))

        boundaries = []
        
        for c in nx.recursive_simple_cycles(G):
            boundary_curves = []
            for node1,node2 in zip(c, c[1:] + c[:1]):
                boundary_curves.append(curves[G.edges[node1,node2]['curve_index']])
            boundaries.append(boundary_curves)
            
        boundaries = [Boundary(b) for b in boundaries]
        [b.build() for b in boundaries]
        self.boundaries = sorted(boundaries, key=lambda boundary: np.min(boundary.t.real))

def pt2tuple(pt):
    assert pt.shape == (2,)
    return (pt[0],pt[1])
