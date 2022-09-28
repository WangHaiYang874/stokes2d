from PipeGraph import *

class Vertex:
    l1: Tuple[int,int]
    l2: Tuple[int,int]
    
    def __init__(self, p1,l1,p2,l2):
        
        """
        p is the index of the pipe, if p == -1, it is the 'out' pipe. 
        l is the index of the let
        """
        
        self.l1 = (p1,l1)
        self.l2 = (p2,l2)


class PipeSystem:
    vertices: List[Vertex]
    
    def __init__(self, gpipes, boundary_lets):
        vertices = []
        
        lets = [(i,j,l) for i,p in enumerate(gpipes) for j,l in enumerate(p.lets)]
        
        let2vertex = {}
        
        while lets:
            i,j,l = lets.pop()
            matched = [k for k, (_,_,l2) in enumerate(free_lets) if l.match(l2)]
            if matched:
                k = matched[0]
                i2,j2,_ = free_lets.pop(k)
                vertex = Vertex((i,j), (i2,j2))
                vertices.append(vertex)
                let2vertex[(i,j)] = vertex
                let2vertex[(i2,j2)] = vertex
            else:
                free_lets.append((i,j,l))
        
        if free_lets:
            raise ValueError("There are free lets in the system")
        
        edges = []
        for pipe_index, pipe in enumerate(gpipes):
            for flow_index in range(1,len(pipe.lets)):
                v1 = let2vertex[(pipe_index,0)]
                v2 = let2vertex[(pipe_index,flow_index)]
                edges.append(Edge(v1,v2,pipe_index,flow_index))
        
        self.vertices = vertices
        self.edges = edges
        