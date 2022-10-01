from GPipe import *
import networkx as nx


class Vertex:
    l1: Tuple[int,int]
    l2: Tuple[int,int]
    
    def __init__(self, a,b):
        
        """
        p is the index of the pipe, if p == -1, it is the 'out' pipe. 
        l is the index of the let
        """
        
        self.l1 = a
        self.l2 = b
        
class PipeSystem:
    vertices: List[Vertex]
    
    def __init__(self, gpipes, boundary_lets):
        vertices = []
        
        self.pipes = gpipes
        self.pipes.append(boundary_lets)
        
        pipe_lets = [(i,j,l) for i,p in enumerate(gpipes) for j,l in enumerate(p.lets)]
        bdr_lets = [(-1,i,l) for i,l in enumerate(boundary_lets.lets)]
        lets = pipe_lets + bdr_lets
        
        let2vertex = {}
        
        while lets:
            i,j,l = lets.pop()
            
            for k,(i2,j2,l2) in enumerate(lets):
                if l.match(l2):
                    break
            
            if l.match(l2):
                    
                i2,j2,l2 = lets.pop(k)
                vertex = Vertex((i,j), (i2,j2))
                vertices.append(vertex)
                let2vertex[(i,j)] = vertex
                let2vertex[(i2,j2)] = vertex
            
            else:
                print("No match for let",i,j)
                raise Exception("No matching let found")    
        
        edges = []
        edge2flow = {}
        
        for pipe_index, pipe in enumerate(gpipes):
            let1 = (pipe_index,0)
            for flow_index in range(1,len(pipe.lets)):
                let2 = (pipe_index,flow_index)
                v1 = let2vertex[let1]
                v2 = let2vertex[let2]
                edges.append((v1,v2))
                edge2flow[(v1,v2)] = (pipe_index,flow_index-1)

        
        self.vertices = vertices
        self.edges = edges
        self.let2vertex = let2vertex
        self.edges2flow = edge2flow
    
    @property
    def n_flows(self): return len(self.edges)
    
    @property
    def n_equations(self): return len(self.vertices) + len(self.edges)
    
    def build_b(self):
        b = np.zeros(self.n_equations)
        for i,v in enumerate(self.vertices):
            assert not (v.l1[0] == -1 and v.l2[0] == -1)
            if v.l1[0] == -1:
                b[i] = - self.pipes[v.l1[0]].lets[v.l1[1]].flux
            elif v.l2[0] == -1:
                b[i] = - self.pipes[v.l2[0]].lets[v.l2[1]].flux
            else: pass
        self.b = b

    def build_A(self):
        A = np.zeros((self.n_equations,self.n_flows))
        for i in range(self.n_equations):
            if i < len(self.vertices):
                pipe1,let1 = self.vertices[i].l1
                pipe2,let2 = self.vertices[i].l2
                if -1 in (pipe1,pipe2):
                    pass
                else:
                    
                    
                    
                
                
            else:
                i = i - len(self.vertices)
            

                
    def build_cycles(self):
        
        G = nx.Graph()
        G.add_nodes_from(self.vertices)
        G.add_edges_from(self.edges)
        cycles = nx.cycle_basis(G)
        cycles = list(map(cycle2edge,cycles))
        # I need to transform each cycle into a flow index with a sign. 
        
        flow_indices = []
        signs = []
        
        for cycle in cycles:
            flow_indices_loc = []
            signs_loc = []
            for edge in cycles:
                
                if edge in self.edges:
                    
                    pass
                elif (edge[1],edge[0]) in self.edges:
                    pass
                else:
                    assert False, "Edge not found"
        pass
        
    
    def build_equations(self,fluxes):
        assert fluxes.shape == (len(self.edges),)
        
def cycle2edge(cycle):
    return list(zip(cycle,cycle[1:]+[cycle[0]]))