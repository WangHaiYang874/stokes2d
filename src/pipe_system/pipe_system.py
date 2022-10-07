import networkx as nx
from abstract_pipe import *
from utils import *
from .vertex import Vertex
from .let_index import LetIndex


class PipeSystem:

    pipes: List[RealPipe]
    boundaryPipe: BoundaryPipe
    vertices: List[Vertex]
    edges: List[Tuple[Vertex, Vertex]]
    let2vertex: Dict[LetIndex, Vertex]

    flows: List[Tuple[int, int]]
    flow2edge: Dict[Tuple[(int, int)], Tuple[Vertex, Vertex]]
    edge2flow: Dict[Tuple[Vertex, Vertex], Tuple[int, int]]
    flow2index: Dict[Tuple[int, int], int]
    edge2flowindex: Dict[Tuple[Vertex, Vertex], int]

    n_flows: int

    cycles = List[List[Vertex]]

    A: np.ndarray
    B: np.ndarray

    @property
    def n_flows(self):
        return len(self.flows)

    def get_let(self, letIndex: LetIndex):
        return self.pipes[letIndex.pipeIndex].lets[letIndex.letIndex]

    def __init__(self, pipes: List[RealPipe], boundaryPipe: BoundaryPipe):

        self.pipes = pipes
        self.boundaryPipe = boundaryPipe

        # Vertices

        lets = [(-1, let_index, let) for let_index, let in enumerate(boundaryPipe.lets)] + \
            [(pipe_index, let_index, let)
             for pipe_index, pipe in enumerate(self.pipes)
             for let_index, let in enumerate(pipe.lets)]

        vertices = []

        while lets:
            pipe_index, let_index, let = lets.pop()
            for i, (pipe2_index, let2_index, let2) in enumerate(lets):
                if let.almost_match(let2):
                    break
            if let.almost_match(let2):
                pipe2_index, let2_index, let2 = lets.pop(i)
                l1 = LetIndex(pipe_index, let_index)
                l2 = LetIndex(pipe2_index, let2_index)
                vertices.append(Vertex(l1, l2))
            else:
                raise RuntimeError("There are unmatched pipe",
                                   pipe_index, let_index)

        self.vertices = sorted(vertices)

        # Edges

        self.let2vertex = dict()
        for v in vertices:
            self.let2vertex[v.l1] = v
            self.let2vertex[v.l2] = v

        self.flow2edge = dict()
        for pipe_index, p in enumerate(self.pipes):
            for let_index in range(1, len(p.lets)):
                flow_index = let_index - 1
                self.flow2edge[(pipe_index, flow_index)] = (
                    self.let2vertex[LetIndex(pipe_index, 0)],
                    self.let2vertex[LetIndex(pipe_index, let_index)])

        self.edge2flow = {v: k for k, v in self.flow2edge.items()}
        self.flows = list(self.flow2edge.keys())
        self.edges = list(self.flow2edge.values())
        self.flow2index = {f: i for i, f in enumerate(self.flows)}
        self.edge2flowindex = {
            e: self.flow2index[self.edge2flow[e]] for e in self.edges}

        # Graph

        G = nx.Graph()
        G.add_nodes_from(self.vertices)
        G.add_edges_from(self.edges)
        self.cycles = nx.cycle_basis(G)

        # Equations:

        # we now have enough information to build the linear equation that controls the fluxes.

        # the linear equation Ax = B
        # A: shape = (n_equations, n_flows).
        # n_equations = n_vertices + n_pipes
        # n_flows = n_edges
        # the first few equations are equation of the vertices, which constraint by the conservation of mass (zero-flux)
        # the last few equations are equation of the cycles, which constraint by single-valuedness of the pressure.

        # n_eq = len(self.vertices) + len(self.pipes)

        A = []
        B = []

        for v in self.vertices:

            a = np.zeros(self.n_flows)
            b = 0

            if v.atBdr:

                assert v.l1.pipeIndex == -1  # v.l1 is at the boundary.
                pipe_index, let_index = v.l2

                for flow_index, sign in self.pipes[pipe_index].flux_indices_at_let(let_index):

                    a[self.flow2index[(pipe_index, flow_index)]] = sign

                b = -self.boundaryPipe.lets[v.l1.letIndex].flux

            else:

                assert v.l1[0] != v.l2[0]

                for pipe_index, let_index in [v.l1, v.l2]:
                    if let_index == 0:
                        for j in range(len(self.pipes[pipe_index].lets)-1):
                            a[self.flow2index[(pipe_index, j)]] = 1
                    else:
                        a[self.flow2index[(pipe_index, let_index-1)]] = -1

            A.append(a)
            B.append(b)

        for c in self.cycles:

            a = np.zeros(self.n_flows)
            b = 0

            for v1, v2 in zip(c, c[1:]+c[:1]):

                if (v1, v2) in self.edge2flow:
                    flow = self.edge2flow[(v1, v2)]
                    sign = 1
                elif (v2, v1) in self.edge2flow:
                    flow = self.edge2flow[(v2, v1)]
                    sign = -1
                else:
                    raise RuntimeError("There is no edge between", v1, v2)

                pipe_index, flow_index = flow
                pressure_coef = sign * \
                    self.pipes[pipe_index].pressure_diff_coef_at_let(
                        flow_index + 1)
                for flow_index, coef in enumerate(pressure_coef):
                    a[self.flow2index[(pipe_index, flow_index)]] = coef

            A.append(a)
            B.append(b)

        self.A = np.array(A)
        self.b = np.array(B)

        self.fluxes = np.linalg.lstsq(self.A, self.b, rcond=None)[0]
