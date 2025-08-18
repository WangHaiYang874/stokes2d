from curve.corner import Corner
from curve.cap import Cap
from .abstract_pipe import AbstractPipe
from .let import Let

from multiply_connected_pipe import *
from utils import *


class RealPipe(AbstractPipe):

    # basic data
    prototye: multiply_connected_pipe
    n_lets: int
    n_flows: int
    
    # affine data
    shift: np.ndarray
    rotation: float
    
    # solver data
    pressure_drops: np.ndarray  # shape = (n_fluxes = n_lets-1, n_lets-1)

    # plotting data
    
    # List[np.ndarray with shape = (*, 2) and dtype = float64]
    # open_bdr: np.ndarray
    # closed_boundary: np.ndarray    # shape=(*, 2), dtype=float64.
    # extent: Tuple[float, float, float, float]  # (xmin, xmax, ymin, ymax)
        
    xs: np.ndarray
    ys: np.ndarray
    # interior: np.ndarray 
    u_fields: np.ndarray  # shape=(n_flows, x, y)
    v_fields: np.ndarray  # shape=(n_flows, x, y)
    p_fields: np.ndarray  # shape=(n_flows, x, y)
    o_fields: np.ndarray  # shape=(n_flows, x, y)
    
    def __init__(self, p: MultiplyConnectedPipe, shift_x=0, shift_y=0, rotation=0) -> None:
        """Initialize RealPipe with rotation and shift transformations.
        
        Args:
            p: Base MultiplyConnectedPipe solver
            shift_x, shift_y: Translation parameters  
            rotation: Counter-clockwise rotation angle in radians [0, 2π)
        """
        self.prototye = p
        self.shift = np.array([shift_x, shift_y])
        self.rotation = rotation % (2 * np.pi)  # Normalize to [0, 2π)
        
        # Cache rotation matrix and trig values for efficiency
        if self.rotation != 0:
            self._rotation_matrix = rotation_matrix_2d(self.rotation)
            self._cos_theta = np.cos(self.rotation)
            self._sin_theta = np.sin(self.rotation)
        else:
            self._rotation_matrix = None
            self._cos_theta = 1.0
            self._sin_theta = 0.0
    @property
    def n_lets(self):return len(self.lets)
    @property
    def n_flows(self): return self.n_lets - 1
    @property
    def lets(self):
        """Get connection points with rotation and shift applied."""
        rotated_lets = []
        for let in self.prototye.lets:
            if self.rotation != 0:
                # Step 1: Rotate position
                rotated_pos = rotate_point_2d(let.matching_pt, self.rotation)
                
                # Step 2: Rotate direction  
                rotated_dir = (let.dir + self.rotation) % (2 * np.pi)
                
                # Step 3: Apply shift
                final_pos = rotated_pos + self.shift
            else:
                # No rotation case - just apply shift
                rotated_dir = let.dir
                final_pos = let.matching_pt + self.shift
                
            rotated_lets.append(Let(final_pos[0], final_pos[1], rotated_dir, let.diameter))
        
        return rotated_lets

    # @property
    # def closed_boundary(self):
    #     return self.prototye.closed_plyg_bdr + self.shift

    # @property
    # def open_bdr(self):
        # return [i+self.shift for i in self.prototye.open_bdr]
    # @property
    # def extent(self):
    #     return self.prototye.extent + self.shift

    @property
    def pressure_drops(self):
        return self.prototye.pressure_drops

    @property
    def pts(self):
        pts = np.array([self.prototye.xs, self.prototye.ys]).T
        pts = self.transform(pts)
        return pts

    @property  
    def xs(self):
        """Get x-coordinates with rotation and shift applied."""
        return self.pts[:,0]

    @property
    def ys(self):
        """Get y-coordinates with rotation and shift applied.""" 
        return self.pts[:,1]
    
    # @property
    # def interior(self):
    #     return self.prototye.interior
    
    @property
    def u_fields(self):
        """Get u-velocity field with rotation applied."""
        if self.rotation != 0:
            u_rot, _ = rotate_vector_field_2d(self.prototye.u_fields, self.prototye.v_fields, self.rotation)
            return u_rot
        else:
            return self.prototye.u_fields

    @property  
    def v_fields(self):
        """Get v-velocity field with rotation applied."""
        if self.rotation != 0:
            _, v_rot = rotate_vector_field_2d(self.prototye.u_fields, self.prototye.v_fields, self.rotation)
            return v_rot
        else:
            return self.prototye.v_fields

    @property
    def p_fields(self):
        """Get pressure field (unchanged by rotation)."""
        return self.prototye.p_fields
        
    @property
    def o_fields(self):
        """Get vorticity field (unchanged by rotation)."""
        return self.prototye.o_fields
    
    # def move(self,shift_x,shift_y):
    #     self.shift += np.array([shift_x, shift_y])
        
    def flux_indices_at_let(self, i: int):

        assert 0 <= i < len(self.lets)

        if i == 0:
            return [(flow_index, 1) for flow_index in range(len(self.lets)-1)]
        return [(i-1, -1)]

    def pressure_diff_coef_at_let(self, i: int):

        assert 0 <= i < len(self.lets)
        if i == 0:
            return np.zeros(len(self.lets)-1)
        else:
            return self.pressure_drops[:, i-1]

    def pressure_at_let(self, fluxes:np.ndarray, eval_let:int, base_let:int=0, base_pressure:float=0):
        
        assert eval_let in range(len(self.lets))
        assert base_let in range(len(self.lets))
        assert fluxes.shape == (len(self.lets)-1,)
        assert fluxes.dtype == np.float64        
        
        if eval_let == base_let:
            return base_pressure
        
        pressure_diff = np.concatenate([[0],fluxes@self.pressure_drops])
        pressure_diff = pressure_diff[eval_let] - pressure_diff[base_let]
        return base_pressure + pressure_diff
    
    def fields_with_fluxes(self, fluxes, base_let_index, base_pressure):
        """Get fields with rotation applied to velocity components.
        
        Args:
            fluxes: Flow rates for each flow scenario
            base_let_index: Reference let for pressure calculation
            base_pressure: Reference pressure value
            
        Returns:
            u, v, p, o: Velocity, pressure, and vorticity fields with transformations applied
        """
        assert isinstance(fluxes, np.ndarray)
        assert fluxes.ndim == 1
        assert len(fluxes) == self.n_flows

        # Get base fields from prototype
        u_base = fluxes @ self.prototye.u_fields
        v_base = fluxes @ self.prototye.v_fields
        p = fluxes @ self.prototye.p_fields
        o = fluxes @ self.prototye.o_fields

        # Apply rotation to velocity fields
        if self.rotation != 0:
            u, v = rotate_vector_field_2d(u_base, v_base, self.rotation)
        else:
            u, v = u_base, v_base

        # Handle pressure offset (unchanged by rotation)
        if base_let_index == 0:
            curr_pressure = 0
        else:
            curr_pressure = (fluxes @ self.pressure_drops)[base_let_index-1]

        p = p - curr_pressure + base_pressure

        return u, v, p, o
    
    
    def transform(self, pts):
        
        shift = self.shift
        rotation = self.rotation
        
        assert pts.ndim == 2
        assert pts.shape[1] == 2
        
        pts = rotate_point_2d(pts.T, rotation).T + shift[None,:]
        return pts
        
    
    def plot_plyg_bdr(self, ax, color='k'):
        
        curves = self.prototye.exterior_boundary.curves
        for c1,c2 in zip(curves,curves[1:]+[curves[0]]):
            if isinstance(c1,Corner):
                pts = [c1.start_pt] + [[c1.x[i], c1.y[i]] for i in range(len(c1.a))] + [c2.start_pt]
                pts = np.array(pts)
                pts = self.transform(pts)                
                ax.plot(pts[:,0],pts[:,1],'-',linewidth=2,color=color)
                ax.plot(pts[:,0],pts[:,1],'-',linewidth=4,color=color)
                
            elif isinstance(c1,Cap):
                pts = [c1.start_pt] + [c2.start_pt]
                pts = np.array(pts)
                pts = self.transform(pts)
                ax.plot(pts[:,0],pts[:,1],'--',linewidth=1,color=color)
            else: 
                pts = [c1.start_pt] + [c2.start_pt]
                pts = np.array(pts)
                pts = self.transform(pts)
                ax.plot(pts[:,0],pts[:,1],'-',linewidth=2,color=color)
                ax.plot(pts[:,0],pts[:,1],'-',linewidth=4,color=color)
            
    
    # def clean_prototype(self):
    #     # TODO: clean the unimportant fields of this prototype.
    #     pass
    