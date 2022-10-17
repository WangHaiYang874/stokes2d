At Aug 10th 2022, professor Greengard wrote

> The basic idea is:
> $$\sum_k \frac {a_k (t_k-t_j)}{\overline{(t_k-t_j)}^2} 
    = \sum_{k} \overline{ (\frac{\overline{a_k(t_k-t_j)}}{(t_k-t_j)^2})} $$
> $$ = \sum_{k} \overline{\frac{\overline{a_kt_k}}{(t_k-t_j)^2}} 
     - \overline{t_j}\sum_k \overline{\frac{\overline{a_k}}{(t_k-t_j)^2}}$$
> These sums are derivatives of calls to $\sum_{k} q_k/(z_k-z_j)$ 
> There is some loss of precision with this formula (as it involves more singular terms than are actually present).
> That's why we have more carefully built biharmonic FMMs but this should be good enough for now.
> I'll also check if there is a more stable python wrapped 2D Stokes library.


What I am really going to do are calling things with charges at the sources, instead of the target. So I might need to get around it somehow. 


# installing fmm2d for python

On a Ubuntu environment, do the following, with the prerequisites of build-essential, gcc, gfortran, etc: 
```
git clone https://github.com/flatironinstitute/fmm2d.git
cd fmm2d
make install PREFIX="somewhere that doesn't require sudo"
# add the complied files into the dynamic link path
make python
```
now one should be able to use `fmm2dpy` as a python library in jupyter notebook, with the correct python environment. 

# usage for this project

for cauchy-type integrals use **cfmm2d**:

$$
u(x) = \sum_{j=1}^{N} c_{j} * log(|x-x_{j}|) + d_{j}/(x-x_{j})
$$

Or don't use fmm at all. Why bother using fmm when I can just run this whole thing on a huge server? 

#### first implementation using fmm
``` python
# commit 73b89c1. 
def build_A_fmm(da,dt_da,k):
    
    dt = dt_da * da
    K1_diagonal = k * np.abs(dt)        / ( 2 * np.pi)
    K2_diagonal = k * dt_da * dt        / (-2 * np.pi * np.abs(dt_da))
    n = len(da)

    def A_fmm(omega_sep):

        omega = omega_sep[:n] + 1j * omega_sep[n:]
        ret = omega + self.K1_fmm(omega) + self.K2_fmm(omega.conj())
        return np.concatenate((ret.real(), ret.imag()))

    self.A_fmm = LinearOperator(dtype=np.float64, shape=(2 * n, 2 * n), matvec=A_fmm)

def K2_fmm(da,dt_da,t,omega):
    
    sources = np.array([t.real, t.imag])
    dt = dt_da * da
    eps = 1e-16
    K2_diagonal

    K21 = fmm.cfmm2d(eps=eps,
                        sources=sources,
                        dipstr=- np.conjugate(dt*omega)/(2j*np.pi),
                        pg=1
                        ).pot.conjugate()
    K221 = t*fmm.cfmm2d(eps=eps,
                        sources=sources,
                        dipstr=- dt * omega.conjugate()/(2j*np.pi),
                        pg=2
                        ).grad.conjugate()
    K222 = fmm.cfmm2d(eps=eps,
                        sources=sources,
                        dipstr=dt * omega.conjugate() * t.conjugate()/(2j*np.pi),
                        pg=2).grad.conjugate()
    # diagonal elements
    return K21 + K221 + K222 + K2_diagonal * omega

def compute_velocity_fmm(self, z, omega):
    '''
    this only support the case when z is a 1-d numpy array
    '''


def phi和d_phi(t,z,dt,omega):
    result = fmm.cfmm2d(eps=eps,
                sources=np.array([t.real(), t.imag()]),
                charges=np.zeros_like(t),
                dipstr=omega * dt,
                targets=np.array([z.real(), z.imag()]),
                pgt=2)

    phi = phi.pottarg / (-2j * np.pi)
    d_phi = phi.gradtarg / (-2j * np.pi)

def psi(t,z,dt,omega)

    psi1 = fmm.cfmm2d(eps=eps,
                        sources=sources,
                        charges=charges,
                        dipstr=np.real(omega.conjugate() * dt),
                        targets=targets,
                        pgt=1).pottarg / (-1j * np.pi)

    psi2 = fmm.cfmm2d(eps=eps,
                        sources=sources,
                        charges=charges,
                        dipstr=np.real(np.conjugate(t) * omega * dt),
                        targets=targets,
                        pgt=2).gradtarg / (2j * np.pi)

    psi = psi1 + psi2
```

#### Another implementation of fmm

```python
class k1_fmm:

    dt: np.ndarray
    t: np.ndarray
    boundary_sources: np.ndarray

    n_boundaries: int
    indices_of_boundary: List[Tuple[int, int]]
    singular_sources: np.ndarray

    @property
    def boundary_sources(self):
        return np.array([self.t.real, self.t.imag])

    @property
    def n_boundaries(self):
        return len(self.indices_of_boundary)

    def __init__(self, t: np.array, da: np.ndarray, dt: np.ndarray, k: np.ndarray,
                 singular_sources: np.ndarray,
                 indices_of_boundary: List[Tuple[int, int]]) -> None:

        self.t = t
        self.da = da
        self.dt = dt
        self.diagonal = k * np.abs(dt) / (2 * np.pi)
        self.singular_sources = singular_sources
        self.indices_of_boundary = indices_of_boundary

    def __call__(self, omega):

        # here are the non-singualr terms
        first_term = cfmm2d(eps=FMM_EPS, sources=self.boundary_sources,
                            dipstr=self.dt*omega/(-2j*np.pi), pg=1).pot
        second_term = cfmm2d(eps=FMM_EPS, sources=self.boundary_sources,
                             dipstr=self.dt*(omega.conj())/(-2j*np.pi), pg=1).pot.conj()
        diagonal_term = self.diagonal*omega

        non_singular_term = first_term + second_term + diagonal_term

        if self.n_boundaries == 1:
            return non_singular_term

        # here are the singular source terms

        third_term = cfmm2d(eps=FMM_EPS,
                            sources=self.singular_sources,
                            charges=self.singular_density(
                                2*np.abs(self.dt)*omega),
                            targets=self.boundary_sources,
                            pgt=1).pottarg

        fourth_term = cfmm2d(eps=FMM_EPS,
                            sources=self.singular_sources,
                            dipstr=self.singular_density(
                                -1j*self.dt*omega.conj()),
                            targets=self.boundary_sources,
                            pgt=1).pottarg.conj()

        singular_term = third_term + fourth_term

        return non_singular_term + singular_term

    def singular_density(self, a):

        ret = []
        for m in range(1, self.n_boundaries):
            start, end = self.indices_of_boundary[m]
            ret.append(np.sum(a[start:end]))

        return np.array(ret)

class k2_fmm:

    dt: np.ndarray
    t: np.ndarray
    boundary_sources: np.ndarray

    n_boundaries: int = 1
    indices_of_boundary: List[Tuple[int, int]]
    singular_sources: np.ndarray

    @property
    def boundary_sources(self):
        return np.array([self.t.real, self.t.imag])

    @property
    def n_boundaries(self):
        return len(self.indices_of_boundary)

    def __init__(self, t: np.array, da: np.ndarray, dt: np.ndarray, dt_da: np.ndarray, k: np.ndarray,
                 singular_sources: np.ndarray,
                 indices_of_boundary: List[Tuple[int, int]]) -> None:

        self.t = t
        self.da = da
        self.dt = dt
        self.diagonal = -k*dt_da*dt/(2*np.pi*np.abs(dt_da))
        self.singular_sources = singular_sources
        self.indices_of_boundary = indices_of_boundary

    def __call__(self, omega_conj):

        # here are the non-singualr terms
        
        first_term = cfmm2d(eps=FMM_EPS,sources=self.boundary_sources,
                        dipstr=-np.conjugate(self.dt*omega_conj)/(2j*np.pi),
                        pg=1
                        ).pot.conjugate()
        second_term = self.t*cfmm2d(eps=FMM_EPS,
                        sources=self.boundary_sources,
                        dipstr=-self.dt*omega_conj.conj()/(2j*np.pi),
                        pg=2
                        ).grad.conjugate()
        
        third_term = cfmm2d(eps=FMM_EPS,
                        sources=self.boundary_sources,
                        dipstr=self.dt*omega_conj.conj()*self.t.conj()/(2j*np.pi),
                        pg=2).grad.conjugate()
        
        diagonal_term = self.diagonal*omega_conj

        non_singular_term = first_term + second_term + third_term + diagonal_term

        if self.n_boundaries == 1:
            return non_singular_term

        # here are the singular source terms

        fourth_term_dipstr = []
        for m in range(1,self.n_boundaries):
            start, end = self.indices_of_boundary[m]
            zm = self.singular_sources[:,m-1]
            zm = zm[0] + 1j*zm[1]
            dt = self.dt[start:end]
            omega = omega_conj[start:end].conj()
            fourth_term_dipstr.append(np.sum(
                (1j*dt.conj() - np.abs(dt)*np.conj(zm))*omega))
        fourth_term_dipstr = np.array(fourth_term_dipstr)
        fourth_term = cfmm2d(eps=FMM_EPS,
                             sources=self.singular_sources,
                             dipstr=fourth_term_dipstr,
                             targets=self.boundary_sources,
                             pgt=1).pottarg.conj()

        fifth_term = cfmm2d(eps=FMM_EPS,
                            sources=self.singular_sources,
                            dipstr=self.singular_density(
                                np.abs(self.dt)*omega_conj.conj()),
                            targets=self.boundary_sources,
                            pgt=1).pottarg.conj()*self.t

        singular_term = fourth_term + fifth_term

        return non_singular_term + singular_term
    
    def singular_density(self, a):

        ret = []
        for m in range(1, self.n_boundaries):
            start, end = self.indices_of_boundary[m]
            ret.append(np.sum(a[start:end]))
        return np.array(ret)
```