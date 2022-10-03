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

# deleted code
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

def K1_fmm(da, dt_da, t, omega):

    sources = np.array([t.real, t.imag])
    dt = dt_da * da
    eps = 1e-16
    K1_diagonal

    K11 = fmm.cfmm2d(eps    =eps,
                        sources=sources,
                        dipstr =dt * omega / (-2j * np.pi),
                        pg     =1
                        ).pot

    K12 = fmm.cfmm2d(eps    =eps,
                        sources=sources,
                        dipstr =dt * omega.conjugate()/(-2j * np.pi),
                        pg=1
                        ).pot.conjugate()

    # ❗ K12 看其实是不是有bug. 为啥只对omega做conjugate? 我懒得回忆了....

    return K11 + K12 + K1_diagonal * omega

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