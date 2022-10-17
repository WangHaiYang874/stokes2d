This subroutine computes the N-body Laplace interactions with Cauchy kernel in two dimensions where the interaction kernel is given by log(r) and its gradients. 

$$
u(x) = \sum_{j=1}^{N} c_{j} * \log(|x-x_{j}|) + d_{j}/(x-x_{j}) 
$$
where $c_{j}$ are the charge densities, $d_{j}$ are the dipole strengths, and $x_{j}$ are the source locations.

When $x=x_{m}$, the term corresponding to $x_{m}$ is dropped from the
sum

Args:
- eps: float, precision requested
- sources: float(2,n), source locations (x_{j})
- charges: complex(nd,n) or complex(n), charge densities (c_{j})
- dipstr: complex(nd,n) or complex(n)
        dipole densities (d_{j})
- targets: float(2,nt)
        target locations (x)
- pg:  integer, source eval flag
    - potential at sources evaluated if pg = 1
    - potenial and gradient at sources evaluated if pg=2
    - potential, gradient and hessian at sources evaluated if pg=3
- pgt:  integer, target eval flag
    - potential at targets evaluated if pgt = 1
    - potenial and gradient at targets evaluated if pgt=2
    - potential, gradient and hessian at targets evaluated if pgt=3
- nd:   integer
        number of densities

Returns:
- out.pot: potential at source locations if requested
- out.grad: gradient at source locations if requested
- out.hess: hessian at source locations if requested
- out.pottarg: potential at target locations if requested
- out.gradtarg: gradient at target locations if requested
- out.hesstarg: hessian at target locations if requested

---

$$
u(x) = \sum_{j=1}^{N} c_{j} * \log(|x-x_{j}|) + d_{j}/(x-x_{j}) 
$$

$x_j = z_m$


$$
u(x) = \sum_{m=1}^{M} c_{j} * \log(|x-x_{j}|) + d_{j}/(x-x_{j}) 
$$