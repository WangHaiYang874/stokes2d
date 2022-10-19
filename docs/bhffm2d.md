This subroutine computes the $N$-body biharmonic interactions 
in two dimensions where the interaction kernel is related to the
biharmonic greens function $r^2 log (r)$ and its derivatives

$$
    u(x) = \sum_{j=1}^{N} c_{j} \log(|x-x_{j}|) + 
    \overline{c_{j}} (x-x_{j})/(\overline{x-x_{j}}) + d_{j,1}/(x-x_{j}) - 
    d_{j,2}/(\overline{x-x_{j}}) - 
    \overline{d_{j,1}} (x-x_{j})/(\overline{x-x_{j}})^2
$$

where $c_{j}$ are the charge densities, $d_{j,1}$, $d_{j,2}$ are the dipole strengths,
and $x_{j}$ are the source locations.

When $x=x_{m}$, the term corresponding to $x_{m}$ is dropped from the
sum


Args:
- eps: float
        precision requested
- sources: float(2,n)   
        source locations (x_{j})
- charges: complex(nd,n) or complex(n)
        charge densities (c_{j})
- dipoles: complex(nd,2,n) or complex(2,n)
        dipole densities (d_{j,1}, d_{j,2})
- targets: float(2,nt)
        target locations (x)
- pg:  integer
        source eval flag
        potential at sources evaluated if pg = 1
        potenial and gradient at sources evaluated if pg=2

- pgt:  integer
        target eval flag
        potential at targets evaluated if pgt = 1
        potenial and gradient at targets evaluated if pgt=2

- nd:   integer
        number of densities

Returns:
- out.pot: potential at source locations if requested
- out.grad: gradient at source locations if requested
- out.pottarg: potential at target locations if requested
- out.gradtarg: gradient at target locations if requested
