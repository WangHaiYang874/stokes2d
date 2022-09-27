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
