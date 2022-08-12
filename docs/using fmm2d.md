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
