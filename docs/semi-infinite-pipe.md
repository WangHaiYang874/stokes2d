# problem statement
to prove the `return to poiseuille`, I would start formulating the following problem. 

Given a seminfinite pipe $\Omega = [0,\infty)\times [-1,1]$, it's boundary is  
$$\partial \Omega = \Omega_{1} \cup \Omega_2 \cup \Omega_3 = [0] \times [-1,1] \cup [0,\infty) \times [1] \cup [0,\infty) \times [-1]$$ 

- $\Omega_1$ is the inlet with boundary conditions of velocity 
    - $$u(0,y) = a(y),\quad v(0,y) = b(y),\quad \forall y \in [-1,1]$$ 
    - $$ a(\pm1) = b(\pm1) = 0 $$
    - and the with a fixed flux of $$\int_{-1}^1 a(y) dy = \frac43$$ which is the flux of the standard poiseuille flow profile $$ u_{poi}(x,y) = 1-y^2,\quad v_{poi}(x,y) = 0$$
- $\Omega_{2,3}$ are the walls of the seminfinite pipe, which have the non-slippery conditions 
    $$
    u(x,\pm1) = v(x,\pm1) = 0, \quad \forall x\in [0,\infty]
    $$
- the boundary values of $u,v$ are denoted by $f,g$. 

Now, what I hope to show is the return to poiseuille phenomenon which can be summarized as 

$$
|(u(x,y) - u_{poi}(x,y), v(x,y) - v_{poi}(x,y))| \le A_f\exp(-Bx)
$$

for some constant $A_f, B$. i.e. **the flow should return to poiseuille at an exponential rate.**

# Green's function for laplacian

The Green's function for laplacian on the semi-infinite pipe can be found in this book (DOI: 10.1201/9781315371412) at page 468. In our case, the Green's function is 

$$
G(x,y|\xi,\eta) = 
\frac1\pi \sum_{n\in \mathbb N, odd} \frac1n(\exp(\frac{-n\pi |x-\xi|}{2}) - \exp(\frac{-n\pi (x+\xi)}{2})) \cos(\frac{n\pi y}2) \cos(\frac{n\pi \eta}2) + \\
\frac1\pi \sum_{n\in \mathbb N, odd} \frac1n(\exp(\frac{-n\pi |x-\xi|}{2}) - \exp(\frac{-n\pi (x+\xi)}{2})) \sin(\frac{n\pi y}2) \sin(\frac{n\pi \eta}2)
$$


The Green's function for the domain $[0,\infty)\times [-R,R]$ is 


$$
G_R(x,y|\xi,\eta) = 
\frac1\pi \sum_{n\in \mathbb N, odd} \frac1n(\exp(\frac{-n\pi |x-\xi|}{2R}) - \exp(\frac{-n\pi (x+\xi)}{2R})) \cos(\frac{n\pi y}{2R}) \cos(\frac{n\pi \eta}{2R}) + \\
\frac1\pi \sum_{n\in \mathbb N, odd} \frac1n(\exp(\frac{-n\pi |x-\xi|}{2R}) - \exp(\frac{-n\pi (x+\xi)}{2R})) \sin(\frac{n\pi y}{2R}) \sin(\frac{n\pi \eta}{2R})
$$

For our problem, it is clear that the smaller the $R$, the faster would the return to poiseuille be. 


# formulations

First let's consider a simpler case with zero flux

$$
a(y) = 0, \forall y \in [-1,1]
$$

then we should expect the flow return to the zero flow. 

We know that each coordinate of the velocity is a harmonic funciton, it is probably best for us to try to solve for $\Delta u, \Delta v$. 

The Green's representation theorem tells us that 

$$
\Delta u(x,y) 
= - \int_{\partial \Omega} \Delta u(\xi,\eta) \frac{\partial G}{\partial \nu}(x,y|\xi,\eta)|dS(\xi,\eta) \\
= \\
-\int_{-1}^1 \Delta u(0,\eta) \frac{\partial G}{\partial \xi}(x,y|0,\eta)|d\eta \\
+\int_0^\infty \Delta u(\xi,1) \frac{\partial G}{\partial \xi}(x,y|\xi,1)|d\xi \\
\quad \ \ +\int_0^\infty \Delta u(\xi,-1) \frac{\partial G}{\partial \xi}(x,y|\xi,-1)|d\xi
$$

notice that $\Delta u = u_{xx} + u_{yy} = -v_{xy} + u_{yy}$ equals to 
- $-v_{x,y}$ on $\Omega_1$
- $u_{yy}$ on $\Omega_{2,3}$

so the above equation simplifies into 


$$
-v_{xy}(x,y) + u_{yy}(x,y)
= \\
\int_{-1}^1 v_{xy}(0,\eta) \frac{\partial G}{\partial \xi}(x,y|0,\eta)|d\eta 
+\int_0^\infty u_{yy}(\xi,1) \frac{\partial G}{\partial \xi}(x,y|\xi,1)|d\xi 
+\int_0^\infty u_{yy}(\xi,-1) \frac{\partial G}{\partial \xi}(x,y|\xi,-1)|d\xi
$$
