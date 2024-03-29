# problem statement
**return to poiseuille** is formulated as the following. 

Given a seminfinite pipe $\Omega = [0,\infty)\times [-1,1]$, its boundary is  
$$\partial \Omega = \Omega_{1} \cup \Omega_2 \cup \Omega_3 = [0] \times [-1,1] \cup [0,\infty) \times [1] \cup [0,\infty) \times [-1]$$ 

- $\Omega_1$ is the inlet with boundary conditions of velocity 
    - $$u(0,y) = a(y),\quad v(0,y) = b(y),\quad \forall y \in [-1,1]$$ 
    - and the with a fixed flux of $$\int_{-1}^1 a(y) dy = \frac43$$ which is the flux of the standard poiseuille flow profile $$ u_{poi}(x,y) = 1-y^2,\quad v_{poi}(x,y) = 0$$
- $\Omega_{2,3}$ are the walls of the seminfinite pipe, which have the non-slippery conditions 
    $$
    u(x,\pm1) = v(x,\pm1) = 0, \quad \forall x\in [0,\infty]
    $$
    and also forces that 
    $$ a(\pm1) = b(\pm1) = 0 $$


Now, what I hope to show is something of the following form: 
$$
|(u(x,y) - u_{poi}(x,y), v(x,y) - v_{poi}(x,y))| \le A_f\exp(-Bx)
$$

for some constant $A_f, B$. i.e. **the flow should return to poiseuille at an exponential rate.**

# Green's function for laplacian

The Green's function for laplacian on the semi-infinite pipe can be found at page 468 of (DOI: 10.1201/9781315371412). 

In our case, the Green's function is 

$$
G(x,y|\xi,\eta) = 
\frac1\pi \sum_{n\in \mathbb N, odd} \frac1n(\exp(\frac{-n\pi |x-\xi|}{2}) - \exp(\frac{-n\pi (x+\xi)}{2})) \cos(\frac{n\pi y}2) \cos(\frac{n\pi \eta}2) + \\
\frac1\pi \sum_{n\in \mathbb N, odd} \frac1n(\exp(\frac{-n\pi |x-\xi|}{2}) - \exp(\frac{-n\pi (x+\xi)}{2})) \sin(\frac{n\pi y}2) \sin(\frac{n\pi \eta}2)
$$


More generally, the Green's function for the domain $[0,\infty)\times [-R,R]$ is 


$$
G_R(x,y|\xi,\eta) = 
\frac1\pi \sum_{n\in \mathbb N, odd} \frac1n(\exp(\frac{-n\pi |x-\xi|}{2R}) - \exp(\frac{-n\pi (x+\xi)}{2R})) \cos(\frac{n\pi y}{2R}) \cos(\frac{n\pi \eta}{2R}) + \\
\frac1\pi \sum_{n\in \mathbb N, odd} \frac1n(\exp(\frac{-n\pi |x-\xi|}{2R}) - \exp(\frac{-n\pi (x+\xi)}{2R})) \sin(\frac{n\pi y}{2R}) \sin(\frac{n\pi \eta}{2R})
$$

> it is clear that: smaller the $R$, faster the return to poiseuille. 


# A simple investigation 

First let's consider a simple case with **zero flux**

$$
a(y) = u(0,y) = 0,\quad \forall y \in [-1,1]
$$

then we should expect the flow return to the zero flow. 

We know that each coordinate of the velocity is a harmonic funciton, it is probably best for us to try to solve for $\Delta u, \Delta v$. 

The Green's representation theorem tells us that 

$$
\Delta u(x,y) 
= - \int_{\partial \Omega} \Delta u(\xi,\eta) \frac{\partial G}{\partial \nu}(x,y|\xi,\eta)dS(\xi,\eta) \\
= \\
-\int_{-1}^1 \Delta u(0,\eta) \frac{\partial G}{\partial \xi}(x,y|0,\eta)d\eta \\
+\int_0^\infty \Delta u(\xi,1) \frac{\partial G}{\partial \xi}(x,y|\xi,1)d\xi \\
\quad \ \ +\int_0^\infty \Delta u(\xi,-1) \frac{\partial G}{\partial \xi}(x,y|\xi,-1)d\xi
$$

stokes eq is $u_x + v_y = 0$, so we have $\Delta u = u_{xx} + u_{yy} = -v_{xy} + u_{yy} $ equals to 
- $ -v_{x,y}$ on $\Omega_1$
- $ -v_{x,y} + u_{yy}$ on $\Omega_{2,3}$

so the above equation simplifies into 


$$
-v_{xy}(x,y) + u_{yy}(x,y)
= \\
\int_{-1}^1 v_{xy}(0,\eta) \frac{\partial G}{\partial \xi}(x,y|0,\eta)d\eta 
+\int_0^\infty u_{yy}(\xi,1) \frac{\partial G}{\partial \xi}(x,y|\xi,1)d\xi 
+\int_0^\infty u_{yy}(\xi,-1) \frac{\partial G}{\partial \xi}(x,y|\xi,-1)d\xi
$$

We can do the similar thing for $\Delta v$: 

notice that $\Delta v = v_{xx} + v_{yy} = v_{xx} - u_{xy}$ equals to 
- $v_{xx}$ on $\Omega_1$
- $0$ on $\Omega_{2,3}$


$$
v_{xx}(x,y) - u_{xy} = \Delta v(x,y) = -\int_{-1}^1 v_{xx}(0,\eta) \frac{\partial G}{\partial \xi}(x,y|0,\eta)d\eta \\
$$

this equation is much simpler so we can start with this one first. 

It can be calculated that

$$
\frac{\partial G}{\partial \xi}(x,y|0,\eta) = \\
 \sum_{n\ odd} \exp(\frac{-n\pi x}{2})  \cos(\frac{n\pi y}2) \cos(\frac{n\pi \eta}2) + \sum_{n\ even} \exp(\frac{-n\pi x}{2})  \sin(\frac{n\pi y}2) \sin(\frac{n\pi \eta}2) 
$$

This basically says that no matter what is $v_{xx}$, $\Delta v(x,y) = O(\exp(-x/2))$ as $x\to \infty$. 
