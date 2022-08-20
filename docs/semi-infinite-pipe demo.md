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
|(u(x,y) - u_{poi}(x,y), v(x,y) - v_{poi}(x,y))| \le A\exp(-Bx)
$$

for some constant $A, B$. $B$ should be dependent on the width of the pipe, and $A$ should be dependent on the initial boundary conditions. 

> **the flow should return to poiseuille at an exponential rate.**

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
\Delta v(x,y)= \\
-\int_{-1}^1 \Delta v (0,\eta) \frac{\partial G}{\partial \xi}(x,y|0,\eta)d\eta \\
+\int_0^\infty \Delta v (\xi,1) \frac{\partial G}{\partial \xi}(x,y|\xi,1)d\xi \\
+\int_0^\infty \Delta v (\xi,-1) \frac{\partial G}{\partial \xi}(x,y|\xi,-1)d\xi
$$

By the stokes eq $u_x + v_y = 0$, we have $\Delta v = v_{xx} + v_{yy} = v_{xx} - u_{xy}$. 

And notice that  $v_{xx} = 0$ on $\Omega_{2,3}$, So we have $\Delta v$ equals to  $-u_{xy}$ on $\Omega_{2,3}$

The Green's representation simplifies into

$$
\Delta v(x,y)= \\
-\int_{-1}^1 (v_{xx} - u_{xy}) (0,\eta) \frac{\partial G}{\partial \xi}(x,y|0,\eta)d\eta \\
-\int_0^\infty u_{xy} (\xi,1) \frac{\partial G}{\partial \xi}(x,y|\xi,1)d\xi \\
-\int_0^\infty u_{xy} (\xi,-1) \frac{\partial G}{\partial \xi}(x,y|\xi,-1)d\xi
$$


We can do the similar thing for $\Delta u$ to get: 


$$
\Delta u(x,y)
= \\
-\int_{-1}^1 v_{xy}(0,\eta) \frac{\partial G}{\partial \xi}(x,y|0,\eta)d\eta \\
-\int_0^\infty u_{yy}(\xi,1) \frac{\partial G}{\partial \xi}(x,y|\xi,1)d\xi \\
-\int_0^\infty u_{yy}(\xi,-1) \frac{\partial G}{\partial \xi}(x,y|\xi,-1)d\xi
$$


---

$$
\frac{\partial G}{\partial \xi}(x,y|0,\eta) = \\
 \sum_{n\ odd} \exp(\frac{-n\pi x}{2})  \cos(\frac{n\pi y}2) \cos(\frac{n\pi \eta}2) + \sum_{n\ even} \exp(\frac{-n\pi x}{2})  \sin(\frac{n\pi y}2) \sin(\frac{n\pi \eta}2) 
$$
