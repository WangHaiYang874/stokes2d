notations and equations follow from professor Greengard's [paper on 2D Stokes and Isotropic Elasticity](https://www.sciencedirect.com/science/article/pii/S0021999196901023?via%3Dihub)

## extracting the pressure difference $\nabla p$

To begin with, recall that

$$
\begin{align}
\nabla p 
    &= \begin{pmatrix}
        \frac{\partial p}{\partial x}\\
        \frac{\partial p}{\partial y}\\
        \end{pmatrix} 
    &= \rho\nu \Delta \vec{u} 
    &= \rho\nu \begin{pmatrix}
        \Delta u\\
        \Delta v
        \end{pmatrix}\\
\vec{u}
    &= \begin{pmatrix}
        u\\ v
    \end{pmatrix}
    &= \begin{pmatrix}
        \frac{\partial W}{\partial y}\\
        -\frac{\partial W}{\partial x}
    \end{pmatrix} & 
    \\ 
W(x,y) 
    &= W(x+yi) 
    &= W(z)
    &= \Re(\bar{z} \phi(z) + \chi(z))\\
\phi(z) &= \frac1{2\pi i} \int_\Gamma \frac{\omega(\xi)}{\xi - z} d\xi \\
\psi(z) &= \frac{1}{2\pi i} \int_\Gamma \frac{\bar \omega(\xi)d\xi + \omega(\xi)d\bar{\xi}}{\xi - z}   -\frac1{2\pi i} \int_\Gamma \frac{\bar\xi \omega(\xi)}{(\xi-z)^2} d\xi
\end{align}
$$ $\Re$ means taking the real part. $\phi, \chi$ are two complex analytic function, and $\psi = \chi'$. 

The first equation is Stokes equation. The second is the definition of stream function, which is bi-harmonic and therefore has the Goursat's representation in the third equation. The fourth and fifth equations are Sherman-Lauricella integral equations. Our solver would solve for $\omega$ using the Nystorm method. This means that we are eventually capable of numerically compute the values of $\phi,\psi$ or their derivatives. 


Now we can see that our task is pretty clear, compute the following

$$
\nabla p 
    = \rho\nu \Delta \vec u 
    = \rho\nu \Delta \begin{pmatrix}
        \frac{\partial W}{\partial y}\\
        -\frac{\partial W}{\partial x}
        \end{pmatrix}
    = \rho\nu \begin{pmatrix}
        \frac{\partial^3 W}{\partial y^3} + \frac{\partial^3 W}{\partial x^2 \partial y} \\
        -\frac{\partial^3 W}{\partial x^3} - \frac{\partial^3 W}{\partial x^2 \partial y} \\
        \end{pmatrix}
$$

into an expression of $\phi,\psi$ and their derivatives. It is tricky to compute the partial derivative of $W$ because (a) it involves both $z,\bar z$ and (b) it has a $\Re$. But I guess this is computable anyway... 


First of all, notice that we can exchange partial derivatives with the $\Re$ operator. So we can ignore the $\Re$ operator for now. After some laboring one can discover that 

$$
\begin{align}
\frac{\partial^3 W}{\partial y^3} + \frac{\partial^3 W}{\partial x^2 \partial y} 
    &= \Re(4i\phi'') \\
-\frac{\partial^3 W}{\partial x^3} - \frac{\partial^3 W}{\partial x^2 \partial y}
    &= \Re(-4\phi'')
\end{align}
$$

Despite the laboring computation, the results end up pretty nicely... 

In conclusion, we have that 

$$
\nabla p = -4 \rho\nu\begin{pmatrix}
    \Im(\phi'')\\
    \Re(\phi'')
\end{pmatrix}
$$

This would be quite easy to compute given that 

$$
\phi''(z) = \frac{1}{\pi i} \int_\Gamma \frac{\omega(\xi)}{(\xi-z)^3}d\xi
$$
