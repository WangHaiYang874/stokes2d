The kernels and BI eq can be greatly simplified in this specific case, compare to the original case in L. Greengard's paper. In this case, it is 

$$
K_1'(t_i,t_j) = \frac{-h}{\pi} \Im(\frac{d_j}{dt})\\
K_2'(t_i,t_j) = \frac{-h}{2\pi i} (\frac{-d_j}{\bar{dt}} + \frac{\bar{d_j}dt}{\bar{dt}^2}) 
$$ 
where $dt = t_i-t_j$, $d_j$ is the derivative of the parametrization. 

And in the limiting case of $i=j$, the kernels should be 

$$
K_1'(t_i,t_i) = \frac{h}{2\pi}\kappa_i |d_i| \\
K_2'(t_i,t_i) = \frac{-h}{2\pi} \kappa_i d_i^2/ |d_i|
$$



The integral equation is of the following form

$$
(I+K_1+K_2\mathfrak C) \omega = h
$$ where $\mathfrak C$ is a symbol for conjugation. 

This system can be discretized and solved using the Nystorm scheme. After that, we also need to separate it into real and complex parts to solve for it. 

$$
\begin{pmatrix}
I+\Re(K_1+K_2) & \Im(-K_1+K_2) \\
\Im(K_1 + K_2) & I+\Re(K_1-K_2)
\end{pmatrix}
\begin{pmatrix}
\Re \omega\\
\Im \omega
\end{pmatrix} = 
\begin{pmatrix}
\Re h\\
\Im h
\end{pmatrix}
$$

this can be break into several parts

$$
\begin{pmatrix}
I &  \\
 & I
\end{pmatrix} + 
\begin{pmatrix}
\real{K_1} &  \\
 & \real{K_1}
\end{pmatrix} + 
\begin{pmatrix}
\real{K_2} &  \\
 & -\real{K_2}
\end{pmatrix}+
\begin{pmatrix}
  &  \\
 & I
\end{pmatrix}

$$
