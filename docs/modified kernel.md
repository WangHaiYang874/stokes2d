the singular terms do not need to be modified. and the non-singular terms are 

$$
\phi(z) + z\bar\phi'(z) + \bar\psi(z) = \\
\frac{1}{2\pi i} \int_\Gamma \frac{\omega(\xi)d\xi}{\xi - z}
 + z \overline{(\frac{1}{2\pi i} \int_\Gamma \frac{\omega(\xi)d\xi}{(\xi - z)^2})}
 + \overline{(\frac{1}{2\pi i} \int_\Gamma \frac{\bar\omega d\xi + \omega d\bar\xi}{\xi - z}) - \frac{1}{2\pi i} \int_\Gamma \frac{\bar \xi \omega d\xi}{(\xi - z)^2}} = \\
 \frac{1}{2\pi i} \int_\Gamma \frac{\omega(\xi)d\xi}{\xi - z}
 -  \frac{z}{2\pi i} \int_\Gamma \frac{\overline{ \omega d\xi}}{(\overline{\xi - z})^2}
 - \frac{1}{2\pi i} \int_\Gamma \frac{\bar\omega d\xi + \omega d\bar\xi}{\overline{\xi - z}} 
 + \frac{1}{2\pi i} \int_\Gamma \frac{ \xi \overline{ \omega d\xi}}{(\overline{\xi - z})^2} = \\
\frac{1}{2\pi i}(
    \int_\Gamma \frac{\omega(\xi)d\xi}{\xi - z}
    - z \int_\Gamma \frac{\overline{ \omega d\xi}}{(\overline{\xi - z})^2}
    - \int_\Gamma \frac{\bar\omega d\xi + \omega d\bar\xi}{\overline{\xi - z}} 
    + \int_\Gamma \frac{ \xi \overline{ \omega d\xi}}{(\overline{\xi - z})^2}
)
= \\ 
\frac{1}{2\pi i}(IC(\omega) - z \overline{IH(\omega)} - \overline{IC(\bar\omega)} - \overline{IC(\omega \frac{d\bar\xi }{d\xi})} + \overline{IH(\omega\bar\xi)})
$$

$$
A\omega - z \overline{B\omega} - \overline{A}\omega - \bar A \frac{d\xi}{d\bar \xi} \bar \omega + \bar{B}\xi\bar{\omega} = \\
(A-\bar A) \omega + (-z\bar B -\bar A \frac{d\xi}{d\bar \xi} + \bar B \xi ) \bar\omega
$$