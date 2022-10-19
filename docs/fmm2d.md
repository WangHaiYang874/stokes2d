
#### hfmm2d
$$
u(x) = \sum_{j=1}^{N} c_{j} H_{0}^{(1)}(k |x-x_{j}|) - d_{j} v_{j}\cdot \nabla  H_{0}^{(1)}(k |x-x_{j}|)
$$

#### rfmm2d

$$
u(x) = \sum_{j=1}^{N} c_{j} \log(\|x-x_{j}\|) + d_{j}v_{j} \cdot \nabla( log(\|x-x_{j}\|) )
$$

#### lfmm2d

$$
u(x) = \sum_{j=1}^{N} c_{j} \log(\|x-x_{j}\|) + d_{j}v_{j} \cdot \nabla(log(\|x-x_{j}\|) )
$$

#### cfmm2d

$$
u(x) = \sum_{j=1}^{N} c_{j} \log(\|x-x_{j}\|) + d_{j}/(x-x_{j})
$$


#### bhfmm2d

$$
    u(x) = \sum_{j=1}^{N}2 c_{j}  \log(|x-x_{j}|) + 
    \overline{c_{j}} (x-x_{j})/(\overline{x-x_{j}}) + d_{j,1}/(x-x_{j}) + d_{j,2}/(\overline{x-x_{j}}) - 
    \overline{d_{j,1}} (x-x_{j})/(\overline{x-x_{j}})^2
$$
