import numpy as np

class MatVec:
    
    K1:np.ndarray
    K2:np.ndarray 
    
    def __init__(self,K1,K2) -> None:
        self.K1 = K1
        self.K2 = K2
    
    @property
    def n(self):
        return self.K1.shape[0]
    
    def __call__(self,omega_sep):
        omega = omega_sep[:self.n] + 1j*omega_sep[self.n:]
        h = omega + self.K1@omega + self.K2@(omega.conjugate())
        return np.concatenate([h.real, h.imag])
        