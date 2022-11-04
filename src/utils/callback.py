from time import time

class Callback:
    
    def __init__(self):
        self.residuals = []
        self.t = time()
        
    def __call__(self, prnorm):
        self.residuals.append(prnorm)
        if len(self.residuals) % 20 == 0:
            print(f"\tResidual norm: {prnorm} at iteration {len(self.residuals)}, time: {(time() - self.t)/60} mins")