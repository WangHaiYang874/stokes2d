from time import time

class Callback:
    
    def __init__(self):
        self.residuals = []
        self.t = time()
        
    def __call__(self, prnorm):
        self.residuals.append(prnorm)
        if len(self.residuals) % 20 == 0:
            print(f"\tresidual = {prnorm},\titer = {len(self.residuals)},\ttime = {(time() - self.t)//60} mins")