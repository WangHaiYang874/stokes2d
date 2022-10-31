class Callback:
    
    def __init__(self):
        self.residuals = []
        
    def __call__(self, prnorm):
        self.residuals.append(prnorm)
        