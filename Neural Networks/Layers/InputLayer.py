import numpy as np

class InputLayer:
    def __init__(self, n=None, i_n_shape=None):
        self.i_n = i_n_shape
        self.n = n
            
    def forward(self, x):
        if self.n is None:
            self.result = np.copy(x)
        else:
            self.result = np.copy(x).T
            
        return self.result
    
    def backward(self, sensitivity, lr):       
        return None
    
    def get_type(self):
        return 'Input'
    
    def update(self, data, lr):
        pass