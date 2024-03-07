import numpy as np

class Flatten:
    def __init__(self, i_n_shape):
        self.i_n = i_n_shape
        self.n = np.sum(np.ones(i_n_shape, dtype=int))
            
    def forward(self, x):
        self.result = np.copy(x).reshape((x.shape[0], -1)).T
        return np.copy(self.result)
    
    def backward(self, sensitivity, lr):
        new_shape = (sensitivity.T.shape[0], self.i_n[0], self.i_n[1], self.i_n[2])
        
        sensitivity = sensitivity.T.reshape(new_shape)
        self.result = None
        
        return sensitivity
    
    def update(self, data, lr):
        pass
    
    def get_type(self):
        return 'Flatten'