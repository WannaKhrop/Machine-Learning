from functions import get_func
import numpy as np

class Layer:
    
    def __init__(self, n, i_n, activation='sigmoid', lr=None):
        self.n = n
        self.i_n = i_n
        self.activ = get_func(activation)
        self.lr = lr
        
        self.matrix = np.random.normal(size=(n, i_n), scale=0.1)
        self.bias = np.random.normal(size=(n, 1), scale=0.1)
            
    def forward(self, x):
        
        self.input = np.copy(x)
        self.result = self.activ.func(np.dot(self.matrix, self.input) + self.bias)
        return np.copy(self.result)
    
    def backward(self, sensitivity, lr):
        
        update_data = {}
        
        deriv_vect = self.activ.deriv(self.result)
        sensitivity = sensitivity * deriv_vect

        update_data['d_matrix'] = np.dot(sensitivity, self.input.T) / self.input.shape[1]
        update_data['d_bias'] =  np.sum(sensitivity, axis=1).reshape((-1, 1)) / self.input.shape[1]

        sensitivity = np.dot(self.matrix.T, sensitivity)
        
        self.input, self.result = None, None
        self.update(update_data, lr)
        
        return sensitivity
    
    def update(self, data, p_lr):
        
        lr = self.lr if self.lr is not None else p_lr
        
        if data != None:
            self.matrix -= lr * data['d_matrix']
            self.bias -= lr * data['d_bias']
            
    def get_out_shape(self):
        return self.n
    
    def get_type(self):
        return 'Simple'