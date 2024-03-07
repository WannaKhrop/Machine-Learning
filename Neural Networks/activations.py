import numpy as np

class Sigmoid:
    
    def __init__(self, name='sigmoid'):
        self.name = name
        
    def func(self, x):
        
        # positive values and zeros
        pos_values = np.copy(x)
        pos_values = 1.0 / (1.0 + np.exp(-pos_values))
        pos_values[x < 0.0] = 0.0
        
        # negative values
        neg_values = np.copy(x)
        neg_values = np.exp(neg_values) / (1.0 + np.exp(neg_values))
        neg_values[x >= 0.0] = 0.0
        
        # return the sum !!!
        return pos_values + neg_values
        
    def deriv(self, x):
        return x * (1 - x)
    
class Tanh:
    
    def __init__(self, name='tanh'):
        self.name = name
        
    def func(self, x):
        return np.tanh(x)
        
    def deriv(self, x):
        return 1 - x ** 2
    
class Linear:
    
    def __init__(self, name='linear'):
        self.name = name
        
    def func(self, x):
        return x
        
    def deriv(self, x):
        return np.ones(x.shape)
    
class Relu:
    
    def __init__(self, name='relu'):
        self.name = name
        
    def func(self, x):
        tmp = np.copy(x)
        tmp[tmp < 0.0] = 0
        
        return tmp
        
    def deriv(self, x):
        tmp = np.copy(x) > 0
        tmp = np.array(tmp, dtype=int)
        
        return tmp
    
class LeakyRelu:
    
    def __init__(self, name='leaky_relu', coeff=0.01):
        self.name = name
        self.coeff = coeff
        
    def func(self, x):
        tmp = np.copy(x)
        tmp[tmp < 0.0] *= self.coeff
        
        return tmp
        
    def deriv(self, x):
        tmp = np.copy(x) > 0
        tmp = np.array(tmp, dtype=float)
        tmp[x <= 0.0] = self.coeff 
        
        return tmp
    
class Softmax:
    
    def __init__(self, name='softmax'):
        self.name = name
        
    def func(self, x):
        
        result = x - np.max(x, axis=0)
        
        result = np.exp(result)
        result = result / np.sum(result, axis=0)
        
        return result
        
    def deriv(self, x):          
        return x * (1 - x)