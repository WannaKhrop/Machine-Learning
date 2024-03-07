from activations import *
from LossFunctions.LogLoss import LogLoss
from LossFunctions.MSE_Loss import MSE

def get_func(name):
    
    func = None
    
    if name == 'tanh':
        func = Tanh()
    
    elif name == 'relu':
        func = Relu()
        
    elif name == 'linear':
        func = Linear()
    
    elif name == 'softmax':
        func = Softmax()  
        
    elif name == 'leaky_relu':
        func = LeakyRelu()
        
    elif name == 'sigmoid':
        func = Sigmoid()
        
    elif name == 'LogLoss':
        func = LogLoss()
        
    elif name == 'MSE':
        func = MSE()
        
    return func