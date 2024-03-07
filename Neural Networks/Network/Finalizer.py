import numpy as np
from functions import get_func

class NetworkFinalizer:
    
    def __init__(self, activation='softmax', loss='LogLoss'):
        
        self.activation = activation
        self.loss = loss
        
    def apply(self, x):
        
        actv_func = get_func(self.activation)
        return actv_func.func(x)
    
    def grad(self, predict, true_value):
        
        gradient = None
        
        if self.activation == 'softmax' and self.loss == 'LogLoss':
            
            # specific case = LogLoss + Softmax
            mask = np.array(true_value, dtype=bool)
            gradient = np.zeros(predict.shape, dtype=float)
            gradient[mask] = predict[mask] - 1.0
            gradient[~mask] = predict[~mask]
            
        else:
            
            loss_func = get_func(self.loss)
            actv_func = get_func(self.activation)
            
            gradient = loss_func.get_grad(predict, real)
            gradient = gradient * activation.deriv(predict)
            
        return gradient
    
    def calculate_loss(self, predict, true_value):
        loss_func = get_func(self.loss)
        return loss_func.func(predict, true_value)