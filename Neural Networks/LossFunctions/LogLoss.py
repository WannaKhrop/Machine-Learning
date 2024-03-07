import numpy as np

class LogLoss:
    
    def func(self, predict, real_y):
        data = np.copy(predict)
        
        #data = real_y * np.log(data) + (1 - real_y) * np.log(1 - data)
        #data = np.where(np.isnan(data), 0, data)
        #data = np.where(np.isinf(data), -5000.0, data)
        #error = -1.0 * np.sum(data)
        
        # cross_entropy
        error = -1.0 * np.sum(real_y * np.log(data))
        return error
    
    def deriv(self, predict, real_y):
        
        grad = -1.0 * (real_y / predict)
        return grad