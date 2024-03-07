import numpy as np 

class MSE:
    
    def func(self, predict, real_y):
        
        return np.sum((predict - real_y) ** 2)
    
    def deriv(self, predict, real_y):
        
        return 2 * (predict - real_y) 