import numpy as np

class PaddingTransformer_2D:
    
    @staticmethod   
    def transform(x, padding):
        n_targets = x.shape[0]
        h = x.shape[2] + 2 * padding[0]
        w = x.shape[3] + 2 * padding[1]
        d = x.shape[1]
        
        result = np.zeros((n_targets, d, h, w))
        result[:, :, padding[0]:h-padding[0], padding[1]:w-padding[1]] = x
        return result
    
    @staticmethod  
    def reversal_transform(x, padding):
        n_targets = x.shape[0]
        h = x.shape[2] - 2 * padding[0]
        w = x.shape[3] - 2 * padding[1]
        d = x.shape[1]
        
        result = np.zeros((n_targets, d, h, w))
        result = x[:, :, padding[0]:h+padding[0], padding[1]:w+padding[1]]
        return result