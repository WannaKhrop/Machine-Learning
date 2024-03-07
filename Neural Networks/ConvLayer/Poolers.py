import numpy as np

# POOLER SUPERCLASS
class Pooler:
    
    def __init__(self, i_n_shape, shape):
        self.shape = shape
        self.i_n_shape = i_n_shape
        self.out_shape = (i_n_shape[0], i_n_shape[1] // shape[0], i_n_shape[2] // shape[1])
    
    def get_view(self, x):
        # dimensions
        n_target, d, h, w = x.shape[0], self.out_shape[0], self.out_shape[1], self.out_shape[2]
        
        # strides
        stride_1 = x.itemsize * x.shape[1] * x.shape[2] * x.shape[3]
        stride_2 = x.itemsize * x.shape[2] * x.shape[3]
        stride_3 = x.itemsize * x.shape[3] * self.shape[0]
        stride_4 = x.itemsize * self.shape[1]
        stride_5 = x.itemsize * x.shape[3]
        stride_6 = x.itemsize
        strides = (stride_1, stride_2, stride_3, stride_4, stride_5, stride_6)
        
        # split as strided
        data = np.lib.stride_tricks.as_strided(x, 
                                               shape=(n_target, d, h, w, self.shape[0], self.shape[1]), 
                                               strides=strides)
        return data
    
    def original_view(self, x):
        
        # dimensions as input !!!
        x = np.transpose(x, axes=(0, 1, 2, 4, 3, 5))
        x = x.reshape(self.input.shape)
        return x
    
    def get_type(self):
        return 'Pooler'
    
    def update(self, data, lr):
        pass
    
# MAX POOLER
class MaxPooler(Pooler):
    
    def __init__(self, i_n_shape, shape):
        super().__init__(i_n_shape, shape)
    
    def forward(self, x):
        self.input = np.copy(x)
        view = self.get_view(x)
        
        # get results and create a mask for back propagation
        self.result = np.max(view, axis=(4, 5), keepdims=True)
        
        # create mask for back propagation
        # if several values are equal to target, then we will distribute the impact
        self.mask = np.array(self.result == view, dtype=int)
        self.mask = self.mask / np.sum(self.mask, axis=(4, 5), keepdims=True)
        
        # get normal size for result
        self.result = np.squeeze(self.result, axis=(4, 5))
        return np.copy(self.result)
    
    def backward(self, x, lr):
        result = {}
        
        # sensitivity reshape
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3], 1, 1)
        # get sensetivities and reshape them back
        
        x = self.mask * x
        sensitivity = self.original_view(x)
        
        # free memory
        self.input, self.result, self.mask = None, None, None
        return sensitivity

# MIN POOLER
class MinPooler(Pooler):
    
    def __init__(self, i_n_shape, shape):
        super().__init__(i_n_shape, shape)
    
    def forward(self, x):
        self.input = np.copy(x)
        view = self.get_view(x)
        
        # get results and create a mask for back propagation
        self.result = np.min(view, axis=(4, 5), keepdims=True)
        
        # create mask for back propagation
        # if several values are equal to target, then we will distribute the impact
        self.mask = np.array(self.result == view, dtype=int)
        self.mask = self.mask / np.sum(self.mask, axis=(4, 5), keepdims=True)
         
        self.result = np.squeeze(self.result, axis=(4, 5))
        return np.copy(self.result)
    
    def backward(self, x, lr):
        result = {}

        # sensitivity reshape
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3], 1, 1)
        # get sensetivities and reshape them back
        
        x = self.mask * x
        sensitivity = self.original_view(x)
        
        # free memory
        self.input, self.result, self.mask = None, None, None
        return sensitivity

# MEAN POOLER
class MeanPooler(Pooler):
    
    def __init__(self, i_n_shape, shape):
        super().__init__(i_n_shape, shape)
    
    def forward(self, x):
        self.input = np.copy(x)
        
        self.result = np.mean(self.get_view(x), axis=(4, 5))
        return np.copy(self.result)
    
    def backward(self, x, lr):
        result = {}
        
        # sensitivity reshape
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3], 1, 1)
        
        # get sensetivities and reshape them back
        x = np.ones(self.get_view(self.input).shape) * x / (self.shape[0] * self.shape[1])
        sensitivity = self.original_view(x)
        
        # free memory
        self.input, self.result = None, None
        
        return sensitivity
    
    def get_out_shape(self):
        return self.out_shape