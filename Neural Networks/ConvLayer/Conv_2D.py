import numpy as np
from functions import get_func
from ConvLayer.PaddingTransformer import PaddingTransformer_2D
from ConvLayer.ConvTransformer import ConvTransformer_2D

class Conv2D:
    
    def __init__(self, n_kernel, k_shape, i_n_shape, padding='as_is', activation='sigmoid', lr=None):
        
        # save constants
        self.k_shape = k_shape
        self.n_kernel = n_kernel
        self.i_n_shape = i_n_shape
        self.activ = get_func(activation)
        self.lr = lr
        
        if (padding == 'as_is' and (k_shape[1] % 2 != 1 or k_shape[2] % 2 != 1)):
            raise Exception('The same size is possible only with odd kernel !!!')
            
        self.padding = ((k_shape[1] - 1) // 2, (k_shape[2] - 1) // 2) if padding == 'as_is' else padding 
        
        # init kernels and bias using normal distribution
        self.kernels = np.random.normal(size=(n_kernel, k_shape[0], k_shape[1], k_shape[2]), scale=0.01)
        self.bias = np.random.normal(size=n_kernel, scale=0.01)
        
        # calculate resulting shape
        h = i_n_shape[1] + 2 * self.padding[0] - k_shape[1] + 1
        w = i_n_shape[2] + 2 * self.padding[1] - k_shape[2] + 1
        self.out_shape = (n_kernel, h, w)
        
        # initialize all interesting transformers
        conv_transformer_shape = (i_n_shape[0], i_n_shape[1] + 2 * self.padding[0], i_n_shape[2] + 2 * self.padding[1])
        self.conv_transformer = ConvTransformer_2D(conv_transformer_shape, k_shape)

    def rotate(self, matrix):
        result = np.copy(matrix)
        result = result[:, :, ::-1, ::-1]
        return result
            
    def forward(self, x):
        self.input = np.copy(x)
        self.input = PaddingTransformer_2D.transform(self.input, self.padding)
        self.result = self.conv_transformer.frwd_conv(self.input, self.kernels, self.bias)
        self.result = self.activ.func(self.result)
        
        return np.copy(self.result)
    
    def backward(self, sensitivity, lr):
        
        self.sensitivity = np.copy(sensitivity)
        deriv_vect = self.activ.deriv(self.result)
        sensitivity = sensitivity * deriv_vect
            
        update_data = {}
        update_data['d_kernel'], update_data['d_bias'] = self.conv_transformer.drvt_conv(self.input, sensitivity)
        update_data['d_kernel'] /= self.input.shape[0]
        update_data['d_bias'] /= self.input.shape[0]
        
        # calculate sensitivity and cut off padded part
        sensitivity = self.conv_transformer.bprop_conv(
            PaddingTransformer_2D.transform(sensitivity, (self.k_shape[1] - 1, self.k_shape[2] - 1)), 
            self.rotate(self.kernels)
        )
        sensitivity = PaddingTransformer_2D.reversal_transform(sensitivity, self.padding)
        
        # free memory
        self.input, self.result = None, None
        self.update(update_data, lr)
        
        return sensitivity
        
    def update(self, data, p_lr):
        
        lr = self.lr if self.lr is not None else p_lr
        
        if data != None:
            self.kernels -= lr * data['d_kernel']
            self.bias -= lr * data['d_bias']
            
    def get_out_shape(self):
        return self.out_shape
    
    def get_type(self):
        return 'Convolution'