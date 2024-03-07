import numpy as np

class ConvTransformer_2D:
## this class transforms convolution operation into a matrix multiplication
## definitely not the most efficient way

## in reality we must use = numpy.lib.stride_tricks.as_strided - method
    def __init__(self, input_size, kernel_size):

        d1, d2, d3 = input_size[-1], input_size[-2], input_size[-3]
        n, m, k = kernel_size[-1], kernel_size[-2], kernel_size[-3]

        if k != d3:
            raise Exception('Convolutional Transformer is 2D. Error in dimensions !!!')

        ## shapes for operations (frwd conv, backwd_conv, back_prop_conv)
        self.conv_shape = (d2 - m + 1, d1 - n + 1, k, m, n)
        self.conv_shape_bkwd = (m, n, k, d2 - m + 1, d1 - n + 1)
        self.conv_shape_bprop = (d2 - m + 1, d1 - n + 1, k, m, n)
        
        # indexes for oprations
        self.indexes = self.create_indexes_3D(k, (d2, d1), (m, n))
        self.indexes_bkwd = self.create_indexes_3D(k, (d2, d1), (d2 - m + 1, d1 - n + 1))
        self.indexes_bprop = self.create_indexes_2D((d2 + m - 1, d1 + n - 1), (m, n))

    def create_indexes_3D(self, depth, base2D, kernel2D):
        n, m = kernel2D[-1], kernel2D[-2]
        d1, d2 = base2D[-1], base2D[-2]

        ## column indexes caluculation
        col_index = np.arange(n) + np.arange(d1 - n + 1).reshape(-1, 1)
        col_index = np.tile(col_index, (d2 - m + 1, m * depth))

        ## row indexes calculation
        row_index = np.arange(m) + np.arange(d2 - m + 1).reshape(-1, 1)
        row_index = np.repeat(np.repeat(row_index, d1 - n + 1, 0), n, 1)
        row_index = np.tile(row_index, (1, depth))

        ## deep transformation
        deep_index = np.arange(depth)
        deep_index = np.tile(np.repeat(deep_index, m * n), ((d2 - m + 1) * (d1 - n + 1), 1))

        return (col_index, row_index, deep_index)

    def create_indexes_2D(self, base2D, kernel2D):
        n, m = kernel2D[-1], kernel2D[-2]
        d1, d2 = base2D[-1], base2D[-2]

        ## column indexes caluculation
        col_index = np.arange(n) + np.arange(d1 - n + 1).reshape(-1, 1)
        col_index = np.tile(col_index, (d2 - m + 1, m))

        ## row indexes calculation
        row_index = np.arange(m) + np.arange(d2 - m + 1).reshape(-1, 1)
        row_index = np.repeat(np.repeat(row_index, d1 - n + 1, 0), n, 1)

        return (col_index, row_index)
    
    def frwd_conv(self, x, kernel, bias):
        data = x[:, self.indexes[-1], self.indexes[-2], self.indexes[-3]]
        ## create convolutional split for all images
        data = data.reshape(x.shape[0], self.conv_shape[0], self.conv_shape[1],
                            self.conv_shape[2], self.conv_shape[3], self.conv_shape[4])

        # reshape it to matrix
        data = data.reshape(x.shape[0] * self.conv_shape[0] * self.conv_shape[1],
                            self.conv_shape[2] * self.conv_shape[3] * self.conv_shape[4])

        # we can have several kernels
        multiplicator = kernel.reshape(kernel.shape[0], -1).T
        
        # calculate the result ==> transform the result in the appropriate form
        result = np.dot(data, multiplicator) + bias        
        result = result.reshape(x.shape[0], self.conv_shape[0] * self.conv_shape[1], kernel.shape[0])
        result = np.transpose(result, axes=(0, 2, 1))
        result = result.reshape(x.shape[0], kernel.shape[0], self.conv_shape[0], self.conv_shape[1])

        return result
    
    def drvt_conv(self, x, kernel):
        if(x.shape[0] != kernel.shape[0]):
            raise Exception('Batch size is wrong !!!')

        # create left parameter for convolution 
        data = x[:, self.indexes_bkwd[-1], self.indexes_bkwd[-2], self.indexes_bkwd[-3]]
        data = data.reshape(x.shape[0], self.conv_shape_bkwd[0], self.conv_shape_bkwd[1],
                            self.conv_shape_bkwd[2], self.conv_shape_bkwd[3], self.conv_shape_bkwd[4])
        
        data = np.transpose(data, axes=(1, 2, 0, 3, 4, 5))
        data = data.reshape(data.shape[0], data.shape[1], data.shape[2], data.shape[3], data.shape[4] * data.shape[5])

        # create right parameter for convolution
        multiplicator = kernel.reshape(kernel.shape[0], kernel.shape[1], kernel.shape[2] * kernel.shape[3])
        multiplicator = np.transpose(multiplicator, axes=(0, 2, 1))

        # calculate results 
        d_k = np.sum(np.matmul(data, multiplicator), axis=2)
        d_k = np.transpose(d_k, axes=(3, 2, 0, 1))
        
        d_b = np.sum(kernel, axis=(0, 2, 3))

        return d_k, d_b
    
    def bprop_conv(self, x, kernel):
        
         # create left parameter for convolution
        data = x[:, :, self.indexes_bprop[-1], self.indexes_bprop[-2]]
        data = data.reshape(x.shape[0], x.shape[1], x.shape[2] - kernel.shape[2] + 1,
                            x.shape[3] - kernel.shape[3] + 1, kernel.shape[2], kernel.shape[3])

        data = np.transpose(data, axes=(0, 2, 3, 1, 4, 5))
        data = data.reshape(data.shape[0], data.shape[1], data.shape[2], data.shape[3], data.shape[4] * data.shape[5], 1)

        # create right parameter for convolution
        multiplicator = kernel.reshape(kernel.shape[0], kernel.shape[1], kernel.shape[2] * kernel.shape[3])

        # calculate results
        result = np.sum(np.matmul(multiplicator, data), axis=3)
        result = result.reshape(result.shape[0], result.shape[1], result.shape[2], result.shape[3])
        result = np.transpose(result, axes=(0, 3, 1, 2))

        return result