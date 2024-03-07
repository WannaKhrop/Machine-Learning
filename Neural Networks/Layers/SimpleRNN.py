from functions import get_func
import numpy as np

class RecurrentLayer:
    
    def __init__(self, i_n, memo_n, out_n, memo_act='leaky_relu', activation='linear', lr=None):
        
        # save shapes
        self.out_shape = out_n
        self.input_shape = i_n
        self.memory_shape = memo_n
        
        # save activations
        self.memory_activation = get_func(memo_act)
        self.out_activation = get_func(activation)
        
        # in case of own learning rate for a layer
        self.lr = lr
        
        # parameters for input and for memory cells
        self.input_matrix = np.random.normal(size=(memo_n, i_n), scale=0.1)
        self.memory_matrix = np.random.normal(size=(memo_n, memo_n), scale=0.1)
        self.memory_bias = np.random.normal(size=(memo_n, 1), scale=0.1)
        
        # parameters for memory cells
        self.out_matrix = np.random.normal(size=(out_n, memo_n), scale=0.1)
        self.out_bias = np.random.normal(size=(out_n, 1), scale=0.1)
        
    def get_type(self):
        return 'Recurrent'
        
    def forward(self, x):
        
        # get the last memory cell and construct the general input            
        self.input_list = []
        self.output_list = []
        self.memory_list = [np.zeros((self.memory_shape, 1), dtype=float)]
        
        for x_elem in x:
            # get memory cell from last iteration
            memo = np.copy(self.memory_list[-1])     
            
            # get the next memory cell
            memo = np.dot(self.memory_matrix, memo) + np.dot(self.input_matrix, x_elem) + self.memory_bias
            memo = self.memory_activation.func(memo)
            self.memory_list.append(np.copy(memo))
            self.input_list.append(np.copy(x_elem))
        
            # get output for this timestep
            out = self.out_activation.func(np.dot(self.out_matrix, memo) + self.out_bias)
            self.output_list.append(np.copy(out))
        
        return self.output_list
        
        
    def backward(self, sensitivities, lr):
        
        update_data = {'d_out_matrix': np.zeros(self.out_matrix.shape),
                       'd_out_bias': np.zeros(self.out_bias.shape),
                       'd_memory_matrix': np.zeros(self.memory_matrix.shape),
                       'd_input_matrix': np.zeros(self.input_matrix.shape),
                       'd_memory_bias': np.zeros(self.memory_bias.shape)}
        
        next_sensitivities = [] # for the next layer in bach propagation
        bptt_sensitivities_memory = [np.zeros((self.memory_shape, 1))] # for this BPTT step for memory
        
        for sensitivity, output, memo in zip(sensitivities[::-1], 
                                                     self.output_list[::-1], 
                                                     self.memory_list[::-1]):
            
            # get derivative from activation
            sensitivity = sensitivity * self.out_activation.deriv(output)
            
            # calculate update for output matrix
            update_data['d_out_matrix'] += np.dot(sensitivity, memo.T)
            update_data['d_out_bias'] +=  np.sum(sensitivity, axis=1).reshape((-1, 1))
            
            # calculate sensitivities to update later memory matrix and for input matrix
            sensitivity = np.dot(self.out_matrix.T, sensitivity) + np.dot(self.memory_matrix.T, bptt_sensitivities_memory[-1])
            sensitivity = sensitivity * self.memory_activation.deriv(memo)
            bptt_sensitivities_memory.append(np.copy(sensitivity))
            
            # for return
            next_sensitivities.append(np.dot(self.input_matrix.T, sensitivity))
            
        # BPTT steps for this layer
        bptt_sensitivities_memory = bptt_sensitivities_memory[::-1]
        
        for sensitivity, memo, x_elem in zip(bptt_sensitivities_memory,
                                                self.memory_list,
                                                self.input_list):
            
            update_data['d_memory_matrix'] += np.dot(sensitivity, memo.T)
            update_data['d_input_matrix'] += np.dot(sensitivity, x_elem.T)
            update_data['d_memory_bias'] += sensitivity
            
        
        self.update(update_data, lr)
        self.input_list, self.output_list, self.memory_list = None, None, None
        
        return next_sensitivities[::-1]
        
    def update(self, data, lr):
        
        lr = self.lr if self.lr is not None else lr
        
        if data != None:
            self.input_matrix -= lr * data['d_input_matrix']
            self.memory_matrix -= lr * data['d_memory_matrix']
            self.out_matrix -= lr * data['d_out_matrix']
            
            self.memory_bias -= lr * data['d_memory_bias']
            self.out_bias -= lr * data['d_out_bias']
            
    def get_out_shape(self):
        return self.out_shape