import time
import numpy as np

from matplotlib import pyplot as plt
from Network.Finalizer import NetworkFinalizer
from Network.Network import Network

class RecurrentNetwork(Network):
    
    def create_batch(self, x, y, batch_size):
        
        assert type(x) == type(y) and type(x) == list, 'Incorrect types for input parameters'
        
        for x_elem in x:
            if len(x_elem) != len(x[0]):
                raise Exception('Sequences have different sizes. Batches can not be formed')
    
    def fit(self, x, y, lr=0.01, n_epochs=5):
        
        self.error_rate = {}
        
        for epoch in range(n_epochs):
            
            error = 0.0
            counter = 0
            start_time = time.time()
            
            print('Epoch #{}:'.format(epoch + 1), end=' ')
            for i in range(len(x)):

                seq_x, seq_y = x[i], y[i]
                
                error += self.run_learning(seq_x, seq_y, lr)
                
                if i > (len(x) / 5) * counter:
                    print('{}%'.format((100 * counter) // 5), end=' ')
                    counter += 1

            self.error_rate[epoch + 1] = error / len(x)
            print('100%')
            print('Time = {} sec'.format(round(time.time() - start_time, 2)))
        
        self.show_error()
       
    def stochastic_fit(self, x, y, lr=0.01, n_rounds=1_000):
        
        self.error_rate = {}
        counter = 0
        start_time = time.time()
        
        print('Progress:', end=' ')
        for i_round in range(n_rounds):

            idx = np.random.choice(np.arange(len(x)))
            seq_x, seq_y = x[idx], y[idx]
                
            error = self.run_learning(seq_x, seq_y, lr)
                
            if i_round > (n_rounds / 5) * counter:
                    print('{}%'.format((100 * counter) // 5), end=' ')
                    counter += 1

            self.error_rate[i_round + 1] = error
            
        print('100%')
        print('Time = {} sec'.format(round(time.time() - start_time, 2)))
        
        self.show_error()
    
    def run_learning(self, seq_x, seq_y, lr):
        
        # run forward propagation to predict
        predict = self.f_propagate(seq_x)
        
        # calculate error rate
        learning_error = 0.0 
        for p_elem, y in zip(predict, seq_y):
            if y is not None:
                learning_error += self.finalizer.calculate_loss(p_elem, y)
        
        # get initial sensetivities
        sensitivity = []
        for p_elem, y in zip(predict, seq_y):
            if y is not None:
                sensitivity.append(self.finalizer.grad(p_elem, y))
            else:
                sensitivity.append(np.zeros(p_elem.shape))
        
        # run backpropagation and update parameters
        self.b_propagate(sensitivity, lr)
        
        return learning_error
    
    def f_propagate(self, x_input):
        
        result = x_input
        sections = len(x_input)
        
        for layer in self.layers:
            if layer.get_type() == 'Recurrent':
                # if we have recurrent layer, then just feed the input to it
                result = layer.forward(result)
                
            elif layer.get_type() == 'Simple':
                # otherwise, transform input into one array and feed it to layer
                result = layer.forward(np.hstack(result))
                result = np.hsplit(result, sections)
                
            elif layer.get_type() == 'Input':
                pass
                
            elif layer.get_type() == 'Flatten':
                ## in this case each element of result must have dimension (number of pictures, channels, w, h)
                ## so we use np.vstack in this case !!!
                result = layer.forward(np.vstack(result))
                result = np.hsplit(result, sections)
                
            elif layer.get_type() == 'Convolution' or layer.get_type() == 'Pooling':
                ## in this case each element of result must have dimension (number of pictures, channels, w, h)
                ## so we use np.vstack in this case !!!
                result = layer.forward(np.vstack(result))
                result = np.split(result, sections, axis=0)
                
            else:
                raise('Unrecognized Layer Type !!!')
                
            
        result = [self.finalizer.apply(elem) for elem in result]
        return result
        
    def b_propagate(self, sensitivity, lr):
        ## back propagation over all layers      
        # for each layer we calculate derivatives and move sensetivities back
        
        sections = len(sensitivity)
        
        for layer in self.layers[::-1]:
            
            if layer.get_type() == 'Recurrent':
                # Recurrent Layer can handle it itself
                sensitivity = layer.backward(sensitivity, lr)
                
            elif layer.get_type() == 'Simple':
                sensitivity = layer.backward(np.hstack(sensitivity), lr)
                sensitivity = np.hsplit(sensitivity, sections)
                
            elif layer.get_type() == 'Input':
                sensitivity = None
            
            elif layer.get_type() == 'Flatten':
                ## in this case each out element must have dimension (number of pictures, channels, w, h)
                ## so we use np.hstack and np.split in this case !!!
                sensitivity = layer.backward(np.hstack(sensitivity), lr)
                sensitivity = np.split(sensitivity, sections, axis=0)
                
            elif layer.get_type() == 'Convolution' or layer.get_type() == 'Pooling':
                ## in this case each element of result must have dimension (number of pictures, channels, w, h)
                ## so we use np.vstack in this case !!!
                sensitivity = layer.backward(np.vstack(sensitivity), lr)
                sensitivity = np.split(sensitivity, sections, axis=0)
                
            else:
                raise('Unrecognized Layer Type !!!')
                
    def predict(self, x):
    
        predicts = []
        for x_elem in x:
            predicts.append(self.f_propagate(x_elem))
        
        return predicts