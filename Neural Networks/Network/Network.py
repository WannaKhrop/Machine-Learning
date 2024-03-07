import numpy as np
from Network.Finalizer import NetworkFinalizer
from matplotlib import pyplot as plt
import time

class Network:
    
    def __init__(self, layers, final_activation='softmax', loss='LogLoss'):
        self.layers = layers
        self.finalizer = NetworkFinalizer(final_activation, loss)

    def f_propagate(self, x_input):
        
        result = x_input
        for layer in self.layers:
            result = layer.forward(result)
            
        return self.finalizer.apply(result)
    
    def b_propagate(self, sensitivity, lr):
        
        ## back propagation over all layers      
        # for each layer we calculate derivatives and move sensetivities back
        for idx, layer in enumerate(self.layers[::-1]):
            sensitivity = layer.backward(sensitivity, lr)

            
    def run_learning(self, x, y, lr):

        # run forward propagation to predict
        predict = self.f_propagate(x)
        # calculate error rate
        learning_error = self.finalizer.calculate_loss(predict, y)
        # get initial sensetivities
        sensitivity = self.finalizer.grad(predict, y)
        # run backpropagation and update parameters
        self.b_propagate(sensitivity, lr)
        
        return learning_error
            
    def stochastic_fit(self, x, y, lr=0.01, batch_size=15, n_rounds=1_000):

        self.error_rate = {}
        counter = 0
        start_time = time.time()
        
        print('Progress:', end=' ')
        for i in range(n_rounds):
            
            indexes = np.random.choice(np.arange(x.shape[0]), size=batch_size, replace=False)
            
            x_check = x[indexes]
            y_check = y[indexes].T
            
            error = self.run_learning(x_check, y_check, lr)
            self.error_rate[i] = error
            
            if i > (n_rounds / 5) * counter:
                    print('{}%'.format((100 * counter) // 5), end=' ')
                    counter += 1
        
        print('100%')
        print('Time = {} sec'.format(round(time.time() - start_time, 2)))
        self.show_error()
                
    def fit(self, x, y, lr=0.01, batch_size=15, n_epoch=30):
        
        self.error_rate = {}
        
        for epoch in range(n_epoch):
            
            error = 0.0
            counter = 0
            start_time = time.time()
            
            print('Epoch #{}:'.format(epoch + 1), end=' ')
            for i in range(x.shape[0] // batch_size + (x.shape[0] % batch_size > 0)):
                
                idx_from = i * batch_size
                idx_to = min(x.shape[0], (i + 1) * batch_size)

                x_check = x[idx_from:idx_to]
                y_check = y[idx_from:idx_to].T
                
                error += self.run_learning(x_check, y_check, lr)
                
                if i * batch_size > (x.shape[0] / 5) * counter:
                    print('{}%'.format(round(100 * counter / 5)), end=' ')
                    counter += 1

            self.error_rate[epoch + 1] = error / (x.shape[0] // batch_size)
            print('100%')
            print('Time = {} sec'.format(round(time.time() - start_time, 2)))
        
        self.show_error()
            
    def show_error(self):
        # just to show the procedure
        plt.figure(figsize=(5, 5))
        plt.plot(list(self.error_rate.keys()), list(self.error_rate.values()), label='Error')
        plt.legend()
        plt.show()
           
    def predict(self, x, batch_size=250):
        
        out_shape = self.layers[-1].get_out_shape()
        out_shape = (x.shape[0], out_shape)
        
        predicts = np.zeros(out_shape, dtype=float)
        
        for i in range(x.shape[0] // batch_size + (x.shape[0] % batch_size > 0)):
            idx_from = i * batch_size
            idx_to = min(x.shape[0], (i + 1) * batch_size)
            
            results = self.f_propagate(x[idx_from:idx_to]).T
            predicts[idx_from:idx_to] = results
        
        return predicts