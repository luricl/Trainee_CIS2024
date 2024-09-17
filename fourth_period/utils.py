import numpy as np

# Defiing an abstract Layer class that will be inherited by all other layers
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_error, learning_rate):
        raise NotImplementedError
    

class FCLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) 
        self.biases = np.random.randn(1, output_size)

    def forward(self, X):
        self.input = X
        self.output = self.biases + np.dot(X, self.weights)

        return self.output
    
    def backward(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        bias_error = output_error

        self.biases -= learning_rate * output_error
        self.weights -= learning_rate * weights_error

        return input_error 
    
class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, X):
        self.input = X
        self.output = self.activation(X)
        return self.output
    
    def backward(self, grad, learning_rate):
        output_error = grad * self.activation_prime(self.input)
        return output_error
