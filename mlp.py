import numpy as np
from random import random

class MLP:

    def __init__(self, num_inputs = 3, num_hidden = [3, 5], num_outputs = 2):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]

        # initiate random weights
        weights = []
        for i in range(len(layers)-1):  
            w = np.random.rand(layers[i], layers[i+1])
            weights.append(w)
        self.weights = weights

        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

        derivatives = []
        for i in range(len(layers)-1):
            d = np.zeros((layers[i], layers[i+1]))
            derivatives.append(d)
        self.derivatives = derivatives

    
    def forward_propagate(self, inputs):
        activations = inputs
        self.activations[0] = activations

        for i, w in enumerate(self.weights):
            # calc net inputs
            net_inputs = np.dot(activations, w) #activations @ w

            # calc the activations
            activations = self._sigmoid(net_inputs)
            self.activations[i+1] = activations

        return activations
    
    def back_propagate(self, error, verbose = False):

        # dE/dW[i] = (y-a[i+1]) sig'(h[i+1]) a[i]
        # sig'(h[i+1]) = sig(h[i+1]) * (1 - sig(h[i+1]))
        # sig(h[i+1]) = a[i+1]

        # dE/dW[i-1] = (dE/dW[i]) sig'(h[i]) a[i]

        for i in reversed(range(len(self.derivatives))):
            # get acitvaion from prev layer
            activations = self.activations[i+1]
            delta = error * self._sigmoid_derivative(activations) # convert [1, 2] to [[1, 2]]
            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            # activation of current layer
            curr_activations = self.activations[i] # convert [1, 2] to [[1], [2]]
            curr_activations_reshaped = curr_activations.reshape(curr_activations.shape[0], -1)
            self.derivatives[i] = np.dot(curr_activations_reshaped, delta_reshaped) # curr_activations_reshaped @ delta_reshaped
            error = np.dot(delta, self.weights[i].T) #delta @ self.weights[i].T
            if verbose:
                print(f'Derivatives for W{i} is {self.derivatives[i]}')

        return error
    
    def gradient_descent(self, learning_rate):

        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            # print(f'OG W{i}: {weights}')
            weights -= learning_rate * derivatives
            # print(f'updated  W{i}: {weights}')


    def train(self, inputs, targets, epochs, learning_rate):
        for i in range(epochs):
            sum_error = 0
            for input, target in zip(inputs, targets):
                # forward pass
                output = self.forward_propagate(input)
                # calc error
                error = -(target - output) # derivative of mse
                # backward pass
                self.back_propagate(error)
                # apply gradient descent
                self.gradient_descent(learning_rate)

                sum_error += self._mse(target, output)
            
            # print error
            print(f'Error: {sum_error/len(inputs)} at epoch {i}')


    def _mse(self, target, output):
        return np.average((target-output)**2)

    
    def _sigmoid_derivative(self, x):
        return x * (1.0-x)

    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    

if __name__ == "__main__":
    
    inputs = np.array([[random()/2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in inputs])

    # create MLP
    mlp = MLP(2, [5], 1)

    mlp.train(inputs, targets, 50, 0.1)

    # making predictions
    input = np.array([0.3, 0.4])
    target = np.array([0.7])

    output = mlp.forward_propagate(input)
    print()
    print(f'Prediction of our network for {input[0]} + {input[1]} is {output[0]}')
