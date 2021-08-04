#single hidden layer feed forward network

import numpy as np

class NeuralNetwork:

    def __init__(self, input_units, hidden_units, output_units):
        
        #hyperparameters
        self.input_units = input_units
        self.hidden_units = hidden_units
        self.output_units = output_units

        #weights(parameters)
        self.W1 = np.random.randn(self.input_units, self.hidden_units)
        self.W2 = np.random.randn(self.hidden_units, self.output_units)
    
    def ForwardProp(self, X):
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)

        return yHat
    
    def sigmoid(self, z):
        return 1/(1+np.exp(z))


if __name__=='__main__':
    
    X = np.array([[2,3],[4,5],[6,7]])
    nn = NeuralNetwork(2, 3, 1)
    yhat = nn.ForwardProp(X)

    print(yhat)


