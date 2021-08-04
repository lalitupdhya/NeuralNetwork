from ForwardPropagation import NeuralNetwork
import numpy as np

def sigmoidPrime(z): # derivative of activation with respect to sigmoid
    return np.exp(-z)/((1+np.exp(-z))**2)

def costFunctionPrime(X, y):
    nn = NeuralNetwork(2,3,1)
    yHat, z2, a2, z3 = nn.ForwardProp(X) # importing forwardProp
    W1, W2 = nn.weights()

    delta3 = np.multiply(-(y- yHat), sigmoidPrime(z3))
    dJdW2 = np.dot(a2.T, delta3)

    delta2 = np.dot(W2.T, delta3) * sigmoidPrime(z2)
    dJdW1 = np.dot(X.T, delta2)

    return dJdW1, dJdW2

X = np.array([[1,2],[3,4],[5,6]])
y = np.array([1,2,3])

q,w = costFunctionPrime(X, y)
print(f'dJdW1:\n{q}\n\ndJdW2:\n{w}')
