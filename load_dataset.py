import numpy as np

def initialize_parameters(layer_dims):
    #n_layer is a list contains the number of neuron at each hidden layer
    depth = len(layer_dims)
    parameters = {}
    np.random.seed(10) # random seed
    for i in range(1,depth):
        parameters['W' + str(i)] = np.random.rand(layer_dims[i], layer_dims[i-1]) * 0.01
        parameters['b' + str(i)] = np.zeros((layer_dims[i], 1))
    
    return parameters