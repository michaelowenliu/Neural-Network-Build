import numpy as np

def initialize_parameters(layer_dims, optimization='sgd'):
    # n_layer is a list contains the number of neuron at each hidden layer
    depth = len(layer_dims)
    parameters = {}
    np.random.seed(10)  # random seed
    for i in range(1, depth):
        parameters['W' + str(i)] = np.random.rand(layer_dims[i], layer_dims[i - 1]) * 0.01
        parameters['b' + str(i)] = np.zeros((layer_dims[i], 1))
    if optimization == 'momentum':
        velocity = initialize_velocity(parameters)
        return parameters,velocity
    elif optimization == 'adam':
        v,s = initialize_adam(parameters)
        return parameters, v,s

    return parameters

def initialize_velocity(parameters):

    L = len(parameters) // 2
    velocity = {}
    for i in range(L):
        velocity['dW' + str( i +1)] = np.zeros(parameters['W' + str( i +1)].shape)
        velocity['db' + str( i +1)] = np.zeros(parameters['b' + str( i +1)].shape)
    return velocity

def initialize_adam(parameters):

    L = len(parameters) // 2
    v = {}
    s = {}
    for i in range(L):
        v['dW' + str( i +1)] = np.zeros(parameters['W' + str(i +1)].shape)
        v['db' + str( i +1)] = np.zeros(parameters['b' + str(i +1)].shape)
        s['dW' + str(i + 1)] = np.zeros(parameters['W' + str(i + 1)].shape)
        s['db' + str(i + 1)] = np.zeros(parameters['b' + str(i + 1)].shape)
    return v,s