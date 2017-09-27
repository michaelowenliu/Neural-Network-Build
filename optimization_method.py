import numpy as np

def update_parameters(parameters, gradients, learning_rate):
    depth = len(parameters) // 2
    for i in range(0,depth):
        parameters['W'+str(i+1)] -= learning_rate * gradients['dW'+str(i+1)]
        parameters['b' + str(i+1)] -= learning_rate * gradients['db' + str(i+1)]
    return parameters

def update_parameters_regularization(parameters, gradients, learning_rate, lamda = 0.001):
    depth = len(parameters) // 2
    for i in range(0,depth):
        parameters['W'+str(i+1)] -= (learning_rate * gradients['dW'+str(i+1)] + lamda*parameters['W'+str(i+1)])
        parameters['b' + str(i+1)] -= learning_rate * gradients['db' + str(i+1)]
    return parameters

def update_parameters_momentum(parameters, gradients, velocity, beta=0.9, learning_rate=0.01):
    depth = len(parameters) // 2
    for i in range(0,depth):
        velocity['dW' + str(i+1)] = beta * velocity['dW' + str(i+1)] + (1-beta) * gradients['dW'+str(i+1)]
        velocity['db' + str(i + 1)] = beta * velocity['db' + str(i + 1)] + (1 - beta) * gradients['db' + str(i+1)]

        parameters['W'+str(i+1)] -= learning_rate * velocity['dW' + str(i+1)]
        parameters['b' + str(i+1)] -= learning_rate * velocity['db' + str(i + 1)]
    return parameters, velocity

def update_parameters_adam(parameters, gradients, v,s, t,beta1=0.9, beta2=0.99, learning_rate=0.01, epsilon=1e-8):
    depth = len(parameters) // 2
    for i in range(0, depth):
        v['dW' + str(i + 1)] = beta1 * v['dW' + str(i + 1)] + (1 - beta1) * gradients['dW' + str(i + 1)]
        v_corrected_w = v['dW' + str(i + 1)] / (1-np.power(beta1,t))
        s['dW' + str(i + 1)] = beta2 * s['dW' + str(i + 1)] + (1 - beta2) * np.square(gradients['dW' + str(i + 1)])
        s_corrected_w = s['dW' + str(i + 1)] / (1-np.power(beta2,t))
        parameters['W' + str(i + 1)] -= learning_rate * (v_corrected_w / (np.sqrt(s_corrected_w) + epsilon))

        v['db' + str(i + 1)] = beta1 * v['db' + str(i + 1)] + (1 - beta1) * gradients['db' + str(i + 1)]
        v_corrected_b = v['db' + str(i + 1)] / (1 - np.power(beta1, t))
        s['db' + str(i + 1)] = beta2 * s['db' + str(i + 1)] + (1 - beta2) * np.square(gradients['db' + str(i + 1)])
        s_corrected_b = s['db' + str(i + 1)] / (1 - np.power(beta2, t))
        parameters['b' + str(i + 1)] -= learning_rate * (v_corrected_b / (np.sqrt(s_corrected_b) + epsilon))
        #print(parameters['W' + str(i + 1)])
    return parameters, v, s

