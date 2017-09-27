from layer_utils import *

def linear_activation_backward(dA, cache, activation, dropout=False):
    linear_cache, activation_cache = cache
    if activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_pre, dW, db = linear_backward(dZ, linear_cache)

    elif activation == 'relu':
        if dropout:
            dZ = relu_backward_dropout(dA, activation_cache)
        else:
            dZ = relu_backward(dA, activation_cache)
        dA_pre, dW, db = linear_backward(dZ, linear_cache)
    elif activation == 'softmax':
        dZ = softmax_backward(dA, activation_cache)
        dA_pre, dW, db = linear_backward(dZ, linear_cache)
    else:
        print('ERROR!!!')

    return dA_pre, dW, db

def DNN_backward(dAL, caches, dropout=False):

    depth = len(caches)
    cache = caches[depth-1]
    gradients = {}
    dA_pre, dW, db = linear_activation_backward(dAL, cache, 'softmax')
    gradients['dA'+str(depth)] = dA_pre
    gradients['dW' + str(depth)] = dW
    gradients['db' + str(depth)] = db

    for i in reversed(range(depth-1)):
        dA_pre, dW, db = linear_activation_backward(gradients['dA'+str(i+2)], caches[i], 'relu', dropout=True)
        gradients['dA' + str(i+1)] = dA_pre
        gradients['dW' + str(i+1)] = dW
        gradients['db' + str(i+1)] = db
        #print(db.shape)
    #print(db)
    return gradients

def DNN_backward_regularization(dAL,caches, lambd, parameters, dropout=False):
    m = dAL.shape[1]
    depth = len(caches)
    cache = caches[depth-1]
    gradients = {}
    dA_pre, dW, db = linear_activation_backward(dAL, cache, 'softmax')
    gradients['dA'+str(depth)] = dA_pre
    gradients['dW' + str(depth)] = dW + lambd/m * parameters['W' + str(depth)]
    gradients['db' + str(depth)] = db

    for i in reversed(range(depth-1)):
        dA_pre, dW, db = linear_activation_backward(gradients['dA'+str(i+2)], caches[i], 'relu', dropout=dropout)
        gradients['dA' + str(i+1)] = dA_pre
        gradients['dW' + str(i+1)] = dW + lambd/m * parameters['W' + str(i+1)]
        #print(lambd/m * parameters['W' + str(i+1)])
        gradients['db' + str(i+1)] = db
        #print(db.shape)
    #print(db)
    return gradients