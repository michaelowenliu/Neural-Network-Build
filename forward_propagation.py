from layer_utils import *

def linear_activation_forward(A_pre, W, b, activation, dropout=False, dropout_prob=0.8):

    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_pre, W, b)
        A, activation_cache = sigmoid_forward(Z)
    elif activation == 'relu':
        Z, linear_cache = linear_forward(A_pre, W, b)
        if dropout:
            A, activation_cache = relu_forward_dropout(Z, prob=dropout_prob)
        else:
            A, activation_cache = relu_forward(Z)
    elif activation == 'softmax':
        Z, linear_cache = linear_forward(A_pre, W, b)
        A, activation_cache = softmax_forward(Z)
    else:
        print('ERROR!!!')
    # linear_cache: (A_pre, W, b); activation_cache: (A, Z)
    cache = (linear_cache, activation_cache)
    #print(cache[1][0])
    return A, cache

def DNN_forward(X, parameters, verbose = True, dropout=False):
    depth = len(parameters) // 2
    A_pre = X
    caches = []
    for i in range(1,depth):
        W, b = parameters['W'+str(i)], parameters['b'+str(i)]
        A, cache = linear_activation_forward(A_pre, W, b, 'relu',dropout=dropout)
        A_pre = A
        caches.append(cache)
    W, b = parameters['W' + str(depth)], parameters['b' + str(depth)]
    AL, cache = linear_activation_forward(A_pre, W, b, 'softmax')

    caches.append(cache)
    if verbose:
        pass
    #print(AL == caches[0][1][0])
    return AL, caches