import numpy as np

def linear_forward(A_pre, W, b):
    Z = np.dot(W, A_pre) + b
    cache = (A_pre, W, b)
    return Z, cache
def sigmoid_forward(Z):
    A = 1 / (1 + np.exp(-Z))
    #print(A)
    cache = (A, Z)
    return A, cache
def relu_forward(Z):
    A = np.maximum(0, Z)

    cache = (A, Z)
    return A, cache
def relu_forward_dropout(Z,prob):
    A, relu_cache = relu_forward(Z)
    A, dropout_cache = dropout_forward(A, prob)
    cache = (relu_cache, dropout_cache)
    return A, cache
def softmax_forward(Z):
    #print(np.sum(np.exp(Z),axis=0))
    A = np.exp(Z) / np.sum(np.exp(Z),axis=0)

    cache = (A, Z)
    return A, cache

def dropout_forward(A, prob):
    D_mask = np.random.rand(A.shape[0],A.shape[1]) < prob
    A = A * D_mask
    A = A / prob
    cache = (D_mask, prob)
    return A,cache

def BN_forward(Z,gamma,beta,avg_mu, avg_var, train=True):

    if train == True:
        mu = np.mean(Z,axis=1,keepdims=True)
        var = np.var(Z,axis=1, keepdims=True)

        Z_hat = (Z - mu) / np.sqrt(var + 1e-15)
        Z_shift = gamma * Z_hat + beta
        cache = (Z, Z_hat, mu, var, gamma)

        avg_mu = 0.9 * avg_mu + 0.1 * mu
        avg_var = 0.9 * avg_var + 0.1 * var
    else:
        Z_shift = gamma * (Z - avg_mu) / np.sqrt(avg_var + 1e-15) + beta
        cache = None
    return Z_shift, cache, avg_mu, avg_var

def BN_backward(dZ_shift, cache):
    Z, Z_hat, mu, var, gamma = cache
    dZ_hat = dZ_shift * gamma
    dvar = np.sum(dZ_hat * (Z - mu), axis=1,keepdims=True) * (-0.5 * np.power(var + 1e-15,-3/2))
    dmu = np.sum(dZ_hat, axis=1,keepdims=True) * (-1 / np.sqrt(var+1e-15)) + dvar * np.mean(-2 * (Z - mu), axis=1,keepdims=True)
    #print(np.sum(dZ_hat * (Z - mu), axis=1).shape)
    #print((Z-mu).shape)
    dZ = dZ_hat * (1 / np.sqrt(var+1e-15)) + dvar*2*(Z-mu) / Z.shape[1] + dmu / Z.shape[1]
    dgamma = np.sum(dZ_shift * Z_hat, axis=1,keepdims=True)
    dbeta = np.sum(dZ_shift, axis=1,keepdims=True)

    return dZ, dgamma, dbeta

def softmax_backward(dA,activation_cache):
    A, Z = activation_cache
    dZ = A + dA * A
    #dZ = A - dA
    # a tricky way to avoid AL = 0, the correct gradient should be above one, and dAL = Y * (-1 / (AL))
    #print('A', A[:, 1])
    #print('dz', dZ[:,1])
    return dZ
def sigmoid_backward(dA,activation_cache):
    A, Z = activation_cache
    dZ = dA * (A*(1-A))

    #print(1/(1+np.exp(-Z)))
    return dZ
def dropout_backward(dA, dropout_cache):
    D_mask, prob = dropout_cache
    dA = dA * D_mask
    dA = dA / prob
    return dA
def relu_backward_dropout(dA, relu_dropout_cache):
    (relu_cache, dropout_cache) = relu_dropout_cache
    dA = dropout_backward(dA, dropout_cache)
    dZ = relu_backward(dA, relu_cache)
    return dZ
def relu_backward(dA,activation_cache):
    A, Z = activation_cache
    dZ = dA
    dZ[Z < 0] = 0
    return dZ
def linear_backward(dZ, linear_cache):
    A_pre, W, b = linear_cache
    m = A_pre.shape[1]
    dW = np.dot(dZ, A_pre.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_pre = np.dot(W.T, dZ)
    return dA_pre, dW, db