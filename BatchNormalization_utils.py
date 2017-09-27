from layer_utils import *

def initialize_parameters_BN(layer_dims):
    # n_layer is a list contains the number of neuron at each hidden layer
    depth = len(layer_dims)
    parameters = {}
    BN_parameters = {}
    BN_avg = {}
    v = {}
    s = {}
    #np.random.seed(10)  # random seed
    for i in range(1, depth):
        parameters['W' + str(i)] = np.random.rand(layer_dims[i], layer_dims[i - 1]) * 0.01
        parameters['b' + str(i)] = np.zeros((layer_dims[i], 1))
        v['dW' + str(i)] = np.zeros(parameters['W' + str(i)].shape)
        v['db' + str(i)] = np.zeros(parameters['b' + str(i)].shape)
        s['dW' + str(i)] = np.zeros(parameters['W' + str(i)].shape)
        s['db' + str(i)] = np.zeros(parameters['b' + str(i)].shape)
        BN_parameters['gamma' + str(i)] = np.ones((layer_dims[i],1))
        BN_parameters['beta' + str(i)] = np.zeros((layer_dims[i],1))
        BN_avg['avg_mu' +str(i)] = np.zeros(1)
        BN_avg['avg_var' + str(i)] = np.zeros(1)

    return parameters, v,s, BN_parameters,BN_avg

def linear_activation_forward_BN(A_pre, W, b, gamma=None, beta=None, activation='relu', dropout=False, dropout_prob=0.8):
    BN_cache = None
    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_pre, W, b)
        A, activation_cache = sigmoid_forward(Z)
    elif activation == 'relu':
        Z, linear_cache = linear_forward(A_pre, W, b)
        Z_shift, BN_cache, avg_mu,avg_mu = BN_forward(Z,gamma,beta)
        if dropout:
            A, activation_cache = relu_forward_dropout(Z_shift, prob=dropout_prob)
        else:
            A, activation_cache = relu_forward(Z_shift)
    elif activation == 'softmax':
        Z, linear_cache = linear_forward(A_pre, W, b)
        A, activation_cache = softmax_forward(Z)
    else:
        print('ERROR!!!')
    # linear_cache: (A_pre, W, b); activation_cache: (A, Z)
    cache = (linear_cache, activation_cache,BN_cache)
    #print(cache[1][0])
    return A, cache

def DNN_forward_BN(X, parameters, BN_parameters, train, BN_avg, dropout=False):
    depth = len(parameters) // 2
    A_pre = X
    caches = []
    for i in range(1,depth):
        #print(i)
        W, b = parameters['W'+str(i)], parameters['b'+str(i)]
        gamma, beta, = BN_parameters['gamma' + str(i)], BN_parameters['beta' + str(i)]
        #A, cache = linear_activation_forward_BN(A_pre, W, b, gamma, beta,'relu',dropout=dropout)
        avg_mu, avg_var = BN_avg['avg_mu' + str(i)], BN_avg['avg_var' +str(i)]
        Z, linear_cache = linear_forward(A_pre, W, b)
        Z_shift, BN_cache, avg_mu, avg_var = BN_forward(Z, gamma, beta, avg_mu, avg_var,train=train)
        BN_avg['avg_mu' + str(i)], BN_avg['avg_var' + str(i)] = avg_mu, avg_var
        A, activation_cache = relu_forward(Z_shift)
        cache = (linear_cache, activation_cache, BN_cache)
        A_pre = A
        caches.append(cache)
    W, b = parameters['W' + str(depth)], parameters['b' + str(depth)]
    #AL, cache = linear_activation_forward_BN(A_pre, W, b, activation='softmax')
    Z, linear_cache = linear_forward(A_pre, W, b)
    AL, activation_cache = softmax_forward(Z)
    BN_cache = None
    cache = (linear_cache, activation_cache, BN_cache)
    caches.append(cache)

    return AL, caches, avg_mu, avg_var

def linear_activation_backward_BN(dA, cache, activation, dropout=False):
    linear_cache, activation_cache, BN_cache = cache
    dgamma = None
    dbeta = None
    if activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_pre, dW, db = linear_backward(dZ, linear_cache)

    elif activation == 'relu':
        if dropout:
            dZ_shift = relu_backward_dropout(dA, activation_cache)
            dZ = BN_backward(dZ_shift, BN_cache)
        else:
            dZ_shift = relu_backward(dA, activation_cache)
            dZ, dgamma, dbeta = BN_backward(dZ_shift, BN_cache)
        dA_pre, dW, db = linear_backward(dZ, linear_cache)
    elif activation == 'softmax':
        dZ = softmax_backward(dA, activation_cache)
        dA_pre, dW, db = linear_backward(dZ, linear_cache)
    else:
        print('ERROR!!!')

    return dA_pre, dW, db, dgamma, dbeta

def DNN_backward_BN(dAL, caches, dropout=False):

    depth = len(caches)
    cache = caches[depth-1]
    gradients = {}
    BN_gradients = {}
    dA_pre, dW, db, dgamma, dbeta = linear_activation_backward_BN(dAL, cache, 'softmax')
    gradients['dA'+str(depth)] = dA_pre
    gradients['dW' + str(depth)] = dW
    gradients['db' + str(depth)] = db
    BN_gradients['dgamma' + str(depth)] = 0
    BN_gradients['dbeta' + str(depth)] = 0
    for i in reversed(range(depth-1)):
        dA_pre, dW, db, dgamma, dbeta = linear_activation_backward_BN(gradients['dA'+str(i+2)], caches[i], 'relu')
        gradients['dA' + str(i+1)] = dA_pre
        gradients['dW' + str(i+1)] = dW
        gradients['db' + str(i+1)] = db
        BN_gradients['dgamma' + str(i + 1)] = dgamma
        BN_gradients['dbeta' + str(i + 1)] = dbeta
        #print(db.shape)
    #print(db)
    return gradients, BN_gradients

def update_parameters_adam_BN(parameters, gradients, v,s,t, BN_parameters, BN_gradients,
                              beta1=0.9, beta2=0.99, learning_rate=0.01, epsilon=1e-8):
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
        BN_parameters['gamma' + str(i + 1)] -= learning_rate * BN_gradients['dgamma' + str(i + 1)]
        BN_parameters['beta' + str(i + 1)] -= learning_rate * BN_gradients['dbeta' + str(i + 1)]

    return parameters, v, s, BN_parameters