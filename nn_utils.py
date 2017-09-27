from sklearn.datasets import *
import numpy as np
import random
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

def load_dataset():
#convert to X.shape = (d,m), Y.shape = (1,m), where d is dimension, m is sample's size
    X, Y = load_digits(return_X_y=True)
    #X = preprocessing.scale(X)
    X = X.T
    class_num = len(set(Y))
    Y = one_hot_key(Y,class_num)
    Y = Y.T
    # if en =='one_hot_key':
    #     Y = one_hot_key(Y,2)
    #     Y = Y.T
    #
    # else:
    #     Y = Y.reshape(1, X.shape[1])

    return split_train_test(X, Y) #default train_size = 0.7

def one_hot_key(Y,class_num):
    Z = np.eye(class_num, dtype=int)
    return Z[Y]

def split_train_test(X, Y, train_size=0.7):
    size = X.shape[1]
    test_size = size - int(train_size * size)
    total_index = [x for x in range(size)]
    random.seed(7)
    test_index = random.sample(range(size), test_size)
    train_index = [item for item in total_index if item not in test_index]
    X_train, Y_train = X[:, train_index], Y[:, train_index]
    X_test, Y_test = X[:, test_index], Y[:, test_index]

    return X_train, Y_train, X_test, Y_test

def compute_cost(AL,Y, verbose = True):
    m = Y.shape[1]
    cost = -1/m * (np.sum(Y * np.log(AL) + (1-Y) * np.log(1-AL)))
    dAL = np.divide(AL-Y, AL*(1-AL))
    if verbose:
        accuracy = evaluate_accuracy(Y, AL)
        #print('cost:', cost ,' ', 'acc:', accuracy)
    return cost, dAL, accuracy

def compute_cost_softmax(AL, Y):
    m = Y.shape[1]
    cost = -1/m * np.sum(Y * np.log(np.maximum(1e-15,AL)))
    dAL = Y * (-1 / np.maximum(1e-15,AL))
    #print('dAL', dAL[:,1])
    accuracy = evaluate_accuracy(Y, AL, loss_func = 'softmax')
    return cost, dAL, accuracy

def compute_cost_softmax_regularization(AL, Y, parameters, lambd, verbose = True):
    m = Y.shape[1]
    reg_cost = 0

    for i in range(0,len(parameters)//2):
        reg_cost += np.sum(np.square(parameters['W'+str(i+1)]))
    reg_cost = 1/2 *lambd * reg_cost / m
    cost = -1/m * np.sum(Y * np.log(np.maximum(1e-15,AL))) + reg_cost
    dAL = Y * (-1 / np.maximum(1e-15,AL))
    #dAL = Y # a tricky way to avoid AL = 0, the correct gradient should be above one
    #print('dAL', dAL[:,1])
    if verbose:
        accuracy = evaluate_accuracy(Y, AL, loss_func = 'softmax')
    return cost, dAL, accuracy




def evaluate_accuracy(Y_truth, Y_predict, loss_func ='sigmoid'):
    if loss_func == 'sigmoid':
        # Y_predict is a probability, i.e. A (0,1)
        Y_predict[np.where(Y_predict >= 0.5)] = 1
        Y_predict[np.where(Y_predict < 0.5)] = 0
        accuracy = np.mean((Y_predict == Y_truth).astype(int))
    elif loss_func == 'softmax':
        max__predict = np.argmax(Y_predict,axis=0)
        max_truth = np.argmax(Y_truth, axis=0)
        accuracy = np.mean((max__predict == max_truth).astype(int))
    return accuracy
