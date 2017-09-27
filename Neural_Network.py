import numpy as np
from nn_utils import *
from forward_propagation import DNN_forward
from backward_propagation import *
from figure_utils import *
from optimization_method import *
import open_dataset
from initialization_method import *
from sklearn.utils import shuffle
import pickle
# neural network structure (hidden layers, activation, cost function)
# initialize parameters W,b
# forward propagation -> calculate cost
# backward propagation -> calculate gradients dW, db
# update parameters W,b using dW, db with learning rate

#without Batch Normalization; BN is ruing BN_Test.py#
def DNN():
    # X.shape = (d,m), Y.shape = (c,m), where d is dimension, m is sample's size
    X_train, Y_train, X_test, Y_test = open_dataset.load_mnist_vector()
    layer_dims = [X_train.shape[0], 100,100,Y_train.shape[0]]
    parameters,v,s = initialize_parameters(layer_dims, optimization='adam')
    lambd = 0
    learning_rate = 0.01
    dropout = True
    minibatch_size = 250 #X_train.shape[1]
    costs = []
    accs = []
    epoch = 100
    for k in range(epoch):

        X_train, Y_train = shuffle(X_train.T, Y_train.T)
        X_train = X_train.T
        Y_train = Y_train.T

        for i in range(0, X_train.shape[1], minibatch_size):
            X_train_mini = X_train[:, i:i + minibatch_size]
            Y_train_mini = Y_train[:, i:i + minibatch_size]
            AL, caches = DNN_forward(X_train_mini, parameters, dropout=dropout)

            #print(np.argmax(AL[:, 0]),np.argmax(Y_train_mini[:,0]))
            cost, dAL, accuracy = compute_cost_softmax_regularization(AL.copy(),
                                                                      Y_train_mini, parameters, lambd=lambd)
            gradients = DNN_backward_regularization(dAL, caches, lambd, parameters, dropout=dropout)
            parameters, v,s = update_parameters_adam(parameters, gradients, v,s,t=k+1,
                                                              learning_rate=learning_rate)
            accs.append(accuracy)
            costs.append(cost)
        print(k, cost, ' acc:', str(accuracy * 100)+'%')
    with open('cost_acc_noBN.pickle'+str(minibatch_size), 'wb') as f:
        pickle.dump((costs, accs), f)

    Y_predict, _ = DNN_forward(X_test, parameters)
    test_cost, _, test_accuracy = compute_cost_softmax(Y_predict.copy(), Y_test)
    print(test_cost, ' test_acc:', test_accuracy * 100, '%')

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(0.01))
    plt.show()
    # draw_wrong_img(Y_predict, Y_test, X_test)
# def DNN_model(X_train,Y_train,parameters,v,t,dropout=False, lambd=0,learning_rate=0.01):
#
#     #print(np.argmax(AL[:,0]))


DNN()
