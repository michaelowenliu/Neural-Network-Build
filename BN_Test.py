from BatchNormalization_utils import *
import open_dataset
from sklearn.utils import shuffle
from nn_utils import compute_cost_softmax
import matplotlib.pyplot as plt
import pickle

X_train, Y_train, X_test, Y_test = open_dataset.load_mnist_vector()
layer_dims = [X_train.shape[0], 100,100,Y_train.shape[0]]
parameters,v,s, BN_parameters,BN_avg = initialize_parameters_BN(layer_dims)
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
        AL, caches, avg_mu, avg_var = DNN_forward_BN(X_train_mini, parameters, BN_parameters, train=True, BN_avg=BN_avg)

        #print(np.argmax(AL[:, 0]),np.argmax(Y_train_mini[:,0]))
        cost, dAL, accuracy = compute_cost_softmax(AL.copy(),Y_train_mini)
        gradients, BN_gradients = DNN_backward_BN(dAL, caches, parameters)
        parameters, v,s,BN_parameters = update_parameters_adam_BN(parameters, gradients, v,s,k+1, BN_parameters,
                                                                  BN_gradients,
                                                          learning_rate=learning_rate)
        costs.append(cost)
        accs.append(accuracy)
    print(k, cost, ' acc:', str(accuracy * 100)+'%')

Y_predict, _,_,_ = DNN_forward_BN(X_test, parameters,BN_parameters, train=False, BN_avg=BN_avg)
test_cost, _, test_accuracy = compute_cost_softmax(Y_predict.copy(), Y_test)
print(test_cost, ' test_acc:', test_accuracy * 100, '%')
with open('cost_acc.pickle'+str(minibatch_size), 'wb') as f:
    pickle.dump((costs, accs), f)

plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per tens)')
plt.title("Learning rate =" + str(0.01))
plt.show()