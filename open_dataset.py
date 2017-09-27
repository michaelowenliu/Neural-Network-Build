import idx2numpy
import numpy as np
def load_mnist_vector():
    root_path = './mnist data/'
    train_image =  idx2numpy.convert_from_file(root_path+'train-images.idx3-ubyte')
    train_label = idx2numpy.convert_from_file(root_path+'train-labels.idx1-ubyte')
    test_image = idx2numpy.convert_from_file(root_path+'t10k-images.idx3-ubyte')
    test_label = idx2numpy.convert_from_file(root_path+'t10k-labels.idx1-ubyte')

    X_train = train_image.reshape(train_image.shape[0],-1).T / 255
    X_test = test_image.reshape(test_image.shape[0],-1).T / 255

    class_num = len(set(train_label))
    Y_train = one_hot_key(train_label, class_num)
    Y_train = Y_train.T
    # convert to X.shape = (d,m), Y.shape = (c,m), where d is dimension, m is sample's size
    Y_test = one_hot_key(test_label, class_num)
    Y_test = Y_test.T

    return X_train, Y_train, X_test, Y_test


def one_hot_key(Y, class_num):
    Z = np.eye(class_num, dtype=int)
    return Z[Y]

# X_train, Y_train, X_test, Y_test = load_mnist_vector()
# print(X_train.shape)
# print(Y_train.shape)
# print(X_test.shape)
# print(Y_test.shape)