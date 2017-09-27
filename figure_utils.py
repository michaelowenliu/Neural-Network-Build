import matplotlib.pyplot as plt
import numpy as np

def draw_wrong_img(Y_predict, Y_truth, X):
    max__predict = np.argmax(Y_predict, axis=0)
    max_truth = np.argmax(Y_truth, axis=0)
    bool_ix = max__predict == max_truth
    while True:
        ix = np.random.randint(len(bool_ix))
        if ~bool_ix[ix]:
            plt.gray()
            plt.title('truth is '+str(max_truth[ix]) +' is mis-classified to '+str(max__predict[ix]))
            plt.imshow(X[:,ix].reshape(8,8))
            plt.show()
            break