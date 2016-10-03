import numpy as np
import theano
from theano import tensor as T

from autoencoder import AutoEncoder
from mnist import load_data

DEBUG = False

path = "../mnist.pkl.gz"
data = load_data(path)

train_set = data[0][0]  # [0] returns a tuple, so need an extra[0] to get the ndarray
valid_set = data[1][0]  # [1] returns a tuple, so need an extra[0] to get the ndarray
test_set = data[2][0]  # [2] returns a tuple, so need an extra[0] to get the ndarray

print "data loaded."
print "train_set shape = ", train_set.shape
print "valid_set shape = ", valid_set.shape
print "test_set shape = ", test_set.shape


def plot_first_k_numbers(X, k):
    """
    plotting function to visualize the learned features.
    @param X - the dataset with features as rows.
        Each row is a 28x28 feature image.
    @param k - the number of features to show.
    """
    from matplotlib import pyplot
    m = X.shape[0]  # number of features
    k = min(m, k)  # k = real number of features that we will show
    j = int(round(k / 10.))

    fig, ax = pyplot.subplots(j, 10)

    for i in range(k):
        w = X[i]
        w = w.reshape(28, 28)  # reshape the features to a 28x28 image

        ax[i/10, i % 10].imshow(
            w,
            cmap=pyplot.cm.gist_yarg,
            interpolation='nearest',
            aspect='equal'
        )
        ax[i/10, i % 10].axis('off')

    pyplot.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off'  # labels along the bottom edge are off
    )
    pyplot.tick_params(
        axis='y_predicted',  # changes apply to the y_predicted-axis
        which='both',  # both major and minor ticks are affected
        left='off',  # ticks along the left edge are off
        right='off',  # ticks along the right edge are off
        labelleft='off'  # labels along the left edge are off
    )

    pyplot.show()


X = train_set
activation_function = T.nnet.sigmoid
output_function = T.nnet.sigmoid
hidden_size = 500

if DEBUG:
    theano.config.optimizer = "fast_compile"
    theano.config.exception_verbosity = "high"

##
# Train the Encoder-Decoder with the training set and output the first 100 learned features.
##
A = AutoEncoder(X, hidden_size, activation_function, output_function)
A.train(n_epochs=20, mini_batch_size=20, learning_rate=0.1)

[W, b1, b2] = A.get_weights()
print "w =", W.shape
print "b1 =", b1.shape
print "b2 =", b2.shape

plot_first_k_numbers(np.transpose(W), 100)

##
# Encode-Decode the validation set and output the first 100 results.
##
print "valid_set.shape = ", valid_set.shape
transformation = A.get_output(valid_set)  # get theano function that computes the output
Y = transformation(valid_set)[0]  # output is of the form [data], so need to take [0]
print "Y = ", Y
print "Y.shape = ", Y.shape
plot_first_k_numbers(Y, 100)
