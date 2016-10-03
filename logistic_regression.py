import numpy as np

import theano
from theano import tensor as T

from mnist import load_data, plot_confusion_matrix

from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


path = "./mnist.pkl.gz"
data = load_data(path)  # data is of the form ((trainX, trainY), (validX, validY), (testX, testY))

train_X = data[0][0]
train_Y = data[0][1]
print "train_X shape = ", train_X.shape
print "train_Y shape = ", train_Y.shape
valid_X = data[1][0]
valid_Y = data[1][1]
print "valid_X shape = ", valid_X.shape
print "valid_Y shape = ", valid_Y.shape
test_X = data[2][0]
test_Y = data[2][1]
print "test_X shape = ", test_X.shape
print "test_Y shape = ", test_Y.shape


def one_hot(y, t=10):
    """
    Convert list of labels to a matrix of probability of size m*t
     with m = number of examples and t = number of possible targets.
    Create a matrix representation of y
    :param y: list of labels of size m
    :param t: number of different targets
    :return: a matrix of size m*t
    """
    if type(y) == list:
        y = np.array(y)
    y = y.flatten()
    o_h = np.zeros((len(y), t))  # create a matrix of 0's of shape m*10
    o_h[np.arange(len(y)), y] = 1  # put a 1 at each index of the target
    return o_h


# Convert each vector of labels to matrices of probabilities.
train_Y = one_hot(train_Y)
valid_Y = one_hot(valid_Y)
test_Y = one_hot(test_Y)


class LogisticRegression(object):
    def __init__(self, x_train, y_train):
        """
        Constructor.
        :x_train: training data, an m x n numpy matrix.
        :y_train: labels for the training data, matrix of size m x t.
        """
        assert type(x_train) is np.ndarray
        assert len(x_train.shape) == 2  # make sure X_train is a 2D numpy array.

        # Create a shared variable self.X
        #  config.floatX and borrow=True are just to make it run faster if GPU is used.
        self.x_train = theano.shared(
            name='x_train',
            value=np.asarray(x_train, dtype=theano.config.floatX),
            borrow=True
        )
        self.n = x_train.shape[1]  # number of features (col)
        self.m = x_train.shape[0]  # number of examples (row)

        assert type(y_train) is np.ndarray
        assert len(y_train.shape) == 2  # make sure Y_train is a 2D numpy array.
        assert y_train.shape[0] == self.m  # make sure there is one label for each example.
        self.y_train = theano.shared(
            name='y_train',
            value=np.asarray(y_train, dtype=theano.config.floatX),
            borrow=True
        )
        self.t = y_train.shape[1]  # number of distinct targets.

        # initial weights W is an n X t matrix where each row i is
        #  a vector of t weights for feature i, and each column j is
        #  a vector of n weights for target j.
        # each n*t weights are uniformly sampled such that:
        #  -4 sqrt(6 / t+n) <= Wij < +4 sqrt(6 / t+n).
        self.w = theano.shared(
            name='w',
            value=np.asarray(
                np.random.uniform(
                    low=-4 * np.sqrt(6. / (self.t + self.n)),
                    high=4 * np.sqrt(6. / (self.t + self.n)),
                    size=(self.n, self.t)
                ),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # b = vector of 0's for each target t
        self.b = theano.shared(
            name='b',
            value=np.zeros(
                shape=(self.t,),  # (t,) and not (t,1) to be flexible! can be >1 if batch-size > 1
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        print "Initial model:"
        print "W ="; print self.w.get_value(); print self.w.get_value().shape
        print "b ="; print self.b.get_value(); print self.b.get_value().shape

    def train(self, n_epochs=100, mini_batch_size=1, learning_rate=0.1):
        """
        Training function that minimize the cross-entropy loss using gradient descent.
        :n_epochs:          number of iterations to train.
        :mini_batch_size:   number of examples to take from X in one training pass.
        :learning_rate:     learning rate for gradient descent.
        """
        # theano function parameter representing the index of X
        #  where we should start training on.
        index = T.lscalar()
        # theano function variable that represents a batch of examples
        #  from X[index] to X[index+batch_size]
        x, y = T.matrices('x', 'y')

        # Probability of being target t = sigmoid = 1 / (1 + e^{-(Wx+b)})
        # probability_t = T.nnet.sigmoid(T.dot(x, self.w)+self.b)        # matrix of probabilities of size m*t
        probability_t = 1. / (1. + T.exp(-T.dot(x, self.w) + self.b))  # matrix of probabilities of size m*t

        # compare matrix of probabilities to the true labels matrix Y of values 0 or 1
        cost = T.mean(T.nnet.categorical_crossentropy(probability_t, y))

        params = [self.w, self.b]  # parameters to optimize
        g_params = T.grad(cost=cost, wrt=params)  # gradient with respect to W and b

        # update W and b like so: param = param - lr*gradient
        updates = []
        for param, g_param in zip(params, g_params):
            updates.append((param, param - learning_rate * g_param))

        # train function: (index -> cost) with x=X[i:i+mini_batch] & y=Y[i:i+mini_batch]
        train = theano.function(
            inputs=[index],
            outputs=[cost],
            updates=updates,
            givens={x: self.x_train[index:index + mini_batch_size],
                    y: self.y_train[index:index + mini_batch_size]
                    }
        )

        import time
        start_time = time.clock()
        for epoch in xrange(n_epochs):  # xrange ~ range but doesn't create a list! (faster and less memory used)
            print "Epoch:", epoch
            # train from 0 to number of examples (m), by skipping batch.
            for row in xrange(0, self.m, mini_batch_size):
                train(row)
        end_time = time.clock()
        print "Average time per epoch = ", (end_time - start_time) / n_epochs

    def get_weights(self):
        """
        return the weights [W, b].
        """
        return [self.w, self.b]

    def get_prediction_function(self):
        """
        Return the theano function that predicts labels given data.
        :return: a theano function taking a matrix as input & returning an array of labels.
        """
        x = T.matrix('x')

        # Probability of being target t = sigmoid = 1 / (1 + e^{-(Wx+b)})
        # probability_t = T.nnet.sigmoid(T.dot(x, self.w) + self.b)      # matrix of probabilities of size m*t
        probability_t = 1. / (1. + T.exp(-T.dot(x, self.w) + self.b))  # matrix of probabilities of size m*t

        # index of max probability for each row (example) = vector of size m
        prediction_t = T.argmax(probability_t, axis=1)

        return theano.function(
            inputs=[x],
            outputs=[prediction_t]
        )

##
# Train the Encoder-Decoder with the training set and output the first 100 learned features.
##
LR = LogisticRegression(train_X, train_Y)
LR.train(n_epochs=50, mini_batch_size=2, learning_rate=0.1)


[w, b] = LR.get_weights()
print "Learned model:"
print "W ="; print w.get_value(); print w.get_value().shape
print "b ="; print b.get_value(); print b.get_value().shape


prediction_function = LR.get_prediction_function()
predicted_labels = prediction_function(test_X)[0]  # array of predicted labels for each test examples
true_labels = np.argmax(test_Y, axis=1)  # array of true labels for each test_examples
print "true labels ="; print true_labels; print true_labels.shape
print "predicted labels ="; print predicted_labels; print predicted_labels.shape

print "similarity:", np.mean(true_labels == predicted_labels)


cm = confusion_matrix(true_labels, predicted_labels)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cm)
# Normalize the confusion matrix by row (i.e by the number of samples in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
plt.show()
