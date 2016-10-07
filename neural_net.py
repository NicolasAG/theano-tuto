import numpy as np

import theano
from theano import tensor as T

from mnist import load_data, plot_confusion_matrix, one_hot

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

# Convert each vector of labels to matrices of probabilities.
train_Y = one_hot(train_Y)
valid_Y = one_hot(valid_Y)
test_Y = one_hot(test_Y)


class NeuralNetwork(object):
    def __init__(self, x_train, y_train, hidden):
        """
        Constructor.
        :x_train: training data, an m x n numpy matrix.
        :y_train: labels for the training data, matrix of size m x t.
        :hidden:  number of hidden nodes in the hidden layer.
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

        assert type(hidden) is int
        self.h = hidden  # number of hidden nodes.

        # initial weights W_h is an n X h matrix where each row i is
        #  a vector of h weights for feature i, and each column j is
        #  a vector of n weights for hidden node j.
        # each n*h weights are uniformly sampled such that:
        #  -4 sqrt(6 / t+n) <= Wij < +4 sqrt(6 / n+h).
        self.w_h = theano.shared(
            name='w_h',
            value=np.asarray(
                np.random.uniform(
                    low=-4 * np.sqrt(6. / (self.n + self.h)),
                    high=4 * np.sqrt(6. / (self.n + self.h)),
                    size=(self.n, self.h)
                ),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # initial weights W_o is an h X t matrix where each row i is
        #  a vector of t weights for hidden node i, and each column j is
        #  a vector of h weights for target j.
        # each h*t weights are uniformly sampled such that:
        #  -4 sqrt(6 / t+n) <= Wij < +4 sqrt(6 / h+t).
        self.w_o = theano.shared(
            name='w_o',
            value=np.asarray(
                np.random.uniform(
                    low=-4 * np.sqrt(6. / (self.h + self.t)),
                    high=4 * np.sqrt(6. / (self.h + self.t)),
                    size=(self.h, self.t)
                ),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # b_h = vector of 0's for each hidden node h
        self.b_h = theano.shared(
            name='b_h',
            value=np.zeros(
                shape=(self.h,),  # (h,) and not (h,1) to be flexible! can be >1 if batch-size > 1
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # b_o = vector of 0's for each target t
        self.b_o = theano.shared(
            name='b_o',
            value=np.zeros(
                shape=(self.t,),  # (t,) and not (t,1) to be flexible! can be >1 if batch-size > 1
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        print "Initial model:"
        print "W_h ="; print self.w_h.get_value(); print self.w_h.get_value().shape
        print "b_h ="; print self.b_h.get_value(); print self.b_h.get_value().shape
        print "W_t ="; print self.w_o.get_value(); print self.w_o.get_value().shape
        print "b_t ="; print self.b_o.get_value(); print self.b_o.get_value().shape

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

        # Hidden layer: sigmoid
        probability_h = T.nnet.sigmoid(T.dot(x, self.w_h)+self.b_h)  # matrix of size m*h

        # Output layer: softmax
        probability_t = T.nnet.softmax(T.dot(probability_h, self.w_o)+self.b_o)  # matrix of probabilities of size m*t

        # compare matrix of probabilities to the true labels matrix Y of values 0 or 1
        cost = T.mean(T.nnet.categorical_crossentropy(probability_t, y))

        params = [self.w_h, self.b_h, self.w_o, self.b_o]  # parameters to optimize
        g_params = T.grad(cost=cost, wrt=params)  # gradient with respect to parameters

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
                    y: self.y_train[index:index + mini_batch_size]}
        )

        import time
        start_time = time.clock()
        for epoch in xrange(n_epochs):  # xrange ~ range but doesn't create a list! (faster and less memory used)
            print "Epoch:", epoch
            current_cost = 0
            # train from 0 to number of examples (m), by skipping batch.
            for row in xrange(0, self.m, mini_batch_size):
                current_cost = train(row)[0]
            print "cost:", current_cost
        end_time = time.clock()
        print "Average time per epoch = ", (end_time - start_time) / n_epochs

    def get_weights(self):
        """
        return the weights [W_h, b_h, W_o, b_o].
        """
        return [self.w_h, self.b_h, self.w_o, self.b_o]

    def get_prediction_function(self):
        """
        Return the theano function that predicts labels given data.
        :return: a theano function taking a matrix as input & returning an array of labels.
        """
        x = T.matrix('x')

        # Hidden layer: sigmoid
        probability_h = T.nnet.sigmoid(T.dot(x, self.w_h) + self.b_h)  # matrix of size m*h
        # Probability of being target t : softmax
        probability_t = T.nnet.softmax(T.dot(probability_h, self.w_o) + self.b_o)  # matrix of probabilities of size m*t

        # index of max probability for each row (example) = vector of size m
        prediction_t = T.argmax(probability_t, axis=1)

        return theano.function(
            inputs=[x],
            outputs=[prediction_t]
        )

##
# Train the Encoder-Decoder with the training set and output the first 100 learned features.
##
NN = NeuralNetwork(train_X, train_Y, 625)
NN.train(n_epochs=5, mini_batch_size=20, learning_rate=0.1)


[w_h, b_h, w_o, b_o] = NN.get_weights()
print "Learned model:"
print "W_h ="; print w_h.get_value(); print w_h.get_value().shape
print "b_h ="; print b_h.get_value(); print b_h.get_value().shape
print "W_o ="; print w_o.get_value(); print w_o.get_value().shape
print "b_o ="; print b_o.get_value(); print b_o.get_value().shape


prediction_function = NN.get_prediction_function()
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
