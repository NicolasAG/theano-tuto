import numpy as np
import theano
from theano import tensor as T


class AutoEncoder(object):
    """
    AutoEncoder implementation from the following tutorial:
    https://triangleinequality.wordpress.com/2014/08/12/theano-autoencoders-and-mnist/

    This autoencoder has:
        an input layer of n nodes,
            x -> f1(W1 x + b1) = h
        a hidden layer of h nodes,
            h -> f2(W2 x + b2) = x'
        an output layer of n nodes.

    The following is an autoencoder with TIED WEIGHTS (W2 = W1^T).
    This gives the model less free parameters and less memory usage.
    """

    def __init__(self, X, hidden_size, activation_function, output_function):
        """
        Constructor.
        @param X is the data, an m x n numpy matrix.
        @param hidden_size is the number of neurons in the hidden layer.
        @param activation_function = f1.
        @param output_function = f2.
        """
        assert type(X) is np.ndarray
        assert len(X.shape) == 2  # make sure X is a 2D numpy array.

        # Create a shared variable self.X
        #  config.floatX and borrow=True are just to make it run faster if GPU is used.
        self.X = theano.shared(
            name='X',
            value=np.asarray(X, dtype=theano.config.floatX),
            borrow=True
        )
        self.n = X.shape[1]  # number of features (col)
        self.m = X.shape[0]  # number of examples (row)

        assert type(hidden_size) is int
        assert hidden_size > 0  # make sure we have a positive amount of nodes in the hidden layer.
        self.hidden_size = hidden_size

        # initial weights W1 is a n X h matrix where each row i is
        #  a vector of h weights for feature i, and each column j is
        #  a vector of n weights for hidden node j.
        # each n x h weights are uniformly sampled such that:
        #  -4 sqrt(6 / h+n) <= Wij < +4 sqrt(6 / h+n).
        initial_w = np.asarray(
            np.random.uniform(
                low=-4 * np.sqrt(6. / (self.hidden_size + self.n)),
                high=4 * np.sqrt(6. / (self.hidden_size + self.n)),
                size=(self.n, self.hidden_size)
            ),
            dtype=theano.config.floatX
        )
        # here we assume that W2 = W1^T. This gives the model less free parameters
        #  and less memory usage. This is refered as an autoencoder with TIED WEIGHTS.
        self.W = theano.shared(
            name='W',
            value=initial_w,
            borrow=True
        )

        # b1 = vector of 0's for each hidden nodes
        self.b1 = theano.shared(
            name='b1',
            value=np.zeros(
                shape=(self.hidden_size,),  # (h,) and not (h,1) to be flexible! can be >1 if batch-size > 1
                dtype=theano.config.floatX),
            borrow=True
        )
        # b2 = vector of 0's for each output nodes
        self.b2 = theano.shared(
            name='b2',
            value=np.zeros(
                shape=(self.n,),  # (n,) and not (n,1) to be flexible! can be >1 if batch-size > 1
                dtype=theano.config.floatX),
            borrow=True
        )

        self.activation_function = activation_function
        self.output_function = output_function

    def train(self, n_epochs=100, mini_batch_size=1, learning_rate=0.1):
        """
        Training function that minimize the cross-entropy loss using gradient descent.
        @param n_epochs is the number of iterations to train.
        @param mini_batch_size is the number of examples to take from X in one training pass.
        @param learning_rate is the learning rate for gradient descent.
        """
        # theano function parameter representing the index of X
        #  where we should start training on.
        index = T.lscalar()
        # theano function variable that represents a batch of examples
        #  from X[index] to X[index+batch_size]
        x = T.matrix('x')

        # parameters to optimize
        params = [self.W, self.b1, self.b2]

        # hidden = h = f1(W1 x + b1)
        hidden = self.activation_function(T.dot(x, self.W) + self.b1)
        # output = x' = f2(W2 h + b2)
        output = self.output_function(T.dot(hidden, T.transpose(self.W)) + self.b2)

        # use cross-entropy loss
        loss = -T.sum(x * T.log(output) + (1 - x) * T.log(1 - output), axis=1)
        cost = loss.mean()

        # return gradient with respect to W, b1, b2
        g_params = T.grad(cost, params)

        # update W,b1,b2 like so: param = param - lr*gradient
        updates = []
        for param, g_param in zip(params, g_params):
            updates.append((param, param - learning_rate * g_param))

        # train function: (index -> cost) with x = X[i:i+mini_batch]
        train = theano.function(
            inputs=[index],
            outputs=[cost],
            updates=updates,
            givens={x: self.X[index:index + mini_batch_size]}
        )

        import time
        start_time = time.clock()
        for epoch in xrange(n_epochs):  # xrange ~ range but ddoesn't create a list! (faster and less memory used)
            print "Epoch: ", epoch
            # train from 0 to number of examples (m), by skipping batch.
            for row in xrange(0, self.m, mini_batch_size):
                train(row)
        end_time = time.clock()
        print "Average time per epoch = ", (end_time - start_time) / n_epochs

    def get_hidden(self, data):
        """
        Return hidden = h = f1(W x + b1).
        @param data is the data in variable 'x'.
        """
        # variable representing data
        x = T.matrix('x')
        # hidden = h = f1(W x + b1)
        hidden = self.activation_function(T.dot(x, self.W) + self.b1)

        transformed_data = theano.function(
            inputs=[x],
            outputs=[hidden]
        )
        return transformed_data

    def get_output(self, data):
        """
        Return output = x' = f2(W2 h + b2) with h = f1(W1 x + b1)
        @param data is the data in variable 'x'.
        """
        # variable representing data
        x = T.matrix('x')
        # hidden = h = f1(W1 x + b1)
        hidden = self.activation_function(T.dot(x, self.W) + self.b1)
        # output = x' = f2(W2 h + b2)
        output = self.output_function(T.dot(hidden, T.transpose(self.W)) + self.b2)

        transformed_data = theano.function(
            inputs=[x],
            outputs=[output]
        )
        return transformed_data

    def get_weights(self):
        """
        return the weights [W, b1, b2].
        """
        return [self.W.get_value(), self.b1.get_value(), self.b2.get_value()]
