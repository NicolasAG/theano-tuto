# https://www.youtube.com/watch?v=BuIsI-YHzj8
# http://deeplearning.net/tutorial/gettingstarted.html

import numpy as np

import matplotlib.pyplot as plt

import theano
import theano.tensor as T

# GPU acceleration
# theano.config.device = "gpu"

"""
>>> np.linspace(-1, 1, 11)
array([-1. , -0.8, -0.6, -0.4, -0.2,  0. ,  0.2,  0.4,  0.6,  0.8,  1. ])
"""
trainX = np.asarray(np.linspace(-1, 1, 101))
"""
>>> np.random.randn(3)
array([ 0.02066501,  0.76675396,  0.62194306])
"""
trainY = 2 * trainX + np.random.randn(trainX.size)

print "trainX is ", trainX
print "trainY is ", trainY

# Initialize symbolic variables
X = T.scalar("X")
Y = T.scalar("Y")


# Define symbolic model
def model(X, w):
    return X * w

# Initialize model parameter W
# hybrid variable -> need data associated with them
W = theano.shared(np.asarray([0.], dtype=theano.config.floatX), "W")
print "W =", W.get_value()

y_predicted = model(X, W)
print "y_predicted = X*W =", y_predicted

##
# plot the training set
# "bo" means blue circles representing each point
# "r" means read line
# x axis from -1.5 to +1.5
# y_predicted axis from -6 to +6
# ---
plt.plot(trainX, trainY, "bo", trainX, model(trainX, W.eval()), "r")
plt.axis([-1.5, 1.5, -6, 6])
plt.show()
##

# Define symbolic loss
cost = T.mean(T.sqr(y_predicted - Y))  # cost = average of sqrt(prediction - target)
print "cost function = mean(sqrt(y_predicted-Y)) =", cost
# Determine partial derivative of cost w.r.t. parameter W
gradient = T.grad(cost=cost, wrt=W)
print "gradient = partial derivative of cost w.r.t. W =", gradient
# Define how to update parameter W based on gradient
learning_rate = 0.01
updates = [[W, W - gradient * learning_rate]]
print "update = [[W, W-gradient*0.01]] =", updates

# Define theano function that compiles symbolic expressions
train = theano.function(
    inputs=[X, Y],
    outputs=cost,
    updates=updates,
    allow_input_downcast=True,
    mode="DebugMode"
)

# Iterate through data 100 times, updating parameter W after each iteration
for i in range(10):
    for x, y_target in zip(trainX, trainY):
        # print "train(x,y) with x=", x, " and y=", y_target
        train(x, y_target)
    print "iteration", i, ": W =", W.get_value()
    ##
    # plot the trained set
    # "bo" means blue circles representing each point
    # "r" means read line
    # x axis from -1.5 to +1.5
    # y_predicted axis from -6 to +6
    # ---
    plt.plot(trainX, trainY, "bo", trainX, model(trainX, W.eval()), "r")
    ##
plt.axis([-1.5, 1.5, -6, 6])
plt.show()
print "W =", W.get_value()  # something around 2
