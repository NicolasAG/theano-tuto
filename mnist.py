import cPickle
import gzip
import os
import matplotlib.pyplot as plt
import numpy as np


def load_data(dataset):
    """
    Loads the dataset
    @param dataset - string to the path of the dataset
    """
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # check if the dataset is in the data directory.
        new_path = os.path.join(os.path.split(__file__)[0], "..", "data", dataset)
        if os.path.isfile(new_path) or data_file == "mnist.pkg.gz":
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == "mnist.pkl.gz":
        import urllib
        origin = "http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz"
        print "Downloading data from %s" % origin
        urllib.urlretrieve(origin, dataset)

    print "loading data ..."
    f = gzip.open(dataset, "rb")
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    # train_set, valid_set, test_set format: tuple (input, target)
    #  input is a numpy.ndarray of 2 dimensions (a matrix) where row = 1 example
    #  target is a numpy.ndarray of dimension 1 (vector) that gives the target value for each example.
    return train_set, valid_set, test_set


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


def plot_confusion_matrix(cm, title='Confusion matrix'):
    """
    Plot confusion matrix.
    :param cm:
    :param title: the title of the image
    :return:
    """
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(range(10), tuple(range(10)), rotation=45)
    plt.yticks(range(10), tuple(range(10)))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
