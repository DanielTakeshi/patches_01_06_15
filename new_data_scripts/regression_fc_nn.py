"""
(c) January 2016 by Daniel Seita

Code for a single-layer, fully connected neural network, though it wouldn't be tough to extend to
ore than one hidden layer, but convolutions ... see the other regression_cnn.py code.

If I extend these to neural networks with more hidden layers, be careful not to mess anything up.
"""

import os
import numpy as np
import theano
import theano.tensor as T
import timeit


class LinearRegression(object):
    """ 
    The Linear Regression layer for the final output of the MLP. It's similar to LogisticRegression,
    but we will only have one output layer, and we don't use their 'errors' method. Also -- and this
    is a bit unusual -- we are going to apply T.nnet.sigmoid( ... ) to the predictions, since we
    KNOW that our output, P(FC) is going to be in [0,1]. This is obviously NOT generally true.
    """

    def __init__(self, input, n_in, n_out):
        """ 
        :input: A symbolic variable that describes the input of the architecture (one mini-batch).
        :n_in: The number of input units, the dimension of the data space.
        :n_out: The number of output units, the dimension of the labels (here it's one).
        """

        # Initialize the weights to be all zeros.
        self.W = theano.shared(value = np.zeros( (n_in, n_out), dtype=theano.config.floatX ),
                               name = 'W',
                               borrow = True)
        self.b = theano.shared(value = np.zeros( (n_out,), dtype=theano.config.floatX ),
                               name = 'b',
                               borrow = True)

        # p_y_given_x forms a matrix, and y_pred will extract first element from each list.
        # This caused a lot of confusion! It's basically the difference between [x] and x in python.
        # Don't forget to "squash" the P(FC) output to be in [0,1].
        self.p_y_given_x = T.nnet.sigmoid( T.dot(input, self.W) + self.b )
        self.y_pred = self.p_y_given_x[:,0]

        # Miscellaneous stuff
        self.params = [self.W, self.b]
        self.input = input

    def squared_errors(self, y):
        """ Returns the mean of squared errors of the linear regression on this data. """
        return T.mean((self.y_pred - y) ** 2)

    def absolute_errors(self, y):
        """ Returns the mean of absolute errors of the linear regression on this data. """
        return T.mean(np.absolute(self.y_pred - y))


class HiddenLayer(object):
    """
    Hidden Layer class for a Multi-Layer Perceptron. This is exactly the same as the reference
    code from the documentation, except for T.sigmoid instead of T.tanh.
    """

    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.nnet.sigmoid):
        """
        :rng: A random number generator for initializing weights.
        :input: A symbolic tensor of shape (n_examples, n_in).
        :n_in: Dimensionality of input.
        :n_out: Number of hidden units.
        :activation: Non-linearity to be applied in the hidden layer.
        """

        # W is initialized with W_values, according to the "Xavier method".
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low = -np.sqrt(6. / (n_in + n_out)),
                    high = np.sqrt(6. / (n_in + n_out)),
                    size = (n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)

        # Initialize the bias weights.
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        # The output of all the inputs, "squashed" via the activation function.
        lin_output = T.dot(input, self.W) + self.b
        self.output = lin_output if activation is None else activation(lin_output)

        # Miscellaneous stuff
        self.params = [self.W, self.b]
        self.input = input


class MLP(object):
    """ Multi-Layer Perceptron class. It consists of a HiddenLayer and a LinearRegression layer. """

    def __init__(self, rng, input, n_in, n_hidden, n_out, hidden_activation=T.nnet.sigmoid):
        """
        :rng: A random number generator for initializing weights.
        :input: Symbolic variable that describes the architecture of one mini-batch.
        :n_in: Dimension of each data point, i.e., the total number of features.
        :n_hidden: The number of hidden units.
        :n_out: The dimension of the space labels lie; here it's a scalar due to regression.
        """

        # One hidden layer with sigmoid activations, connected to the final LinearRegression layer
        self.hiddenLayer = HiddenLayer(rng = rng,
                                       input = input,
                                       n_in = n_in,
                                       n_out = n_hidden,
                                       activation = hidden_activation)

        # The logistic regression layer gets as input the hidden units of the linear reg. layer
        self.linRegressionLayer = LinearRegression(input = self.hiddenLayer.output,
                                                   n_in = n_hidden,
                                                   n_out = n_out)

        # Two norms, along with squared error and absolute errors (output of LinearRegression layer)
        self.L1 = abs(self.hiddenLayer.W).sum() + abs(self.linRegressionLayer.W).sum()
        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() + (self.linRegressionLayer.W ** 2).sum()
        self.squared_errors = self.linRegressionLayer.squared_errors
        self.absolute_errors = self.linRegressionLayer.absolute_errors

        # Miscellaneous
        self.params = self.hiddenLayer.params + self.linRegressionLayer.params
        self.input = input


def convert_data_theano(dataset):
    """ 
    Copying this from documentation online, including some of the nested 'shared_dataset' function,
    but I'm also returning the number of features, since it's easiest to detect that here.
    """
    train_set, valid_set = dataset[0], dataset[1]
    assert (train_set[0].shape)[1] == (valid_set[0].shape)[1], \
        "Number of features for train,val do not match: {} and {}.".format(train_set.shape[1],valid_set.shape[1])
    num_features = (train_set[0].shape)[1]

    def shared_dataset(data_xy, borrow=True):
        """ 
        Function that loads the dataset into shared variables. It is DIFFERENT from the online
        documentation since we can keep shared_y as floats; we won't be needing them as indices.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, shared_y

    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    rval = [(train_set_x,train_set_y), (valid_set_x,valid_set_y)]
    return rval,num_features


def do_regression(dataset, learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
                  batch_size=100, n_hidden=500, hidden_activation=T.nnet.sigmoid):
    """
    This will run the code for a fully-connected neural network with one hidden layer. The only
    thing we need is the data; all other values can be set at defaults.

    :dataset: A tuple (t,v) where t is itself a tuple of (train_data,train_values) and similarly for
        v, except it stands for the validation set.
    :hidden_activation: This is the activation for the sole hidden layer in this network. Possible
        values are T.nnet.sigmoid (default), T.tanh, or T.nnet.relu.
    """

    print("\nRunning regression for neural network with {} hidden units.".format(n_hidden))
    print("The activation: {}.".format(hidden_activation))

    # Get data into shared, correct Thenao format, and compute number of mini-batches.
    datasets, num_features = convert_data_theano(dataset) 
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    n_train_batches = int(train_set_x.get_value(borrow=True).shape[0] / batch_size)
    n_valid_batches = int(valid_set_x.get_value(borrow=True).shape[0] / batch_size)

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')
    
    # Symbolic variables for the data. 'index' is the index to a mini-batch.
    index = T.lscalar()
    x = T.matrix('x')
    y = T.vector('y') # This is NOT ivector because we have continuous outputs!

    # Construct my own MLP class based on similar code, but using a single continuous output, so n_out = 1.
    rng = np.random.RandomState(1234)
    classifier = MLP(rng = rng,
                     input = x,
                     n_in = num_features,
                     n_hidden = n_hidden,
                     n_out = 1,
                     hidden_activation = hidden_activation)

    # The cost function, symbolically, is DIFFERENT from their (logistic regression) example.
    cost = classifier.squared_errors(y) + (L1_reg * classifier.L1) + (L2_reg * classifier.L2_sqr)

    # Compiling a Theano function that computes the error by the model on a minibatch. For
    # intuition, let's actually return classifier.absolute_errors(), not squared_errors().
    validate_model = theano.function(inputs = [index],
                                     outputs = classifier.absolute_errors(y),
                                     givens = { 
                                        x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                                        y: valid_set_y[index * batch_size:(index + 1) * batch_size]
                                     })

    # Compute the gradient of the cost w.r.t. parameters; code matches the documentation.
    gparams = [T.grad(cost, param) for param in classifier.params]

    # How to update the model parameters as list of (variable, update_expression) pairs.
    updates = [ (param, param - learning_rate * gparam) for param, gparam in zip(classifier.params, gparams)]

    # Compiling a Theano function `train_model` that returns the cost AND updates parameters.
    train_model = theano.function(inputs = [index],
                                  outputs = cost,
                                  updates = updates,
                                  givens = {
                                      x: train_set_x[index * batch_size:(index + 1) * batch_size],
                                      y: train_set_y[index * batch_size:(index + 1) * batch_size]
                                  })

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    print("n_train_batches,n_valid_batches = ({},{})".format(n_train_batches,n_valid_batches))

    # Stopping parameters (might have to tweak).
    patience = 10000                # No matter what, look at this many examples in training
    patience_increase = 2           # If a new best is found, wait a factor of this much longer
    improvement_threshold = 0.995   # Relative improvement of this much is "significant"

    # Go through this many mini-batches before checking network on the validation set.
    validation_frequency = min(n_train_batches, patience/2)

    # Other variables of interest
    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()
    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch += 1

        for minibatch_index in range(n_train_batches):
            # Train model on this particular mini-batch, a fraction of the total training data.
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            # Evaluate on the ENTIRE validation set, because of the loop over n_valid_batches.
            if (iter + 1) % validation_frequency == 0:
                this_validation_loss = np.mean( [validate_model(i) for i in range(n_valid_batches)] )
                print("epoch {} (iter {}), minibatch {}/{}, validation MAE {:.5f}".format(epoch,
                                iter, minibatch_index + 1, n_train_batches, this_validation_loss))

                # If best valid so far, improve patience (sometimes) and update the 'best' variables.
                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    best_validation_loss = this_validation_loss
                    best_iter = iter
            
            # Break out of *inner* for loop, and break out of *outer* while loop.
            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print("Optimization complete with best validation score of {:.5f}.".format(best_validation_loss))
    print("The code ran for {} epochs, with {:.5f} epochs/sec.".format(epoch, 1. * epoch / (end_time - start_time)))


if __name__ == '__main__':
    """ Let's test it out with some default data if I'm calling this directly. """
    print("I shouldn't be calling this directly!")
