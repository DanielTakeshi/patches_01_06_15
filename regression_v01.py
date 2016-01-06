'''
This will be my first attempt at forming a regression pipeline. For now, call this in the same
directory with all the other numpy files.

(c) January 2016 by Daniel Seita
'''

import numpy as np
import random
import sys
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge


def print_results(clf, results_absolute, results_squared, name, coefficients=False):
    '''
    This might be useful to standardize the output that gets printed.
    '''
    print "Results for {}.".format(name)
    print "-> Mean residual squared: {}".format(np.mean(results_squared))
    print "-> Mean absolute difference: {}".format(np.mean(results_absolute))
    if coefficients:
        print "-> Part of the coefficient (full size = {}): \n{}".format(len(clf.coef_), clf.coef_[:10])
        print "-> Intercept: {}".format(clf.intercept_)


def perform_regression(X_train, X_val, y_train, y_val):
    '''
    Now we perform regression using whatever procedure we like. The kernels, especially RBF, will
    take a long time with high dimensional data and is probably not even that effective.
    '''

    # This is the other major customization. Modify and add tuples of (regressor, name) to list.
    regressors = [ 
                   (linear_model.LinearRegression(),             'Linear Regression'),
                   (linear_model.Ridge(alpha = 1.0),             'Ridge Regression'),
                   (KernelRidge(alpha = 1.0, kernel = 'linear'), 'Kernel Regression')
                 ]

    # Regress on all of these and print the output.
    for item in regressors:
        (clf,name) = item
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_val)
        results_absolute = np.absolute(predictions - y_val)
        results_squared = (predictions - y_val) ** 2
        print_results(clf, results_absolute, results_squared, name, coefficients=False)


def concatenate_data(N, target_data, axis, center, grasp_axis_gravity_angle, moment_arm_mag,
                     moment_arms_gravity_angle, w1_curvature_window, w1_gradx_window,
                     w1_grady_window, w1_projection_disc, w1_projection_window, w2_curvature_window, 
                     w2_gradx_window, w2_grady_window, w2_projection_disc, w2_projection_window): 
    '''
    Given all the data points at our disposal, we must format it into a way Python can understand
    (note that we can switch which data we use as input), and pretty print the output.
    '''

    function_arguments = locals()
    X = np.zeros((N,1)) # Just because we have to deal with concatenation and need a baseline.

    # Iteratively concatenate feature vectors "horizontally" to form design matrix X
    for key in sorted(function_arguments.keys()):
        if (key != 'N' and key != 'target_data' and function_arguments[key] is not None):

            # Before adding this to the data, check for NaNs.
            assert not np.isnan(np.sum(function_arguments[key])), "Problem with key = {}.".format(key)

            if (key == 'w1_projection_disc' or key == 'w2_projection_disc'):
                shape = function_arguments[key][0].shape # Lets us use different-sized grids
                reshape_tuple = (N, shape[0]*shape[1])
                X = np.concatenate( (X,function_arguments[key].reshape(reshape_tuple)) , axis=1)
            else:
                assert function_arguments[key].shape[0] == N
                X = np.concatenate( (X,function_arguments[key]) , axis=1)

    y = target_data

    # Daniel: If we're using pfc, we don't need to worry about the next part, just return y.
    ## Now remove spurious/erroneous cases; note that target_data should be aligned w/data.
    #index = 0
    #for fc_metric in target_data:
    #    if fc_metric < 0:
    #        X = np.delete(X, index, axis=0)
    #        y = np.delete(y, index, axis=0)
    #    else:
    #        index += 1

    return ((X.T)[1:].T, y)


def main():
    '''
    Here, pick the values we want to use for regression. The features we have are:

    - axis
    - center
    - grasp_axis_gravity_angle
    - moment_arm_mag
    - moment_arms_gravity_angle
    - w1_curvature_window
    - w1_gradx_window
    - w1_grady_window
    - w1_projection_disc
    - w1_projection_window
    - w2_curvature_window
    - w2_gradx_window
    - w2_grady_window
    - w2_projection_disc
    - w2_projection_window

    The (old) outputs are (were):

    - efc_L1_f_0.100000_tg_0.005000_rg_0.100000_to_0.005000_ro_0.100000
    - pfc_f_0.100000_tg_0.005000_rg_0.100000_to_0.005000_ro_0.100000

    The methods we use for regression are:

    - Ordinary Least Squares (Note: don't use this ... only for a simple start)
    - Reguliarized Least Squares (Actually this might seem OK)
    - Kernel Ridge Regression (with several different kernels but most aren't that effective)

    We also determine how much data to use, and the split between training and validation. Right now
    there are 302205 grasps, so we first pick a subset of those elements to use. Then determine a
    ratio for training (to get different models) and validation (to see what is best in practice).
    '''

    # These settings should be the only things to change, apart from perform_regression(...).
    # options = [n_1, n_2, ..., n_k] where n_0 is the number of test set instances, and each n_i
    # beyond that is the number of training instances to use. We iterate k-1 times.
    val = [1000]
    train = [10, 10, 10, 100, 100, 100, 1000, 1000, 1000, 5000, 5000, 5000, 10000, 10000, 10000,
             25000, 25000, 25000, 50000, 50000, 50000]
    options = val + train
    num_val = options[0]
    target = 'pfc'  # This is either pfc or efc. Use pfc only because values are a bit 'better'.

    # Load in the target data.
    target_data = None
    if target == 'efc':
        target_data = np.load('efc.npy')
    elif target == 'pfc':
        target_data = np.load('pfc.npy')
    else:
        raise ValueError("Error: target = " + target + " is not valid.")
    print "For output, we have target = {}.".format(target)

    # Load the rest of the data. We extract a subset of this for training purposes.
    print "Now loading all of our raw data, unshuffled (this may take a couple of minutes) ..."
    _axis = np.load('c_axis.npy')
    _center = np.load('c_center.npy')
    _grasp_axis_gravity_angle = np.load('c_grasp_axis_gravity_angle.npy')
    _moment_arm_mag = np.load('c_moment_arm_mag.npy')
    _moment_arms_gravity_angle = np.load('c_moment_arms_gravity_angle.npy')

    _w1_curvature_window = np.load('c_w1_curvature_window.npy')
    _w1_gradx_window = np.load('c_w1_gradx_window.npy')
    _w1_grady_window = np.load('c_w1_grady_window.npy')
    _w1_projection_disc = np.load('c_w1_projection_disc.npy')
    _w1_projection_window = np.load('c_w1_projection_window.npy')

    _w2_curvature_window = np.load('c_w2_curvature_window.npy')
    _w2_gradx_window = np.load('c_w2_gradx_window.npy')
    _w2_grady_window = np.load('c_w2_grady_window.npy')
    _w2_projection_disc = np.load('c_w2_projection_disc.npy')
    _w2_projection_window = np.load('c_w2_projection_window.npy')
    print "Done with loading the raw data."

    # Iterate through as many training instances as possible, based on 'options' and other stuff.
    # Each time this script is called, we have ONE validation set, but different training data.
    N = len(target_data)
    indices = np.random.permutation(N)
    validation_indices = indices[:num_val]
    other_indices = indices[num_val:]

    # For number of designated iterations, shuffle data THE SAME WAY for all data. Then regression.
    for i in range(len(options)-1):
        print "\n\n\tCurrently on regression iteration {} of {}.\n".format(i+1, len(options)-1)
        print "Now extracting the training data of {} elements ...".format(options[i+1])
        assert options[i+1] <= len(other_indices)
        train_indices = np.array(random.sample(other_indices, options[i+1]))
        assert len(np.intersect1d(validation_indices, train_indices)) == 0
        indices = np.concatenate((validation_indices, train_indices)) # validation stays first/same

        # Set variables (after the target data) to "None" if I want to avoid using it/them.
        # Otherwise, for variable name 'xyz', use '_xyz[indices]' as the input.
        X,y = concatenate_data(len(indices), target_data[indices],
                               axis = _axis[indices],
                               center = _center[indices],
                               grasp_axis_gravity_angle = _grasp_axis_gravity_angle[indices],
                               moment_arm_mag = _moment_arm_mag[indices],
                               moment_arms_gravity_angle = _moment_arms_gravity_angle[indices],
                               w1_curvature_window = _w1_curvature_window[indices],
                               w1_gradx_window = _w1_gradx_window[indices],
                               w1_grady_window = _w1_grady_window[indices],
                               w1_projection_disc = _w1_projection_disc[indices],
                               w1_projection_window = _w1_projection_window[indices],
                               w2_curvature_window = _w2_curvature_window[indices],
                               w2_gradx_window = _w2_gradx_window[indices],
                               w2_grady_window = _w2_grady_window[indices],
                               w2_projection_disc = _w2_projection_disc[indices],
                               w2_projection_window = _w2_projection_window[indices])

        # Now split into training and validation.
        X_train = X[num_val:]
        y_train = y[num_val:]
        X_val = X[:num_val]
        y_val = y[:num_val]
        print "Some statistics on our data for this regression shape:"
        print "\tX_train.shape = {}\n\tX_val.shape = {}".format(X_train.shape, X_val.shape)

        # Now call a regression-specific code and pretty print output.
        perform_regression(X_train, X_val, y_train, y_val)


if __name__ == '__main__':
    main()
