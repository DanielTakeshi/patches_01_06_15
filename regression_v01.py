'''
This will be my first attempt at forming a regression pipeline. For now, call this in the same
directory with all the other numpy files.

(c) January 2016 by Daniel Seita
'''

import numpy as np
import random
import sys
from sklearn.kernel_ridge import KernelRidge
from sklearn import linear_model
from sklearn import preprocessing


def print_results(absolute, absolute_clipped, squared, squared_clipped, median, mean, name):
    '''
    This might be useful to standardize the output that gets printed.
    '''
    print "{} on validation set:".format(name)
    print "  -> Mean *ABSOLUTE* difference: {:.5f}".format(absolute)
    print "    -> clipped: {:.5f}, uniform guess (median): {:.5f})".format(absolute_clipped, median)
    print "  -> Mean *SQUARED* difference: {:.5f}.".format(squared)
    print "    -> clipped: {:.5f}, uniform guess (mean): {:.5f})".format(squared_clipped, mean)


def perform_regression(X_train, X_val, y_train, y_val, results, standardized=True, full=True):
    '''
    Now we perform regression using whatever procedure we like. The kernels, especially RBF, will
    take a long time with high dimensional data and are not that effective (they also seg-fault).
    Also, store the results in 'results' according to key name.
    '''

    # This is the other major customization. Modify and add tuples of (regressor, name) to list.
    # Don't use the Lasso predictor, it doesn't seem to do well.
    print "Note: for y_train, mean = {:.5f}, median = {:.5f}.".format(np.mean(y_train), np.median(y_train))
    regressors = [ 
                   (linear_model.LinearRegression(), 'Linear Regression'),
                   #(linear_model.Ridge(alpha = 1e-10), 'Ridge Regression (alpha = 1e-10)'),
                   (linear_model.Ridge(alpha = 1e-6), 'Ridge Regression (alpha = 1e-6)'),
                   #(linear_model.Ridge(alpha = 1e-2), 'Ridge Regression (alpha = 1e-2)'),
                   (linear_model.SGDRegressor(loss="huber", penalty="l2"), 'SGD Regressor (Huber, L2)')
                 ]

    #if len(y_train) <= 40000:
    #    regressors.append((KernelRidge(alpha = 1.0, kernel = 'linear'), 'Kernel Regression (Linear)'))

    # Regress on all of these and print the output.
    median = np.mean( np.absolute(np.median(y_train) - y_val) )
    mean = np.mean( (np.mean(y_train) - y_val) ** 2 )

    for item in regressors:
        clf,name = item
        predictions = clf.fit(X_train, y_train).predict(X_val) # This fits model then tests it
        absolute            = np.mean( np.absolute(predictions - y_val) )
        absolute_clipped    = np.mean( np.absolute(np.clip(predictions,0,1) - y_val) )
        squared             = np.mean( (predictions - y_val) ** 2 )
        squared_clipped     = np.mean( (np.clip(predictions,0,1) - y_val) ** 2 )
        print_results(absolute, absolute_clipped, squared, squared_clipped, median, mean, name)

        # New, this is for the ultimate output
        end = '-no-standardized-no-full'
        if standardized and full:
            end = '-yes-standardized-yes-full'
        if standardized and not full:
            end = '-yes-standardized-no-full'
        if not standardized and full:
            end = '-no-standardized-yes-full'

        key1 = name + ' abs-no-clip' + end
        key2 = name + ' abs-yes-clip' + end
        key3 = name + ' sqerr-no-clip' + end
        key4 = name + ' sqerr-yes-clip' + end
        if key1 in results:
            results[key1] = results[key1] + [absolute]
        else:
            results[key1] = [absolute]
        if key2 in results:
            results[key2] = results[key2] + [absolute_clipped]
        else:
            results[key2] = [absolute_clipped]
        if key3 in results:
            results[key3] = results[key3] + [squared]
        else:
            results[key3] = [squared]
        if key4 in results:
            results[key4] = results[key4] + [squared_clipped]
        else:
            results[key4] = [squared_clipped]

        # Now let's save this to files (if desired, for histogram of output values)
        # Actually this way would overwrite the other names from smaller training sizes...
        #np.savetxt('predictions_' + name + '.txt', predictions)

        # NEW AGAIN! Now let's divide into four quartiles based on y_val. I guess I'll just print
        # here even though it'll be a bit awkward to try and retrieve this information. Yeah, have
        # to figure out how to re-organize this nicely...
        print "\nNow let's deal with the PERCENTILES:"
        thresh1 = np.percentile(y_val, 33)
        thresh2 = np.percentile(y_val, 66)
        indices1 = np.where(y_val < thresh1)[0]
        tmp1 = np.where(y_val >= thresh1)[0]
        tmp2 = np.where(y_val < thresh2)[0]
        indices2 = np.intersect1d(tmp1,tmp2)
        indices3 = np.where(y_val >= thresh2)[0]
       
        # Now, using indices, extract the appropriate values, clip, take absolute value, then print.
        predictions1 = predictions[indices1]
        y_val1 = y_val[indices1]
        results1 = np.mean(np.absolute(np.clip(predictions1,0,1)-y_val1))
        print "For bottom third, avg abs diff = {}.".format(results1)
        
        predictions2 = predictions[indices2]
        y_val2 = y_val[indices2]
        results2 = np.mean(np.absolute(np.clip(predictions2,0,1)-y_val2))
        print "For middle third, avg abs diff = {}.".format(results2)

        predictions3 = predictions[indices3]
        y_val3 = y_val[indices3]
        results3 = np.mean(np.absolute(np.clip(predictions3,0,1)-y_val3))
        print "For top third, avg abs diff = {}.".format(results3)
        print ""

    print "\nDone with this regression. The (current) results:"
    keys_ordered = []
    for key in results:
        keys_ordered.append(key)
    keys_ordered.sort()
    for item in keys_ordered:
        rounded = [float("{:5f}".format(x)) for x in results[item]]
        print "\t{} \t{}".format(item, rounded)
    print ""


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
    # train_sizes = [n_1, n_2, ..., n_k] where each n_i is the # of training instances to use.
    num_val = 10000 # Change this to change validation size
    train_sizes = [10000, 50000, 100000, 280000]
    target = 'pfc'  # This is either pfc or efc. Use pfc because their values are 'better'.

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

    # Let's store results in here. A bit manual since it requires some fixed values but w/e.
    results = {}

    # For number of designated iterations, shuffle data THE SAME WAY for all data. Then regression.
    for (i,num) in enumerate(train_sizes):
        print "\n\n\tCurrently on regression iteration {} of {}.\n".format(i+1, len(train_sizes))
        print "Now extracting the training data of {} elements ...".format(num)
        assert num <= len(other_indices)
        train_indices = np.array(random.sample(other_indices, num))
        if num > 20:
            print "train_indices[:20] = {}".format(train_indices[:20])
        else:
            print "train_indices = {}".format(train_indices)
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
        scaler = preprocessing.StandardScaler().fit(X_train) # Only scale based on training
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        print "Some statistics on our data for this regression shape:"
        print "\tX_train.shape = {}\n\tX_val.shape = {}".format(X_train_scaled.shape, X_val_scaled.shape)

        # NEW! let's run regression using different feature sets!
        print "\nDoing regression on the FULL data:" # full=True (DIFFERENT later)
        perform_regression(X_train_scaled, X_val_scaled, y_train, y_val, results, standardized=True, full=True)

        ##X,y = concatenate_data(len(indices), target_data[indices],
        ##                       axis = _axis[indices],
        ##                       center = _center[indices],
        ##                       grasp_axis_gravity_angle = _grasp_axis_gravity_angle[indices],
        ##                       moment_arm_mag = _moment_arm_mag[indices],
        ##                       moment_arms_gravity_angle = _moment_arms_gravity_angle[indices],
        ##                       w1_curvature_window = _w1_curvature_window[indices],
        ##                       w1_gradx_window = None,#_w1_gradx_window[indices],
        ##                       w1_grady_window = None,#_w1_grady_window[indices],
        ##                       w1_projection_disc = _w1_projection_disc[indices],
        ##                       w1_projection_window = _w1_projection_window[indices],
        ##                       w2_curvature_window = _w2_curvature_window[indices],
        ##                       w2_gradx_window = None,#_w2_gradx_window[indices],
        ##                       w2_grady_window = None,#_w2_grady_window[indices],
        ##                       w2_projection_disc = _w2_projection_disc[indices],
        ##                       w2_projection_window = _w2_projection_window[indices])

        ##X_train = X[num_val:]
        ##y_train = y[num_val:]
        ##X_val = X[:num_val]
        ##y_val = y[:num_val]
        ##scaler = preprocessing.StandardScaler().fit(X_train) # Only scale based on training
        ##X_train_scaled = scaler.transform(X_train)
        ##X_val_scaled = scaler.transform(X_val)
        ##print "Some statistics on our data for this regression shape:"
        ##print "\tX_train.shape = {}\n\tX_val.shape = {}".format(X_train_scaled.shape, X_val_scaled.shape)

        ##print "\nDoing regression on the PARTIAL data:" # Hence, full=False (NOT same as earlier)
        ##perform_regression(X_train_scaled, X_val_scaled, y_train, y_val, results, standardized=True, full=False)

    print "All done. Now let's print out the results (i.e., a dictionary):"


if __name__ == '__main__':
    main()
