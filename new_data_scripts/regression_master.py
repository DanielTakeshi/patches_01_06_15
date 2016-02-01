"""
(c) January 2016 by Daniel Seita

This is the master script that will manage the data, etc., and then call other scripts. Do this in
the same directory as all the other numpy files.
"""

import numpy as np
import random
import sys
import theano
import theano.tensor as T
import timeit
from sklearn import preprocessing
import regression_linear
import regression_fc_nn
import regression_svm


def concatenate_data(indices, target, data_features):
    """
    Given all the features, we combine them together to form the full data matrix X. The
    concatenation is done "horizontally" with axis=1, NOT the default vertical stacking.
    
    :param: indices The indices to use for this particular training (and validation!) batch.
    :param: target The output (y), which we extract using target[indices].
    :param: data_features The data, where data_features[k] = either data, or None (ignoring feature)
    """

    # We need a "base" array to serve as concatenation; we will delete it later.
    N = len(indices)
    X = np.zeros((N,1)) 

    # Iteratively concatenate feature vectors "horizontally" to form data matrix X
    for item in data_features:
        if item is not None:

            # Let's pick out the indices we need first, a subset of the data.
            data = item[indices]

            # Now concatenate. First special case: these are actually 2D arrays, so "flatten" them.
            if (len(data[0].shape) > 1):
                shape = data[0].shape # The first item is a 2D array (a disc)
                reshape_tuple = (N, shape[0]*shape[1])
                X = np.concatenate((X,data.reshape(reshape_tuple)) , axis=1)
            elif (len(data.shape) == 1): # Second case, with shape of (N,) which we need to re-arrange
                X = np.concatenate((X,data.reshape((N,1))), axis=1)
            else:
                X = np.concatenate((X,data) , axis=1)

    y = target[indices]
    return ((X.T)[1:].T, y)


def load_combined_data(feature_name, file_numbers):
    """
    Loads data for one feature and concatenates them "vertically". Also, due to the presence of NaNs
    in the data, let's NOT put in any assertions here and save those for later. There are different
    numberings for each file, hence the 'num_file' parameter, though we start with 0-indexing which
    is why we add a -1.

    I start with np.load( ... ) for the zero case, because we know the zero case has appropriate
    dimensions, but that is only so we have a reference when we do concatenation, which is why we
    return results[1:] at the end to ignore the first part we tacked on.

    :param: file_numbers A list [f1, ..., fk] of indices of the files to use (e.g., [3, 5, 10]).
    """

    first_shape = np.load(feature_name + "/" + feature_name + "_00.npz")['arr_0'].shape

    result = np.zeros(1)
    if len(first_shape) == 2:
        result = np.zeros((1, first_shape[1]))
    elif len(first_shape) == 3:
        result = np.zeros((1, first_shape[1], first_shape[2]))

    for i in file_numbers:
        padded_digit = '{0:02d}'.format(i)
        next_file = np.load(feature_name + "/" + feature_name + "_" + padded_digit + ".npz")['arr_0']
        result = np.concatenate((result,next_file), axis=0)
    return result[1:]


def main():
    """
    Set up the data loading pipeline, feature selection, regression options, and other miscellaneous
    tasks as needed to make the results clean, crisp, readable and informative.
    """
    
    ###########
    # OPTIONS #
    ###########

    # The number of data points in the validation set. This is FIXED across all training sets.
    num_val = 10000

    # The training set sizes, [n_1, n_2, ..., n_k], to use, to test on a fixed validation set.
    train_sizes = [5000, 10000, 30000, 50000, 100000, 200000, 300000]

    # Which of the files to use for each feature (currently have 0 through 20 to use). Note that each
    # file contains perhaps 100k points so this list should be large enough for num_val and train_sizes.
    file_numbers = [0,1,2]

    # Decide which features to use (True/False). If using all, data loading takes about 90 seconds.
    center                  = True
    com                     = True
    moment_arms             = True
    obj_ids                 = True
    patch_orientation       = True
    surface_normals         = True
    w1_curvature_window     = True
    w1_gradx_window         = True
    w1_grady_window         = True
    w1_projection_disc      = True
    w1_projection_window    = True
    w2_curvature_window     = True
    w2_gradx_window         = True
    w2_grady_window         = True
    w2_projection_disc      = True
    w2_projection_window    = True

    # Load features separately, using load_combined_data due to a few quirks. Put this all in a list.
    print("Now loading our raw data, unshuffled (this may take a few minutes) ...")
    data_features = [
        load_combined_data("center",file_numbers) if center else None,
        load_combined_data("com",file_numbers) if com else None,
        load_combined_data("moment_arms",file_numbers) if moment_arms else None,
        load_combined_data("obj_ids",file_numbers) if obj_ids else None,
        load_combined_data("patch_orientation",file_numbers) if patch_orientation else None,
        load_combined_data("surface_normals",file_numbers) if surface_normals else None,
        load_combined_data("w1_curvature_window",file_numbers) if w1_curvature_window else None,
        load_combined_data("w1_gradx_window",file_numbers) if w1_gradx_window else None,
        load_combined_data("w1_grady_window",file_numbers) if w1_grady_window else None,
        load_combined_data("w1_projection_disc",file_numbers) if w1_projection_disc else None,
        load_combined_data("w1_projection_window",file_numbers) if w1_projection_window else None,
        load_combined_data("w2_curvature_window",file_numbers) if w2_curvature_window else None,
        load_combined_data("w2_gradx_window",file_numbers) if w2_gradx_window else None,
        load_combined_data("w2_grady_window",file_numbers) if w2_grady_window else None,
        load_combined_data("w2_projection_disc",file_numbers) if w2_projection_disc else None,
        load_combined_data("w2_projection_window",file_numbers) if w2_projection_window else None
    ]
    
    # Just for some extra debugging/information -- show dimensions of features.
    for i in range(len(data_features)):
        if data_features[i] is not None:
            print(data_features[i].shape)
    print("The above with file_numbers = {}".format(file_numbers))

    # Load the target data. Be careful that the prefix is right, and that we have enough files.
    target_prefix = "pfc_f_01/pfc_f_0.100000_tg_0.005000_rg_0.100000_to_0.005000_ro_0.100000_"
    target = np.zeros(1)
    for i in file_numbers:
        padded_digit = '{0:02d}'.format(i)
        next_file = np.load(target_prefix + padded_digit + ".npz")['arr_0']
        target = np.concatenate((target,next_file), axis=0)
    assert not np.isnan(np.sum(target)), "NaN with the target values."
    target = target[1:] # Daniel: don't forget this due to target = np.zeros(1) to start.
    print("Done with loading raw data, AND the targets (shape: {}).".format(target.shape))

    # Individual features are in one np.array, but before proceeding, detect and remove NaNs.
    print("Now detecting NaNs and removing them if necessary ...")
    nan_indices = []
    for index,item in enumerate(data_features):
        if item is None:
            continue
        if np.isnan(np.sum(item)):
            nans_here = np.unique(np.where(np.isnan(item))[0])
            print("Detecting {} NaNs from index {} ...".format(len(nans_here),index))
            nan_indices = np.concatenate( (nan_indices, nans_here) )
    unique_nan_indices = np.unique(nan_indices)
    print("Total unique NaN indices to remove: {}.".format(len(unique_nan_indices)))

    # Removing NaNs if needed; 'i' indexes into # of elements, but 'index' is what we use to delete.
    if len(unique_nan_indices) > 0:
        print("Now removing the NaNs (this may take a few minutes if >= 50 things to remove) ...")
        index = 0
        for i in range(len(target)):
            if i in unique_nan_indices:
                for k in range(len(data_features)):
                    if data_features[k] is not None:
                        data_features[k] = np.delete(data_features[k], index, axis=0)
            else:
                index += 1
   
    # Let's just reaffirm that the data is what we need.
    for index,val in enumerate(data_features):
        if val is not None:
            assert not np.isnan(np.sum(val))
    print("All NaN tests passed; our data should be reasonable.")

    # Whew. Now we iterate through as many times as we designated in 'train_sizes' earlier.
    # We have ONE fixed validation set each time, but the TRAINING data will be different.
    print("Our data has {} elements total (not all used in each training run).".format(len(target)))
    print("Now starting the training process with train_sizes =\n{}.\n".format(train_sizes))
    N = len(target)
    indices = np.random.permutation(N)
    validation_indices = indices[:num_val]
    other_indices = indices[num_val:]

    # For number of designated iterations, shuffle data THE SAME WAY for all data. Then regression.
    for (i,num) in enumerate(train_sizes):

        ################
        # PREPARE DATA #
        ################

        print("\n\n\tCurrently on regression iteration {} of {}.\n".format(i+1, len(train_sizes)))
        print("Now extracting the training data of {} elements ...".format(num))
        assert num <= len(other_indices)
        train_indices = np.array(random.sample(list(other_indices), num))
        assert len(np.intersect1d(validation_indices, train_indices)) == 0
        indices = np.concatenate((validation_indices, train_indices)) # validation stays first/same

        # NOW let's finally form our data in the (X,y) format by forming X based on our features.
        # Then split the data into training and validation sets.
        X,y = concatenate_data(indices, target, data_features)
        X_train = X[num_val:]
        y_train = y[num_val:]
        X_val = X[:num_val]
        y_val = y[:num_val]

        # Scale the data so it gets normalized --- ONLY on the training data, obviously!
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        data = [(X_train_scaled,y_train),(X_val_scaled,y_val)]

        # Now let's print some basic statistics, including a "baseline" mean and median performance.
        print("Some statistics on our data for this regression shape:")
        print("\tX_train.shape = {}\n\tX_val.shape = {}".format(X_train_scaled.shape, X_val_scaled.shape))
        mean = np.mean( (np.mean(y_train) - y_val) ** 2 )
        median = np.mean( np.absolute(np.median(y_train) - y_val) )        
        print("Baseline M.S.E. performance: {:.5f}.".format(mean))
        print("Baseline M.A.E. performance: {:.5f}.".format(median))

        #####################
        # ACTUAL REGRESSION #
        #####################

        output = 'results/'

        # Linear regression first. Save predictions for plotting.
        preds_01 = regression_linear.do_regression(data, alpha=1e-3)
        preds_02 = regression_linear.do_regression(data, alpha=1e-8)
        np.save(output + 'preds_01_train_' + str(num) + '_iter_' + str(i), preds_01)
        np.save(output + 'preds_02_train_' + str(num) + '_iter_' + str(i), preds_02)

        # SVM kernel regression, but if I'm doing more than 30k data points, it takes a long time.
        if num < 40000:
            preds_03 = regression_svm.do_regression(data, kernel='rbf', cache_size=2000)
            np.save(output + 'preds_03_train_' + str(num) + '_iter_' + str(i), preds_03)
    
        # Fully connected neural networks. Unfortunately I don't know how to return predictions.
        regression_fc_nn.do_regression(data, learning_rate=0.001, L1_reg=0.00, L2_reg=0.00001, 
                                       n_epochs=100, batch_size=100, n_hidden=100,
                                       hidden_activation=T.nnet.sigmoid)

        # At the very end, let's actually save the y_val output so we can plot the residuals.
        np.save(output + 'y_val_train_' + str(num) + '_iter_' + str(i), y_val)

    print("\nAll Done! Whew!")


if __name__ == '__main__':
    main()
