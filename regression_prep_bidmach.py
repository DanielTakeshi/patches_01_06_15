"""
(c) January 2016 by Daniel Seita

This consists of three parts. Call the functions (in the main method) based on the stage I'm in.

The first part will go through the 21 data files and, for each one, create a BIDMach-like matrix,
stored as a simple text file using np.savetxt(...) with some number of digits of precision. Note
that we will be using the object IDs as the "first feature" so the first row in all 21 matrices
should contain the object IDs, and also, each file will have 501 objects. ALSO, we will go through
all six of the possible targets and then make those into BIDMach arrays as well! EDIT: use pickle
files, not text files.

The second (and third) parts will use these 21 data files and reorganize the data to make it more
suitable for BIDMach and regression. In particular, it will: (1) create a smaller number of data
matrices (I'm thinking 10), though obviously we have to make this stuff coincide with all the
targets, (2) SHUFFLE the data by object IDs so that no object ID has its data points dispersed
across multiple data matrices, (3) delete any feature that is all zero, if any [EDIT this is not an
issue, ignore ...], and (4) normalize the data, so zero-mean plus variance. This will require going
through each file individually to collect the statistics (weighted by the number of data points),
and then going through each one AGAIN to actually perform the standardization.
"""

import numpy as np
import random
import sys


def concatenate_data(N, data_features):
    """
    Given all the features, we combine them together to form the full data matrix. The concatenation
    is done "vertically" with axis=1, so that one column represents one instance.

    :param: N The number of data points for this matrix, i.e., columns.
    :param: data_features The data, where data_features[k] = either data, or None (ignoring feature)
    """

    # We need a "base" array to serve as concatenation; we will delete it later.
    X = np.zeros((1,N)) 

    # Iteratively concatenate feature vectors "vertically" to form data matrix X
    for data in data_features:
        if data is not None:

            # Now concatenate. First special case: these are actually 2D arrays, so "flatten" them.
            if (len(data[0].shape) > 1):
                shape = data[0].shape # The first item is a 2D array (a disc)
                reshape_tuple = (N, shape[0]*shape[1])
                data = np.reshape(data, reshape_tuple)
                X = np.concatenate((X, data.T)) # I'm transposing here just to keep things safe
                print("Just concatenated data with shape {}.".format((data.T).shape))
            elif (len(data.shape) == 1): # Second case, with shape of (N,) which we need to re-arrange
                data = np.reshape(data, (1,N))
                X = np.concatenate((X, data)) # No transpose needed since we did a (1,N) reshape
                print("Just concatenated data with shape {}.".format(data.shape))
            else:
                X = np.concatenate((X, data.T)) # Again, have to transpose
                print("Just concatenated data with shape {}.".format((data.T).shape))

    # This time just do X[1:] which will remove the first sub-array, which is all zeros.
    return X[1:]


def load_combined_data(feature_name, file_number):
    """
    Loads data for one feature and concatenates them "vertically", but here there's no need to
    concatenate because we only have one file number. Yeah, it's a lot easier now.

    :param: file_number A single file index to use.
    """
    padded_digit = '{0:02d}'.format(file_number)
    data_file = np.load(feature_name + "/" + feature_name + "_" + padded_digit + ".npz")['arr_0']
    return data_file


def do_part_one():
    """
    Set up the data loading pipeline, feature selection, regression options, and other miscellaneous
    tasks as needed to make the results clean, crisp, readable and informative.

    OH, now let's do something new: keep track of all the rows (i.e., features) that are zero, at
    the end, and print those out.
    """
    
    for file_number in range(0,20+1):
        padded_digit = '{0:02d}'.format(file_number)

        # Decide which features to use (True/False).
        obj_ids                 = True # Do this first and then the rest in alphabetical order
        center                  = True
        com                     = False # This is a useless feature, everything is zero.
        moment_arms             = True
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
        print("\nCurrently on digit = {}.".format(padded_digit))
        print("Now loading our raw data, unshuffled (this may take a few minutes) ...")
        data_features = [
            load_combined_data("obj_ids",file_number) if obj_ids else None,
            load_combined_data("center",file_number) if center else None,
            load_combined_data("com",file_number) if com else None,
            load_combined_data("moment_arms",file_number) if moment_arms else None,
            load_combined_data("patch_orientation",file_number) if patch_orientation else None,
            load_combined_data("surface_normals",file_number) if surface_normals else None,
            load_combined_data("w1_curvature_window",file_number) if w1_curvature_window else None,
            load_combined_data("w1_gradx_window",file_number) if w1_gradx_window else None,
            load_combined_data("w1_grady_window",file_number) if w1_grady_window else None,
            load_combined_data("w1_projection_disc",file_number) if w1_projection_disc else None,
            load_combined_data("w1_projection_window",file_number) if w1_projection_window else None,
            load_combined_data("w2_curvature_window",file_number) if w2_curvature_window else None,
            load_combined_data("w2_gradx_window",file_number) if w2_gradx_window else None,
            load_combined_data("w2_grady_window",file_number) if w2_grady_window else None,
            load_combined_data("w2_projection_disc",file_number) if w2_projection_disc else None,
            load_combined_data("w2_projection_window",file_number) if w2_projection_window else None
        ]
        
        # Just for some extra debugging/information -- show dimensions of features.
        for i in range(len(data_features)):
            if data_features[i] is not None:
                print(data_features[i].shape)

        # Load all the six target datasets.
        target_pfc_005 = np.load("pfc_f_005/pfc_f_0.050000_tg_0.002500_rg_0.050000_to_0.002500_ro_0.050000_" + padded_digit + ".npz")['arr_0']
        target_pfc_010 = np.load("pfc_f_01/pfc_f_0.100000_tg_0.005000_rg_0.100000_to_0.005000_ro_0.100000_" + padded_digit + ".npz")['arr_0']
        target_pfc_020 = np.load("pfc_f_02/pfc_f_0.200000_tg_0.010000_rg_0.200000_to_0.010000_ro_0.200000_" + padded_digit + ".npz")['arr_0']
        target_vfc_005 = np.load("vfc_f_005/vfc_f_0.050000_tg_0.002500_rg_0.050000_to_0.002500_ro_0.050000_" + padded_digit + ".npz")['arr_0']
        target_vfc_010 = np.load("vfc_f_01/vfc_f_0.100000_tg_0.005000_rg_0.100000_to_0.005000_ro_0.100000_" + padded_digit + ".npz")['arr_0']
        target_vfc_020 = np.load("vfc_f_02/vfc_f_0.200000_tg_0.010000_rg_0.200000_to_0.010000_ro_0.200000_" + padded_digit + ".npz")['arr_0']

        assert not np.isnan(np.sum(target_pfc_005)), "NaN with the target values."
        assert not np.isnan(np.sum(target_pfc_010)), "NaN with the target values."
        assert not np.isnan(np.sum(target_pfc_020)), "NaN with the target values."
        assert not np.isnan(np.sum(target_vfc_005)), "NaN with the target values."
        assert not np.isnan(np.sum(target_vfc_010)), "NaN with the target values."
        assert not np.isnan(np.sum(target_vfc_020)), "NaN with the target values."

        # Now getting this into a (1,N) shape rather than (N,) (i.e., nothing after the comma).
        N = len(target_pfc_005) 
        target_pfc_005 = np.reshape(target_pfc_005, (1,N))
        target_pfc_010 = np.reshape(target_pfc_010, (1,N))
        target_pfc_020 = np.reshape(target_pfc_020, (1,N))
        target_vfc_005 = np.reshape(target_vfc_005, (1,N))
        target_vfc_010 = np.reshape(target_vfc_010, (1,N))
        target_vfc_020 = np.reshape(target_vfc_020, (1,N))
        print("Done with loading raw data, AND the targets (shape: {}).".format(target_pfc_005.shape))

        # Now concatenate the data into one matrix. We need N to be the number of elements (i.e., columns)
        data_matrix = concatenate_data(N, data_features)
        
        # Detect if any rows are zero (effectively, 1e-7, btw). That way, we should delete them later.
        # UPDATE: This is no longer an issue, but I have the framework in case I need to change it.
        epsilon = 1e-7 # I.e., 0.0000001 (this means 7 decimal places, including that final "1")
        print("Here are the features that are all zero ({}) and thus we should later remove:".format(epsilon))
        boolean_data_matrix = data_matrix > epsilon
        print(np.where(~boolean_data_matrix.any(axis=1))[0])

        # The really annoying thing is that %1.6f will require at least one float before the decimal,
        # but allows more if the number is at least 10, BUT the six means that there will ALWAYS be six
        # decimal places, so 1.1 turns into 1.100000. That's annoying.
        # UPDATE change of plans, don't use fmt='%1.7f', just use np.save(...).
        print("Now saving the data_matrix, with shape: {} with SEVEN decimal places of precision.".format(data_matrix.shape))
        np.save("grasp_data_" + padded_digit, data_matrix)
        np.save("grasp_target_pfc_005_" + padded_digit, target_pfc_005)
        np.save("grasp_target_pfc_010_" + padded_digit, target_pfc_010)
        np.save("grasp_target_pfc_020_" + padded_digit, target_pfc_020)
        np.save("grasp_target_vfc_005_" + padded_digit, target_vfc_005)
        np.save("grasp_target_vfc_010_" + padded_digit, target_vfc_010)
        np.save("grasp_target_vfc_020_" + padded_digit, target_vfc_020)

    print("\nAll Done! Whew!")


def do_part_two(num_features = 1595, num_groups):
    """ 
    Now proceed to the second part. With all of the data files defined, we have to re-shuffle. I
    really hope Python's garbage collection can handle this kind of stuff, or else I'm going to have
    to split this up into two or more runs. Here is the procedure for the saving of output files:

    We go through each of the 21 normalized files. Split up the 501 object IDs into, say, 5 groups
    of 10. So object IDs for group 0 are [0,0,1,1,1,2,...,499,500,500,500], i.e., the first two were
    from object ID 0, etc. These groups will be distinct, i.e., group 1 might be [4,6,8,10,...],
    group 2 might be [0,12,204,...], etc., but they should be roughly equal in size.

    Then, for each group, we copy the grasps that correspond to those object IDs (i.e., columns --
    though we may want to transpose the matrix to make it easier to refer to it by rows) and save
    them into arrays. What this means is, after each of the 21 files has been processed, we have
    21*5 total data files, all with 1594 rows (we're *NOT* including the object IDs after this
    point, because with normalization it's not really a "good" feature).

    Then the third procedure (not this method) will randomly pick some ordering to combine these
    21*5 files into a smaller set of, say, 10 files. This is not true randomization, more like
    "block randomization" but it's the best I can do on short notice. Basically, if the last of the
    21 original files is from a different dataset, then it gets split into 5, and then those 5 get
    mixed in with other data files, so (ideally) those grasps from that special dataset are
    distributed among those 10 files somehow. To increase the randomness, I can create finer and
    finer divisons, e.g., using 15 groups instead of 5 groups for each of the 21 files.

    Remember that all of this MUST get coordinated with ALL of the target files!

    :param: num_features The number of features. I should know this in advance. The default is 1595
        since I'm ignoring the three from "center of mass."
    :param: num_groups For each of the 21 original files, we split it up into 'num_groups' groups.
    """

    num_files = 21

    # Store mean and std here as we sift through data. We'll keep appending them as extra rows, so
    # these will be (21 x num_features) arrays. This is approximate as we'll assume equal weight.
    feature_mean = []
    feature_std = []

    # Also store where the object IDs were located, to help redistribute data later.
    object_ids = []

    # Load each file and get the data information. Python should garbage-collect as needed.
    print("Now gathering the means and standard deviations for the original files.")
    for k in range(0, num_files):
        padded_digit = '{0:02d}'.format(k)
        data = np.load("grasp_data_" + padded_digit + ".npy")
        feature_mean.append(np.mean(data, axis=1))
        feature_std.append(np.std(data, axis=1))
        object_ids.append(data[0])
    assert np.array(feature_mean).shape == np.array(feature_std).shape
    print("Done with data gathering for feature means and standard deviations.")

    # Get (approximate) mean and weighted std for each feature, averaged across columns (axis=0).
    full_mean = np.mean(np.array(feature_mean), axis=0)
    full_std = np.mean(np.array(feature_std), axis=0)

    # Have to rehspae to get broadcasting to work
    full_mean = np.reshape(full_mean, (len(full_mean),1))
    full_std = np.reshape(full_std, (len(full_std),1))

    # Go through each file *again* to change the data for each matrix, and store it in new files.
    # To be specific, *after* we normalize the full matrix (to get normalized_data), we will have to
    # split it up into N groups, where N = num_groups. But we have to divide according to object ID.
    for k in range(0, num_files):
        padded_digit = '{0:02d}'.format(k)
        data = np.load("grasp_data_" + padded_digit + ".npy")
        normalized_data = (data - full_mean) / full_std

        # TODO need to fix this to split into five groups!
        np.save("grasp_data_normalized_" + padded_digit, normalized_data)

    print("Done with standardizing of each of the " + str(num_files) + " data files.")

    print("\nAll Done! Whew!")


def do_part_three(num_output = 10):
    """ 
    Part 3, where we take all the resulting shuffled files and combine them together. See the
    comments from the part two method, which actually contains some of this stuff.

    Then, *after* this combination, re-shuffle *again* just to be safe. Then we're done.
    """

    print("\nAll Done! Whew!")


if __name__ == '__main__':
    # Do NOT do both parts. It's either the first one, or the second one!
    #do_part_one()
    do_part_two(num_features = 1595, num_groups = 5)
    #do_part_three(num_output = 10)

