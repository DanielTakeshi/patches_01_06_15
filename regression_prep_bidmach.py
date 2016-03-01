"""
(c) January 2016 by Daniel Seita

This consists of three parts. Call the functions (in the main method) based on the stage I'm in.

The first part will go through the 21 data files and, for each one, create a BIDMach-like matrix,
stored as a simple npy file using np.save(...), along with all the targets (the targets are in their
own matrix). Note that we will be using the object IDs as the "first feature" so the first row in
all 21 matrices should contain the object IDs, and also, each file will have 501 objects.

The second (and third) parts will use these 21 data files and reorganize the data to make it more
suitable for BIDMach and regression. In particular, it will: (1) create a smaller number of data
matrices (I'm thinking 10), though obviously we have to make this stuff coincide with all the
targets, (2) SHUFFLE the data by object IDs so that no object ID has its data points dispersed
across multiple data matrices, (3) delete any feature that is all zero, if any [EDIT this is not an
issue, ignore ...], and (4) normalize the data, so zero-mean plus variance. This will require going
through each file individually to collect the statistics (weighted by the number of data points),
and then going through each one AGAIN to actually perform the standardization.

UPDATE: Huh, apparently some object IDs are missing. Expect to see <= 501 instead of 501 exactly.
"""

import numpy as np
import os
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

        # Now getting this into a (1,N) shape rather than (N,) (i.e., nothing after the comma).
        N = len(target_pfc_005) 
        target_pfc_005 = np.reshape(target_pfc_005, (1,N))
        target_pfc_010 = np.reshape(target_pfc_010, (1,N))
        target_pfc_020 = np.reshape(target_pfc_020, (1,N))
        target_vfc_005 = np.reshape(target_vfc_005, (1,N))
        target_vfc_010 = np.reshape(target_vfc_010, (1,N))
        target_vfc_020 = np.reshape(target_vfc_020, (1,N))
        target = np.concatenate((target_pfc_005,
                                 target_pfc_010,
                                 target_pfc_020,
                                 target_vfc_005,
                                 target_vfc_010,
                                 target_vfc_020),
                                axis=0) # I.e., concatenate the default way.
        assert not np.isnan(np.sum(target)), "NaN with the target values."
        print("Done with loading raw data, AND the targets (shape of target: {}).".format(target.shape))

        # Now concatenate the data into one matrix. We need N to be the number of elements (i.e., columns)
        data_matrix = concatenate_data(N, data_features)
        print("Shape of data matrix: {}.".format(data_matrix.shape))
        
        # Detect if any rows are zero (effectively, 1e-7, btw). That way, we should delete them later.
        # UPDATE: This is no longer an issue, but I have the framework in case I need to change it.
        epsilon = 1e-7 # I.e., 0.0000001 (this means 7 decimal places, including that final "1")
        print("Here are the features that are all zero ({}) and thus we should later remove:".format(epsilon))
        boolean_data_matrix = data_matrix > epsilon
        print(np.where(~boolean_data_matrix.any(axis=1))[0])

        # Now just save using np.save(...) to make reloading this data faster.
        print("Now saving the data_matrix, with shape: {}...".format(data_matrix.shape))
        np.save("grasp_data_" + padded_digit, data_matrix)
        np.save("grasp_target_" + padded_digit, target)

    print("\nAll Done! Whew!")


def do_part_two(num_features = 1595, num_groups = 5):
    """ 
    Now proceed to the second part. With all of the data files defined, we have to re-shuffle.
    Fortunately, Python's garbage collection can handle more than 16 GB of total RAM.

    We go through each of the 21 normalized files. Split up the 501 object IDs into, say, 5 groups
    (that's the default). If object IDs for group 0 are [0,0,1,1,1,2,...,499,500,500,500], i.e., the
    first two were from object ID 0, etc. These groups should be distinct, i.e., one group might be
    [4,6,8,10,...], another might be [0,12,204,...], etc., but they should be roughly equal in size.

    Then, for each group, we copy the grasps that correspond to those object IDs (i.e., columns --
    though we may want to transpose the matrix to make it easier to refer to it by rows) and save
    them into arrays. What this means is, after each of the 21 files has been processed, we have
    21*num_groups total data files, all with 1594 rows (we're *NOT* including the object IDs after
    this point, because with normalization it's not really a "good" feature).

    Then the third procedure (not this method) will randomly pick some ordering to combine these
    21*num_groups files into a smaller set of, say, 10 files. This is not true randomization, more
    like "block randomization" but it's the best I can do on short notice. Basically, if the last of
    the 21 original files is from a different dataset, then it gets split into groups, and then
    those get mixed in with other data files, so (ideally) those grasps from that special dataset
    are distributed among those 10 files somehow. To increase the randomness, I can create finer and
    finer divisons, e.g., using 15 groups instead of 10 groups for each of the 21 files.

    Remember that all of this MUST get coordinated with the target matrix; use same indexing trick.

    :param: num_features The number of features. I should know this in advance. The default is 1595
        since I'm ignoring the three from "center of mass."
    :param: num_groups For each of the 21 original files, we split it up into 'num_groups' groups.
    """
    num_files = 21

    # Store mean and std here as we sift through data. We'll keep appending them as extra rows, so
    # these will be (21 x num_features) arrays. This is approximate as we'll assume equal weight.
    feature_mean = []
    feature_std = []

    # Load each file and get the data information. Python should garbage-collect as needed.
    print("Now gathering the means and standard deviations for the original files.")
    for k in range(0, num_files):
        padded_digit = '{0:02d}'.format(k)
        print("Currently on " + padded_digit)
        data = np.load("grasp_data_" + padded_digit + ".npy")
        feature_mean.append(np.mean(data, axis=1))
        feature_std.append(np.std(data, axis=1))
    assert np.array(feature_mean).shape == np.array(feature_std).shape
    print("Done with data gathering for feature means and standard deviations.")

    # Get (approximate) mean and weighted std for each feature, averaged across columns (axis=0).
    full_mean = np.mean(np.array(feature_mean), axis=0)
    full_std = np.mean(np.array(feature_std), axis=0)

    # Have to reshape to get broadcasting to work, and ignore the first element.
    full_mean = np.reshape(full_mean[1:], (len(full_mean)-1,1))
    full_std = np.reshape(full_std[1:], (len(full_std)-1,1))

    # Go through each file *again* to change the data for each matrix, and store it in new files.
    for k in range(0, num_files):
        padded_digit = '{0:02d}'.format(k)
        old_data = np.load("grasp_data_" + padded_digit + ".npy")
        object_ID_info = old_data[0] # Get all the object IDs present in this matrix (w/duplicates).
        old_data = old_data.T # Transpose so old_data[i] is the i-th grasp instead of i-th feature.

        # New, don't forget the target! Again, transpose so it's (num_elements x 6)-dimensional.
        target = np.load("grasp_target_" + padded_digit + ".npy").T

        # Randomize the (unique) object IDs for this particular file and split into groups.
        objectIDs_rand = np.random.permutation( np.unique(object_ID_info) )
        assert len(objectIDs_rand) <= 501, "Error, len(objectIDs_rand) = {}".format(len(objectIDs_rand))
        objectIDs_groups = np.array_split(objectIDs_rand, num_groups)

        # For each group of random object IDs, extract elements with that ID, and save the file.
        for (index,list_of_IDs) in enumerate(objectIDs_groups):
            padded_index = '{0:02d}'.format(index)
            group_indices = []

            # Iterate through to get the *indices*, NOT the data.
            for (index2,val) in enumerate(object_ID_info):
                if val in list_of_IDs:
                    group_indices.append(index2)
            assert len(np.unique(np.array(group_indices))) == len(group_indices)

            # With the indices in hand, extract data from old data, dump ID, and transpose. We can
            # *also* do it the same way with the target data!
            group_indices = np.array(group_indices)
            partitioned_data = (old_data[group_indices].T)[1:]
            assert partitioned_data.shape[0] == num_features-1, "partitioned_data.shape[0] = {}".format(partitioned_data.shape[0])
            partitioned_target = target[group_indices].T
            assert partitioned_target.shape[0] == 6, "partitioned_target.shape[0] = {}".format(partitioned_target.shape[0])
            assert partitioned_target.shape[1] == partitioned_data.shape[1]

            # Let's normalize that partitioned data *transposed*. And also do the target!
            normalized_data = ((partitioned_data - full_mean) / full_std)
            np.save("grasp_data_norm_" + padded_digit + "_" + padded_index, normalized_data)
            np.save("grasp_target_norm_" + padded_digit + "_" + padded_index, partitioned_target)

        print("Done with " + padded_digit)

    print("\nAll Done! Whew!")


def do_part_three(num_features = 1595, num_output = 10):
    """ 
    Part 3, where we take all the resulting shuffled files and combine them together. See the
    comments from the part two method, which actually contains some of this stuff.

    Then, *after* this combination, re-shuffle *again* just to be safe. Then we're done.
    """

    # Make sure this is the correct naming convention, i.e., "grasp_{data,target}_norm". These
    # should already be sorted (they HAVE to be aligned) but just to be safe add "sorted".
    normalized_data_files   = np.array(sorted([x for x in os.listdir(".") if "grasp_data_norm" in x]))
    normalized_target_files = np.array(sorted([x for x in os.listdir(".") if "grasp_target_norm" in x]))
    assert len(normalized_data_files) == len(normalized_target_files)

    # Now determine the splits using the indices of files, so we can extract data and targets.
    indices = np.random.permutation( np.arange(len(normalized_data_files)) )
    indices_splits = np.array_split(indices, num_output)

    # For each output index, we take the list of files to load and (horizontally) combine them.
    for k in range(num_output):
        padded_digit = '{0:02d}'.format(k)
        indices_in_list = np.array(indices_splits[k])
        data_files   = normalized_data_files[indices_in_list]
        target_files = normalized_target_files[indices_in_list]
        
        # Serve as the first "baseline" as usual, then build it using concatenation.
        result_data   = np.load(data_files[0])
        result_target = np.load(target_files[0])

        for i in range(1,len(indices_in_list)):
            result_data   = np.concatenate((result_data,   np.load(data_files[i])),   axis=1)
            result_target = np.concatenate((result_target, np.load(target_files[i])), axis=1)
        
        # At last, save the combined data as text files. And the target. (We'll split inside BIDMach)
        print("result_data.shape = {}".format(result_data.shape))
        print("result_target.shape = {}".format(result_target.shape))

        # Use six digits of decimal precision. Note that data is already normalized.
        np.savetxt("grasp_bidmach_data_" + padded_digit + ".txt",   result_data,   fmt='%1.6f')
        np.savetxt("grasp_bidmach_target_" + padded_digit + ".txt", result_target, fmt='%1.6f')
        print("Finished with " + padded_digit)

    print("\nAll Done! Whew!")


if __name__ == '__main__':
    # Only do ONE of the parts at any given time. Actually, not strictly necessary but helps me to
    # keep track of whether the code is working correctly. Also, I need to delete files to save on
    # hard disk space. Yeah.
    K = 1595 # Includes the object ID
    do_part_one()
    #do_part_two(num_features = K, num_groups = 10)
    #do_part_three(num_features = K, num_output = 10)

