"""
(c) March 2016 by Daniel Seita

This code will, for a certain index into the training data, form the 3-D plots (scatter, mesh,
whatever) of the training data.  Ideally, this will let us plot and visualize the grasps, so we can
add that as a figure to the paper.
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def main():

    # Some annoying administrative stuff.
    grasp_index = 100000
    matrix_index = 0 
    print("Analyzing grasp index {} in matrix {}.".format(grasp_index, matrix_index))
    padded_digit = '{0:02d}'.format(matrix_index)

    # Load only the data I want to plot, and only the certain grasp_index from those as well!
    center               = np.load("center/center_" + padded_digit + ".npz")['arr_0'][grasp_index]
    moment_arms          = np.load("moment_arms/moment_arms_" + padded_digit + ".npz")['arr_0'][grasp_index]
    patch_orientation    = np.load("patch_orientation/patch_orientation_" + padded_digit + ".npz")['arr_0'][grasp_index]
    surface_normals      = np.load("surface_normals/surface_normals_" + padded_digit + ".npz")['arr_0'][grasp_index]
    w1_projection_window = np.load("w1_projection_window/w1_projection_window_" + padded_digit + ".npz")['arr_0'][grasp_index]
    w2_projection_window = np.load("w2_projection_window/w2_projection_window_" + padded_digit + ".npz")['arr_0'][grasp_index]
    print("center.shape = {},\ncenter.T = {}.\n".format(center.shape, center.T))
    print("moment_arms.shape = {},\nmoment_arms.T = {}.\n".format(moment_arms.shape, moment_arms.T))
    print("patch_orientation.shape = {},\npatch_orientation.T = {}.\n".format(patch_orientation.shape, patch_orientation.T))
    print("surface_normals.shape = {},\nsurface_normals.T = {}.\n".format(surface_normals.shape, surface_normals.T))
    print("w1_projection_window.shape = {}.".format(w1_projection_window.shape))
    diff = np.absolute(w1_projection_window - w2_projection_window)
    print("Maximum difference is {}.".format(np.max(diff)))
    w1_projection_window = np.reshape(w1_projection_window, (13,13))
    w2_projection_window = np.reshape(w2_projection_window, (13,13))

    # These are the contact points, right? 
    c1 = moment_arms[0:3]
    c2 = moment_arms[3:6]
    s1 = surface_normals[0:3]
    s2 = surface_normals[3:6]

    # Now let's plot. Can use, e.g., ax.scatter(xs,ys,zs) or ax.plot_trisurf(xs,ys,zs).
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # data[0] = xs, data[1] = ys, data[2] = zs_w1 or zs_w2 depending on which one
    w1_data = [[],[],[]]
    w2_data = [[],[],[]]

    for y in range(13): # Identify a specific row (a specific y).
        for x in range(13):
            # Use -6 to get axis in terms of {-6,...,0,...,6} rather than {0,1,...,12}?
            w1_data[0].append(0.00384*(x-6))
            w2_data[0].append(0.00384*(x-6))
            w1_data[1].append(0.00384*(y-6))
            w2_data[1].append(0.00384*(y-6))
            w1_data[2].append( w1_projection_window[x][y] )
            w2_data[2].append( w2_projection_window[x][y] )

    w1_data = np.array(w1_data) # Shape (3,169)
    w2_data = np.array(w2_data) # Shape (3,169)
    patch_orientation = np.reshape(patch_orientation, (3,1))
    w1_data = w1_data + patch_orientation
    w2_data = w2_data - patch_orientation

    ax.plot_trisurf(w1_data[0], w1_data[1], w1_data[2], color='b')
    ax.plot_trisurf(w2_data[0], w2_data[1], w2_data[2], color='r')
    ax.scatter(center[0], center[1], center[2], color='k')
    ax.scatter(c1[0], c1[1], c1[2], color='b')
    ax.scatter(c2[0], c2[1], c2[2], color='r')
    #ax.scatter(s1[0], s1[1], s1[2], color='b')
    #ax.scatter(s2[0], s2[1], s2[2], color='b')
    plt.show()
    plt.savefig("testing_3d_scatter.png")


if (__name__ == "__main__"):
    main()
