'''
(c) January 2016 by Daniel Seita

This script will take the data files from Jeff and process them so that we get rid of the bad cases:
whenever we have "NaN"s in a component, and whenever our efc is very negative. When we see a
component with a bad info point in it, we must delete the entire thing from the data. Fortunately,
only a small fraction of these have this problem, so we should still get over 300k training data
points. After this code is run, we will have new files to use.
'''

import numpy as np

data = [
    np.load('efc_L1_f_0.100000_tg_0.005000_rg_0.100000_to_0.005000_ro_0.100000.npz')['arr_0'],
    np.load('pfc_f_0.100000_tg_0.005000_rg_0.100000_to_0.005000_ro_0.100000.npz')['arr_0'],
    np.load('axis.npz')['arr_0'],
    np.load('center.npz')['arr_0'],
    np.load('grasp_axis_gravity_angle.npz')['arr_0'],
    np.load('moment_arm_mag.npz')['arr_0'],
    np.load('moment_arms_gravity_angle.npz')['arr_0'],
    np.load('w1_curvature_window.npz')['arr_0'],
    np.load('w1_gradx_window.npz')['arr_0'],
    np.load('w1_grady_window.npz')['arr_0'],
    np.load('w1_projection_disc.npz')['arr_0'],
    np.load('w1_projection_window.npz')['arr_0'],
    np.load('w2_curvature_window.npz')['arr_0'],
    np.load('w2_gradx_window.npz')['arr_0'],
    np.load('w2_grady_window.npz')['arr_0'],
    np.load('w2_projection_disc.npz')['arr_0'],
    np.load('w2_projection_window.npz')['arr_0']
]

print "Printing shapes at the start:"
for item in data:
    print item.shape
print ""

indices = np.array([])
for item in data[:2]:
    bad_elems = np.where(item < -1e100)[0]
    if len(bad_elems) > 0:
        print "Number of bad elements = {}, indices =\n{}.".format(len(bad_elems), bad_elems)
        indices = np.concatenate( (indices, bad_elems) )

for item in data[2:]:
    if np.isnan(np.sum(item)):
        print np.unique(np.where(np.isnan(item))[0])
        indices = np.concatenate( (indices, np.unique(np.where(np.isnan(item))[0])) )

unique_indices = np.unique(indices)
print "Indices to remove: {}".format(unique_indices)

index = 0
for i in range(len(data[0])):
    if i in unique_indices:
        print "Removing index = {}.".format(i)
        for k in range(len(data)):
            data[k] = np.delete(data[k], index, axis=0)
    else:
        index += 1

print "All done with deleting. Now printing shapes and saving."
for (index,item) in enumerate(data):
    print "Shape = {} at index {}.".format(item.shape, index)

# Daniel: er, actually these shouldn't have 'npz' in them...
np.save('efc.npz', data[0])
np.save('pfc.npz', data[1])

np.save('c_axis.npz', data[2])
np.save('c_center.npz', data[3])
np.save('c_grasp_axis_gravity_angle.npz', data[4])
np.save('c_moment_arm_mag.npz', data[5])
np.save('c_moment_arms_gravity_angle.npz', data[6])

np.save('c_w1_curvature_window.npz', data[7])
np.save('c_w1_gradx_window.npz', data[8])
np.save('c_w1_grady_window.npz', data[9])
np.save('c_w1_projection_disc.npz', data[10])
np.save('c_w1_projection_window.npz', data[11])

np.save('c_w2_curvature_window.npz', data[12])
np.save('c_w2_gradx_window.npz', data[13])
np.save('c_w2_grady_window.npz', data[14])
np.save('c_w2_projection_disc.npz', data[15])
np.save('c_w2_projection_window.npz', data[16])
