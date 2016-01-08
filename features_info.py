'''
(c) January 2016 by Daniel Seita

This will look at the distribution of features to see if there is anything interesting.
'''

from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import regression_v01

target_data = np.load('efc.npy')

axis = np.load('c_axis.npy')
center = np.load('c_center.npy')
grasp_axis_gravity_angle = np.load('c_grasp_axis_gravity_angle.npy')
moment_arm_mag = np.load('c_moment_arm_mag.npy')
moment_arms_gravity_angle = np.load('c_moment_arms_gravity_angle.npy')

w1_curvature_window = np.load('c_w1_curvature_window.npy')
w1_gradx_window = np.load('c_w1_gradx_window.npy')
w1_grady_window = np.load('c_w1_grady_window.npy')
w1_projection_disc = np.load('c_w1_projection_disc.npy')
w1_projection_window = np.load('c_w1_projection_window.npy')

w2_curvature_window = np.load('c_w2_curvature_window.npy')
w2_gradx_window = np.load('c_w2_gradx_window.npy')
w2_grady_window = np.load('c_w2_grady_window.npy')
w2_projection_disc = np.load('c_w2_projection_disc.npy')
w2_projection_window = np.load('c_w2_projection_window.npy')

# Re-using old code
X,y = regression_v01.concatenate_data(len(target_data), target_data,
                       axis = axis,
                       center = center,
                       grasp_axis_gravity_angle = grasp_axis_gravity_angle,
                       moment_arm_mag = moment_arm_mag,
                       moment_arms_gravity_angle = moment_arms_gravity_angle,
                       w1_curvature_window = w1_curvature_window,
                       w1_gradx_window = w1_gradx_window,
                       w1_grady_window = w1_grady_window,
                       w1_projection_disc = w1_projection_disc,
                       w1_projection_window = w1_projection_window,
                       w2_curvature_window = w2_curvature_window,
                       w2_gradx_window = w2_gradx_window,
                       w2_grady_window = w2_grady_window,
                       w2_projection_disc = w2_projection_disc,
                       w2_projection_window = w2_projection_window)

# Now scripting to get data. Can fit on the data itself.
scaler = preprocessing.StandardScaler().fit(X)
# scaler.mean_ is same as np.mean(X,axis=0)
# scaler.scale_ is same as np.std(X,axis=0)
assert scaler.mean_.shape == (1587,)

# Let's do the mean
plt.figure()
plt.title('Mean of Features in Data', size='xx-large')
plt.xlabel('Feature Averages', size='x-large')
plt.ylabel('Count', size='x-large')
plt.hist(scaler.mean_, bins=100, color='purple')
plt.savefig('fig_mean.png')

# Let's do the standard deviation
plt.figure()
plt.title('Standard Deviation of Features in Data', size='xx-large')
plt.xlabel('Feature Standard Deviations', size='x-large')
plt.ylabel('Count', size='x-large')
plt.hist(scaler.scale_, bins=100, color='yellow')
plt.savefig('fig_std.png')

# Now raw values.
plt.figure()
plt.title('All data points in X', size='xx-large')
plt.xlabel('Feature Values', size='x-large')
plt.ylabel('Count', size='x-large')
plt.hist(X.flatten(), bins=100, color='red')
plt.savefig('fig_allpoints.png')

# Some text data.
print "Lowest mean = {}, highest mean = {}".format(np.min(scaler.mean_), np.max(scaler.mean_))
print "Lowest std = {}, highest std = {}".format(np.min(scaler.scale_), np.max(scaler.scale_))
print "Lowest value = {}, highest value = {}".format(np.min(X), np.max(X))
print "Shape of X = {}, and number of NONzeros is = {}.".format(X.shape, np.count_nonzero(X))

# Analyze the scaled data if desired.
X_scaled = scaler.transform(X)
