import radar_data_processor
import numpy as np

# measurements
A = 0  # azimuth
R = 1  # range
X = 0
Y = 1
D = 2 # Doppler
I = 3  # intensity

XY_RES = [100, 100]

PLOT_VELOCITY_AS_ARROW = True

# for removing static environment
ALPHA = 0.0#8

G = 9.8

INTENSITY_CUTOFF = 9
CENTER_OF_ROTATION_SHIFT = [0.2, 0.0, 0.08]  # sensor shift from center of rotation in local y-direction (assuming the sensor points in y-direction)
SIGMA_Q = [0.1, 0.1, 0.00003]
SIGMA_A = [0.1, 0.002, 0.05]
SIGMA_W = [0.01, 0.05, 0.05]
SIGMA_GYRO = [0.001, 0.001, 0.001]
SIGMA_ACC = [0.001, 0.001, 0.001]
SIGMA_GYRO_BIAS = [0.00001, 0.00001, 0.00001]
SIGMA_ACC_BIAS = [0.00001, 0.00001, 0.00001]
SIGMA_DOPPLER = 0.01#0.01#  # numerical instabilities can occur in yaw rate tracker if this is too low
SIGMA_THETA = 0.05*np.pi#0.01*np.pi#
SIGMA_CLUTTER_DOPPLER = 1.0
CFAR_WINDOW = 64
CFAR_WINDOW_R = 64  # window in range direction
CFAR_SCALING = 8#15
MIN_RANGE_M = 0.5  # ignore some points close to sensor due to overlap

# RANSAC
R_EPSILON = 0.5
R_INLIERS = 0.5  # due to noise, higher lower expected inliers
R_SECURITY = 0.99
