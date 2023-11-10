import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow

import constants


def plot_2d_points_doppler(points, max_range_m, max_angle_degrees, fig, ax, step, ego_state=None,
                           progressed_time=None, extent=None):
    # points are the 3D Cartesian output of radar processing
    # ego_state has the shape [m1, m2, m3, v1, v2, v3, q0, q1, q2, q3, w1, w2, w3], will be written in the title
    # axs needs to contain at least 2 axis, the first two of which are updated with the current points

    minmin = 8
    maxmax = 15

    ax.clear()

    # xy-plane
    if extent is None:
        extent = [0,
                  max_range_m,
                  -np.sin(max_angle_degrees)*max_range_m,
                  np.sin(max_angle_degrees)*max_range_m]
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("auto")
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])

    ax.scatter(points[:, constants.X], points[:, constants.Y], c=points[:, constants.I], cmap='viridis_r')#, vmin=minmin, vmax=maxmax)

    points_d = points[abs(points[:, constants.D]) != 0.0]
    cmap = plt.cm.viridis_r
    color_inds = np.array(np.maximum(np.minimum((points_d[:, constants.I] - minmin) / maxmax * 255, 255), 0), int)
    for i in range(len(points_d)):
        ax.add_patch(Arrow(points_d[i, constants.X], points_d[i, constants.Y],
                           points_d[i, constants.X] * points_d[i, constants.D]
                           / np.linalg.norm(points_d[i, [constants.X, constants.Y]]),
                           points_d[i, constants.Y] * points_d[i, constants.D]
                           / np.linalg.norm(points_d[i, [constants.X, constants.Y]]),
                           color=cmap.colors[color_inds[i]], width=0.1))

    title = f'Step: {step:03d}'
    if progressed_time is not None:
        title += f' Time: {progressed_time:.3f}'
    elif ego_state is not None:
        title += f' x:{ego_state[0]:1.3f}, y:{ego_state[1]:1.3f}, vx:{ego_state[2]:1.3f}, ' \
                 f'vy:{ego_state[3]:1.3f},, yaw:{ego_state[4]:1.3f}, yawrate:{ego_state[5]:1.3f}'
    ax.set_title(title)

    plt.draw()
    plt.savefig(f'./figures/frame{step:03d}.png')
    plt.pause(1e-3)


def plot_ego_pose_2d(ego_state, max_range_m, max_angle_degrees, fig, ax_pose, angle_index, state_color=['blue'],
                     ground_truth=None, extent=None):
    ax_pose.clear()
    if extent is None:
        ax_pose.set_xlim([0, max_range_m])
        ax_pose.set_ylim([-np.sin(max_angle_degrees)*max_range_m,
                          np.sin(max_angle_degrees)*max_range_m])
    else:
        ax_pose.set_xlim(extent[:2])
        ax_pose.set_ylim(extent[2:])

    if len(ego_state.shape) > 1:
        # multiple states
        assert len(ego_state) == len(state_color)
    else:
        ego_state = np.atleast_2d(ego_state)
        state_color = np.atleast_1d(state_color)

    for i in range(len(ego_state)):
        pose_vector = np.array([np.cos(ego_state[i, angle_index]), np.sin(ego_state[i, angle_index])])
        ax_pose.quiver(ego_state[i, constants.M1], ego_state[i, constants.M2], pose_vector[0], pose_vector[1],
                       color=state_color[i])
    if ground_truth is not None:
        pose_vector_gt = np.array([np.cos(ground_truth[angle_index]), np.sin(ground_truth[angle_index])])
        ax_pose.quiver(ground_truth[constants.M1], ground_truth[constants.M2], pose_vector_gt[0], pose_vector_gt[1],
                       color='black', alpha=0.5)
