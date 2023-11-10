import numpy as np

from simulator.abstract_simulator import AbstractSimulator
from utils import rot_mat
import constants


# position
M1 = 0
M2 = 1
# velocity
V1 = 2
V2 = 3
# orientation and yaw rate
AL = 4
OM = 5


class Simulator2D(AbstractSimulator):
    def __init__(self, rng: np.random.Generator, polar=False, lam_meas=5, lam_clutter=0, area=[-3.0, 3.0, -3.0, 3.0], use_ct=False):
        self._rng = rng

        self._state = self._rng.multivariate_normal(constants.PRIOR, np.diag(constants.PRIOR_COV)**2)

        self._trajectory = []
        self._trajectory.append(self._state[[M1, M2, AL]])

        self._center_of_rotation_shift = np.array(constants.CENTER_OF_ROTATION_SHIFT[:2])
        self._sigma_a = constants.SIGMA_A[:2]  # acceleration noise on velocity
        self._sigma_w = constants.SIGMA_W[0]  # noise on yaw rate
        self._use_coordinated_turn = use_ct

        self._lam_clutter = lam_clutter
        self._area = area

        self._obstacles = np.array([
            [2.2, 0.0],
            [1.5, 1.2],
            [1.5, -1.2]
        ])  # azimuth, range
        self._obstacles = np.array([self._rng.uniform(self._area[0], self._area[1], 4),
                                    self._rng.uniform(self._area[2], self._area[3], 4)]).T
        self._lam_n_points = lam_meas
        self._detection_noise = np.array([constants.SIGMA_THETA, 0.05, constants.SIGMA_DOPPLER])
        self._clutter_noise = constants.SIGMA_CLUTTER_DOPPLER

        self._polar = polar  # ignore v2 and move in heading direction

    def get_state(self):
        return self._state[[M1, M2, V1, AL, OM]] if self._polar else self._state

    def reset(self):
        self._state = self._rng.multivariate_normal(constants.PRIOR, np.diag(constants.PRIOR_COV)**2)
        self._obstacles = np.array([self._rng.uniform(self._area[0], self._area[1], 4),
                                    self._rng.uniform(self._area[2], self._area[3], 4)]).T
        self._trajectory = []

    def propagate(self, time_difference, step_id=None):
        if self._use_coordinated_turn:
            error_mat = np.array([
                [0.5*time_difference**2*np.cos(self._state[AL]), 0.0],
                [0.5*time_difference**2*np.sin(self._state[AL]), 0.0],
                [time_difference, 0.0],
                [0.0, 0.0],
                [0.0, 0.5*time_difference**2],
                [0.0, time_difference]
            ])
            error = np.zeros(2)
            if step_id is not None:
                error[0] = 0.025 if int(step_id % 20) in [0, 1] else 0.025 if int(step_id % 20) in [10, 11] else 0.0
                error[1] = 0.025*np.pi if int(step_id % 20) in [0, 1] else -0.05*np.pi if int(step_id % 20) in [10, 11] \
                    else 0.0
            error = self._rng.multivariate_normal(np.zeros(2), np.diag([self._sigma_a[0], self._sigma_w])**2)

            self._state[[M1, M2, V1, AL, OM]] = np.array([
                self._state[M1] + 2.0 * self._state[V1] / self._state[OM] * np.sin(0.5*self._state[OM]*time_difference)
                    * np.cos(self._state[AL] + 0.5*self._state[OM]*time_difference),
                self._state[M2] + 2.0 * self._state[V1] / self._state[OM] * np.sin(0.5*self._state[OM]*time_difference)
                    * np.sin(self._state[AL] + 0.5*self._state[OM]*time_difference),
                self._state[V1],
                self._state[AL] + time_difference*self._state[OM],
                self._state[OM]
            ]) + (error_mat @ error)[[M1, M2, V1, AL, OM]]
        else:
            if self._polar:
                tran_mat = np.eye(len(self._state))
                tran_mat[M1, V1] = time_difference * np.cos(self._state[AL])
                tran_mat[M2, V1] = time_difference * np.sin(self._state[AL])
                tran_mat[AL, OM] = time_difference

                error_mat = np.array([
                    [0.5*time_difference**2*np.cos(self._state[AL]), 0.0],
                    [0.5*time_difference**2*np.sin(self._state[AL]), 0.0],
                    [time_difference, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.5*time_difference**2],
                    [0.0, time_difference]
                ])
                error = np.zeros(2)
                if step_id is not None:
                    error[0] = 0.025 if int(step_id % 20) in [0, 1] else 0.025 if int(step_id % 20) in [10, 11] else 0.0
                    error[1] = 0.025*np.pi if int(step_id % 20) in [0, 1] else -0.05*np.pi if int(step_id % 20) in [10, 11] \
                        else 0.0
                error = self._rng.multivariate_normal(np.zeros(2), np.diag([self._sigma_a[0], self._sigma_w])**2)
            else:
                tran_mat = np.eye(len(self._state))
                tran_mat[M1, V1] = time_difference
                tran_mat[M2, V2] = time_difference
                tran_mat[AL, OM] = time_difference

                error_mat = np.array([
                    [0.5*time_difference**2, 0.0, 0.0],
                    [0.0, 0.5*time_difference**2, 0.0],
                    [time_difference, 0.0, 0.0],
                    [0.0, time_difference, 0.0],
                    [0.0, 0.0, 0.5*time_difference**2],
                    [0.0, 0.0, time_difference]
                ])
                error = np.zeros(3)
                if step_id is not None:
                    error[0] = 0.1 if int(step_id % 20) in [0, 1] else -0.1 if int(step_id % 20) in [10, 11] else 0.0
                    error[2] = 0.05*np.pi if int(step_id % 20) in [0, 1] else -0.05*np.pi if int(step_id % 20) in [10, 11] \
                        else 0.0
                error = self._rng.multivariate_normal(np.zeros(3), np.diag([self._sigma_a[0], self._sigma_a[0], self._sigma_w])**2)

            self._state = tran_mat @ self._state + error_mat @ error

        self._trajectory.append(self._state[[M1, M2, AL]])

    def plot_trajectory(self, arrow_length=0.5):
        import matplotlib.pyplot as plt
        state_plot_intensities = np.linspace(0, 1.0, len(self._trajectory))
        for idx, state in enumerate(self._trajectory):
            plt.arrow(state[0], state[1], arrow_length*np.cos(state[2]), arrow_length*np.sin(state[2]),
                      alpha=state_plot_intensities[idx], head_width=0.05, color='black')
        plt.xlabel('x / m')
        plt.ylabel('y / m')
        plt.show()

    def generate_data(self):
        # determine sensor velocity
        if self._polar:
            vel_local = np.array([self._state[V1], 0.0])
        else:
            vel_local = rot_mat(self._state[AL]).T @ self._state[[V1, V2]]
        ego_velocity_local = vel_local + self._state[OM] * rot_mat(0.5*np.pi) @ self._center_of_rotation_shift

        # calculate points
        points = np.zeros((0, 5))
        for i in range(len(self._obstacles)):
            obstacle_local = rot_mat(self._state[AL]).T @ (self._obstacles[i] - self._state[[M1, M2]]
                                                           - rot_mat(self._state[AL]) @ self._center_of_rotation_shift)
            obstacle_polar = np.array([np.arctan2(obstacle_local[1], obstacle_local[0]),
                                       np.linalg.norm(obstacle_local)])
            obstacle_detections = self._rng.multivariate_normal(obstacle_polar, np.diag(self._detection_noise[:2])**2,
                                                                np.maximum(1, self._rng.poisson(self._lam_n_points)))
            obstacle_velocity = np.array([np.cos(obstacle_polar[0]),
                                         np.sin(obstacle_polar[0])]).T @ -ego_velocity_local
            obstacle_velocity_detections = self._rng.normal(obstacle_velocity, self._detection_noise[2],
                                                            len(obstacle_detections))
            if len(obstacle_velocity_detections) > 0:
                points = np.vstack([points,
                                    np.vstack([obstacle_detections[:, 0][None, :],
                                               np.zeros((1, len(obstacle_detections))),  # no elevation
                                               obstacle_detections[:, 1][None, :],
                                               obstacle_velocity_detections[None, :],
                                               np.ones((1, len(obstacle_detections)))]).T
                                    ])
        if self._lam_clutter > 0:
            n_clutter = self._rng.poisson(self._lam_clutter)
            if n_clutter > 0:
                clutter_cartesian = np.zeros((n_clutter, 2))

                clutter_cartesian[:, 0] = self._rng.uniform(self._area[0], self._area[1], n_clutter)
                clutter_cartesian[:, 1] = self._rng.uniform(self._area[2], self._area[3], n_clutter)

                points = np.vstack([points,
                                    np.vstack([np.arctan2(clutter_cartesian[:, 1], clutter_cartesian[:, 0]),
                                               np.zeros((1, n_clutter)),
                                               np.linalg.norm(clutter_cartesian, axis=1),
                                               self._rng.multivariate_normal(np.zeros(n_clutter),
                                                                             np.diag(np.ones(n_clutter)
                                                                                     * self._clutter_noise**2))[None, :],
                                               np.ones((1, n_clutter))]).T
                                    ])
        else:
            n_clutter = 0

        # import matplotlib.pyplot as plt
        # plt.xlim(-0.5*np.pi, 0.5*np.pi)
        # angle_bins = np.linspace(-0.5*np.pi, 0.5*np.pi, 100)
        # plt.plot(angle_bins, np.array([np.cos(angle_bins), np.sin(angle_bins)]).T @ vel_local, color='blue')
        # plt.scatter(points[:(len(points)-n_clutter), 0], -points[:(len(points)-n_clutter), 3], color='blue')
        # plt.scatter(points[(len(points)-n_clutter):, 0], -points[(len(points)-n_clutter):, 3], color='red')
        # plt.show()

        return points, n_clutter
