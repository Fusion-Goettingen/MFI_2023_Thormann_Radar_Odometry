import numpy as np

from simulator.abstract_simulator import AbstractSimulator
from utils import rot_mat
from array_indices import state_index


class Simulator2D(AbstractSimulator):
    def __init__(self, rng: np.random.Generator, scenario_parameters):
        self._rng = rng

        self._prior = scenario_parameters.get('prior')
        self._prior_covariance = scenario_parameters.get('prior_covariance')
        self._state = self._rng.multivariate_normal(self._prior, self._prior_covariance)

        self._trajectory = []
        self._trajectory.append(self._state[[state_index['m1'], state_index['m2'], state_index['angle']]])

        self._center_of_rotation_shift = scenario_parameters.get('center_of_rotation_shift_local')
        self._velocity_standard_deviation = scenario_parameters.get('velocity_standard_deviation')
        self._yaw_rate_standard_deviation = scenario_parameters.get('yaw_rate_standard_deviation')

        self._expected_number_of_clutter = scenario_parameters.get('expected_number_of_clutter')
        self._clutter_doppler_noise = scenario_parameters.get('clutter_doppler_standard_deviation')

        self._area = scenario_parameters.get('estimate_area')
        self._obstacles = self.__create_obstacles()

        self._expected_number_of_measurements = scenario_parameters.get('expected_number_of_measurements')
        self._standard_deviation_theta_r_doppler = np.array([
            scenario_parameters.get('theta_standard_deviation'),
            scenario_parameters.get('range_standard_deviation'),
            scenario_parameters.get('doppler_standard_deviation'),
        ])

    def get_state(self):
        return self._state

    def reset(self):
        self._state = self._rng.multivariate_normal(self._prior, self._prior_covariance)
        self._obstacles = self.__create_obstacles()
        self._trajectory = []

    def propagate(self, time_difference):
        error_mat = self.__calculate_error_matrix(time_difference)
        error = self._rng.multivariate_normal(np.zeros(2), np.diag([self._velocity_standard_deviation,
                                                                    self._yaw_rate_standard_deviation])**2)
        self._state = np.array([
            self._state[state_index['m1']] + 2.0 * self._state[state_index['velocity']] / self._state[state_index['omega']] * np.sin(0.5*self._state[state_index['omega']]*time_difference)
            * np.cos(self._state[state_index['angle']] + 0.5*self._state[state_index['omega']]*time_difference),
            self._state[state_index['m2']] + 2.0 * self._state[state_index['velocity']] / self._state[state_index['omega']] * np.sin(0.5*self._state[state_index['omega']]*time_difference)
            * np.sin(self._state[state_index['angle']] + 0.5*self._state[state_index['omega']]*time_difference),
            self._state[state_index['velocity']],
            self._state[state_index['angle']] + time_difference*self._state[state_index['omega']],
            self._state[state_index['omega']]
        ]) + (error_mat @ error)

        self._trajectory.append(self._state[[state_index['m1'], state_index['m2'], state_index['angle']]])

    def generate_data(self):
        sensor_velocity_local = (np.array([self._state[state_index['velocity']], 0.0])
                                 + self._state[state_index['omega']] * rot_mat(0.5*np.pi)
                                 @ self._center_of_rotation_shift)

        detection_points_theta_r_doppler_intensity = self.__calculate_detection_points(sensor_velocity_local)
        clutter_points_theta_r_doppler_intensity = self.__calculate_clutter_points()
        detection_points_theta_r_doppler_intensity = np.vstack([detection_points_theta_r_doppler_intensity,
                                                                clutter_points_theta_r_doppler_intensity])

        return detection_points_theta_r_doppler_intensity

    def __create_obstacles(self):
        return np.array([self._rng.uniform(self._area[0], self._area[1], 4),
                         self._rng.uniform(self._area[2], self._area[3], 4)]).T

    def __calculate_error_matrix(self, time_difference):
        return np.array([
            [0.5*time_difference**2 * np.cos(self._state[state_index['angle']]), 0.0],
            [0.5*time_difference**2 * np.sin(self._state[state_index['angle']]), 0.0],
            [time_difference, 0.0],
            [0.0, 0.5*time_difference**2],
            [0.0, time_difference]
        ])

    def __calculate_obstacle_detections_and_velocity(self, obstacle_index, sensor_velocity_local):
        obstacle_local_x_y = (rot_mat(self._state[state_index['angle']]).T
                              @ (self._obstacles[obstacle_index] - self._state[[state_index['m1'], state_index['m2']]]
                                 - rot_mat(self._state[state_index['angle']]) @ self._center_of_rotation_shift))
        obstacle_polar_theta_r = np.array([np.arctan2(obstacle_local_x_y[1], obstacle_local_x_y[0]),
                                           np.linalg.norm(obstacle_local_x_y)])
        number_of_detections = np.maximum(1, self._rng.poisson(self._expected_number_of_measurements))
        obstacle_detections_theta_r = self._rng.multivariate_normal(obstacle_polar_theta_r,
                                                                    np.diag(self._standard_deviation_theta_r_doppler[:2])**2,
                                                                    number_of_detections)
        obstacle_velocity = np.array([np.cos(obstacle_polar_theta_r[0]),
                                      np.sin(obstacle_polar_theta_r[0])]).T @ -sensor_velocity_local
        obstacle_velocity_detections = self._rng.normal(obstacle_velocity,
                                                        self._standard_deviation_theta_r_doppler[2]**2,
                                                        len(obstacle_detections_theta_r))
        return obstacle_detections_theta_r, obstacle_velocity_detections

    def __calculate_detection_points(self, sensor_velocity_local):
        detection_points_theta_r_doppler_intensity = np.zeros((0, 4))
        for i in range(len(self._obstacles)):
            obstacle_detections_theta_r, obstacle_velocity_detections \
                = self.__calculate_obstacle_detections_and_velocity(i, sensor_velocity_local)
            if len(obstacle_velocity_detections) > 0:
                detection_points_theta_r_doppler_intensity = np.vstack([detection_points_theta_r_doppler_intensity,
                                                                        np.vstack([obstacle_detections_theta_r[:, 0][None, :],
                                                                                   obstacle_detections_theta_r[:, 1][None, :],
                                                                                   obstacle_velocity_detections[None, :],
                                                                                   np.ones((1, len(obstacle_detections_theta_r)))]).T
                                                                        ])
        return detection_points_theta_r_doppler_intensity

    def __calculate_clutter_points(self):
        number_of_clutter_points = self._rng.poisson(self._expected_number_of_clutter) \
            if self._expected_number_of_clutter > 0 else 0
        if number_of_clutter_points > 0:
            clutter_cartesian = np.zeros((number_of_clutter_points, 2))

            clutter_cartesian[:, 0] = self._rng.uniform(self._area[0], self._area[1], number_of_clutter_points)
            clutter_cartesian[:, 1] = self._rng.uniform(self._area[2], self._area[3], number_of_clutter_points)
            clutter_velocity = self._rng.multivariate_normal(np.zeros(number_of_clutter_points),
                                                             np.diag(np.ones(number_of_clutter_points)
                                                                     * self._clutter_doppler_noise ** 2))

            clutter_points_theta_r_doppler_intensity = np.vstack([np.arctan2(clutter_cartesian[:, 1], clutter_cartesian[:, 0]),
                                                                  np.linalg.norm(clutter_cartesian, axis=1),
                                                                  clutter_velocity[None, :],
                                                                  np.ones((1, number_of_clutter_points))
                                                                  ]).T
        else:
            clutter_points_theta_r_doppler_intensity = np.zeros((0, 4))
        return clutter_points_theta_r_doppler_intensity

    def plot_trajectory(self, arrow_length=0.5):
        import matplotlib.pyplot as plt
        state_plot_intensities = np.linspace(0, 1.0, len(self._trajectory))
        for idx, state in enumerate(self._trajectory):
            plt.arrow(state[0], state[1], arrow_length*np.cos(state[2]), arrow_length*np.sin(state[2]),
                      alpha=state_plot_intensities[idx], head_width=0.05, color='black')
        plt.xlabel('x / m')
        plt.ylabel('y / m')
        plt.show()
