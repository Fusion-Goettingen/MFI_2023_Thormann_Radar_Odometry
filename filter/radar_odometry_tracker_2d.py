import numpy as np

from filter.abstract_filter import AbstractFilter
from utils import rot_mat, invert
from array_indices import measurement_index, state_index

MODE_BASE = 0
MODE_RANSAC = 1
MODE_RANSAC_THETA = 2
MODE_GNC = 3
MODE_GNC_THETA = 4

MODE_COLORS = ['black', 'blue', 'cyan', 'red', 'orange']
MODE_LABELS = ['baseline', 'RANSAC', 'RANSAC (bearing noise)', 'GNC', 'GNC (bearing noise)']
MODE_MARKERS = ['*', '+', '+', 'x', 'x']


class RadarOdometryTracker2D(AbstractFilter):
    def __init__(self, time_stamp, scenario_parameters, mode, rng, runs, time_steps, real_data=False):
        assert mode in self.get_valid_modes()
        self._mode = mode
        self._rng = rng
        if (self._mode in [MODE_RANSAC, MODE_RANSAC_THETA]) & (self._rng is None):
            raise ValueError('RNG needs to be provided when using tracker with Mode: ' + self.get_label())

        self._use_theta_noise = (mode in [MODE_RANSAC_THETA, MODE_GNC_THETA])

        self._prior_state = scenario_parameters.get('prior')
        self._prior_cov = scenario_parameters.get('prior_covariance')
        self._initial_time_step = time_stamp

        self._last_time = self._initial_time_step
        self._state = self._prior_state.copy()
        self._cov = self._prior_cov.copy()
        self._velocity_standard_deviation = scenario_parameters.get('velocity_standard_deviation')
        self._yaw_rate_standard_deviation = scenario_parameters.get('yaw_rate_standard_deviation')

        self._center_of_rotation_shift_local = scenario_parameters.get('center_of_rotation_shift_local')
        self._doppler_standard_deviation = scenario_parameters.get('doppler_standard_deviation')
        self._theta_standard_deviation = scenario_parameters.get('theta_standard_deviation')

        self._real_data = real_data

        self._h_mat = np.zeros((0, 2))

        self._error = np.zeros((runs, time_steps, len(self._state)))

    def reset(self, rng):
        self._last_time = self._initial_time_step
        self._state = self._prior_state.copy()
        self._cov = self._prior_cov.copy()
        self._rng = rng

    def calculate_squared_error(self, true_state, run, time_step):
        error = true_state - self._state
        error[state_index['angle']] = ((error[state_index['angle']] + np.pi) % (2.0*np.pi)) - np.pi
        self._error[run, time_step] = error**2

    def get_error(self):
        return self._error

    @staticmethod
    def get_valid_modes():
        return np.array([MODE_BASE, MODE_RANSAC, MODE_RANSAC_THETA, MODE_GNC, MODE_GNC_THETA])

    def get_color(self):
        return MODE_COLORS[self._mode]

    def get_label(self):
        return MODE_LABELS[self._mode]

    def get_marker(self):
        return MODE_MARKERS[self._mode]

    def get_state(self):
        return self._state

    def get_covariance(self):
        return self._cov

    def __predict(self, time_stamp):
        time_difference = time_stamp - self._last_time
        self._last_time = time_stamp

        state_aug = np.block([self._state, np.zeros(2)])
        cov_aug = np.zeros((len(self._state)+2, len(self._state)+2))
        cov_aug[:len(self._state), :len(self._state)] = self._cov
        cov_aug[len(self._state):, len(self._state):] = np.diag([self._velocity_standard_deviation,
                                                                 self._yaw_rate_standard_deviation])**2

        x_sig = np.zeros((len(state_aug)*2 + 1, len(state_aug)))
        w_sig = np.zeros(len(x_sig))
        x_sig[0] = state_aug.copy()
        w_sig[0] = 1.0 / len(state_aug)
        cov_sig = np.linalg.cholesky(cov_aug)
        if not np.allclose(cov_sig, np.tril(cov_sig)):
            cov_sig = cov_sig.T
        if not np.allclose(cov_sig, np.tril(cov_sig)):
            print('Error in Cholesky decomposition in UKF!')
        cov_sig *= np.sqrt(len(state_aug) / (1.0 - w_sig[0]))
        # calculate sigma points
        for i in range(0, len(state_aug)):
            x_sig[i + 1, :] = state_aug + cov_sig[:, i]
            w_sig[i + 1] = (1.0 - w_sig[0]) / (2.0*len(state_aug))
            x_sig[i + len(state_aug) + 1, :] = state_aug - cov_sig[:, i]
            w_sig[i + len(state_aug) + 1] = (1.0 - w_sig[0]) / (2.0*len(state_aug))
        for i in range(len(x_sig)):
            error_mat = np.array([
                [0.5*time_difference**2*np.cos(x_sig[i, state_index['angle']]), 0.0],
                [0.5*time_difference**2*np.sin(x_sig[i, state_index['angle']]), 0.0],
                [time_difference, 0.0],
                [0.0, 0.5*time_difference**2],
                [0.0, time_difference]
            ])
            if abs(x_sig[i, state_index['omega']]) > 1e-6:
                x_sig[i, :len(self._state)] = np.array([
                    x_sig[i, state_index['m1']] + 2.0 * x_sig[i, state_index['velocity']] / x_sig[i, state_index['omega']] * np.sin(0.5*x_sig[i, state_index['omega']]*time_difference)
                    * np.cos(x_sig[i, state_index['angle']] + 0.5*x_sig[i, state_index['omega']]*time_difference),
                    x_sig[i, state_index['m2']] + 2.0 * x_sig[i, state_index['velocity']] / x_sig[i, state_index['omega']] * np.sin(0.5*x_sig[i, state_index['omega']]*time_difference)
                    * np.sin(x_sig[i, state_index['angle']] + 0.5*x_sig[i, state_index['omega']]*time_difference),
                    x_sig[i, state_index['velocity']],
                    x_sig[i, state_index['angle']] + time_difference*x_sig[i, state_index['omega']],
                    x_sig[i, state_index['omega']]
                ])
            else:
                x_sig[i, :len(self._state)] = np.array([
                    x_sig[i, state_index['m1']]
                    + time_difference*x_sig[i, state_index['velocity']]*np.cos(x_sig[i, state_index['angle']]),
                    x_sig[i, state_index['m2']]
                    + time_difference*x_sig[i, state_index['velocity']]*np.sin(x_sig[i, state_index['angle']]),
                    x_sig[i, state_index['velocity']],
                    x_sig[i, state_index['angle']] + time_difference*x_sig[i, state_index['omega']],
                    x_sig[i, state_index['omega']],
                ])
            x_sig[i, :len(self._state)] += error_mat @ x_sig[i, len(self._state):]
        self._state = np.sum(w_sig[:, None] * x_sig[:, :len(self._state)], axis=0)
        self._cov = np.einsum('x, xa, xb -> ab',
                              w_sig, x_sig[:, :len(self._state)] - self._state[None, :],
                              x_sig[:, :len(self._state)] - self._state[None, :])

    def __set_h_matrix(self, points_polar):
        self._h_mat = np.array([np.cos(points_polar[:, measurement_index['bearing']]),
                                np.sin(points_polar[:, measurement_index['bearing']])]).T

    def __no_measurement_update_possible(self, points_polar):
        if len(points_polar) < 3:
            print('Not enough points to solve LS. Skipping measurement update!')
            return True
        if all(self._h_mat[0, 0] == self._h_mat[:, 0]) & all(self._h_mat[0, 1] == self._h_mat[:, 1]):
            print('All rows of H are the same, LS not possible! Skipping measurement update!')
            return True
        return False

    def __calculate_expected_velocity(self):
        return -(np.array([self._state[state_index['velocity']], 0.0])
                 + self._state[state_index['omega']] * rot_mat(0.5 * np.pi) @ self._center_of_rotation_shift_local)

    def __calculate_angle_of_arrival_noise_parameters(self, points_polar):
        theta_cov = np.array([
            [0.5 + 0.5 * np.cos(2 * points_polar[:, measurement_index['bearing']]) * np.exp(-2 * self._theta_standard_deviation ** 2)
             + (np.cos(points_polar[:, measurement_index['bearing']])**2) * (np.exp(self._theta_standard_deviation ** 2) - 2.0),
             0.5 * np.sin(2 * points_polar[:, measurement_index['bearing']]) * np.exp(-2 * self._theta_standard_deviation ** 2)
             + (np.cos(points_polar[:, measurement_index['bearing']]) * np.sin(points_polar[:, measurement_index['bearing']])) * (np.exp(self._theta_standard_deviation ** 2) - 2.0)],
            [0.5 * np.sin(2 * points_polar[:, measurement_index['bearing']]) * np.exp(-2 * self._theta_standard_deviation ** 2)
             + (np.cos(points_polar[:, measurement_index['bearing']]) * np.sin(points_polar[:, measurement_index['bearing']])) * (np.exp(self._theta_standard_deviation ** 2) - 2.0),
             0.5 - 0.5 * np.cos(2 * points_polar[:, measurement_index['bearing']]) * np.exp(-2 * self._theta_standard_deviation ** 2)
             + (np.sin(points_polar[:, measurement_index['bearing']]) ** 2) * (np.exp(self._theta_standard_deviation ** 2) - 2.0)]
        ]).transpose((2, 0, 1))
        return theta_cov

    def __debias_h_mat(self):
        self._h_mat = self._h_mat * np.exp(0.5 * self._theta_standard_deviation**2)

    def __find_velocity_estimate_via_gnc(self, points_polar, velocity_hat, theta_cov):
        # based on Y. Zhuang, B. Wang, J. Huai, and M. Li, “4D iRIOM: 4D Imaging Radar Inertial Odometry and
        # Mapping,” IEEE Robotics and Automation Letters, vol. 8, no. 6, pp. 3246–3253, Jun. 2023,
        # doi: 10.1109/LRA.2023.3266669.
        weights = np.diag(np.ones(len(points_polar)))
        y_ls = velocity_hat  # initialize with prediction
        y_cov = np.eye(len(y_ls)) * 1e6  # no info if no velocity can be identified
        ls_gain = invert(self._h_mat.T @ weights @ self._h_mat) @ self._h_mat.T @ weights
        if self._use_theta_noise:
            r2_max = np.max((points_polar[:, measurement_index['doppler']] - self._h_mat @ velocity_hat)**2)

            jac = np.array([
                [-1, self._center_of_rotation_shift_local[1]],
                [0, -self._center_of_rotation_shift_local[0]],
            ])

            error_cov_theta \
                = (np.einsum('a, nab, b -> n',
                             velocity_hat, theta_cov, velocity_hat)
                   + np.array([np.trace(self._cov[[state_index['velocity'],
                                                   state_index['omega']]][:, [state_index['velocity'], state_index['omega']]]
                                        @ jac.T @ theta_cov[t] @ jac)
                               for t in range(len(theta_cov))]))
            c_dash_2 = 4.0 * (self._doppler_standard_deviation ** 2 + error_cov_theta)
        else:
            r2_max = np.max((points_polar[:, measurement_index['doppler']] - self._h_mat @ velocity_hat) ** 2)
            c_dash_2 = 4.0 * self._doppler_standard_deviation ** 2
        mu = 4.0 * r2_max / np.mean(c_dash_2)
        condition = mu > 0.0
        while condition:
            if self._use_theta_noise:
                y_ls = ls_gain @ (points_polar[:, measurement_index['doppler']])
                r_i = points_polar[:, measurement_index['doppler']] - self._h_mat @ y_ls
            else:
                y_ls = ls_gain @ points_polar[:, measurement_index['doppler']]
                r_i = points_polar[:, measurement_index['doppler']] - self._h_mat @ y_ls

            mu /= 1.4
            condition = mu >= 1.0
            if condition:
                if self._use_theta_noise:
                    jac = np.array([
                        [-1, self._center_of_rotation_shift_local[1]],
                        [0, self._center_of_rotation_shift_local[0]],
                    ])
                    error_cov_theta \
                        = (np.einsum('a, nab, b -> n',
                                     velocity_hat, theta_cov, velocity_hat)
                           + np.array([np.trace(self._cov[[state_index['velocity'],
                                                           state_index['omega']]][:, [state_index['velocity'], state_index['omega']]]
                                                @ jac.T @ theta_cov[t] @ jac)
                                       for t in range(len(theta_cov))]))
                    c_dash_2 = 4.0 * (self._doppler_standard_deviation ** 2 + error_cov_theta)
                else:
                    c_dash_2 = 4.0 * self._doppler_standard_deviation ** 2
                weights = np.diag((mu * c_dash_2 / (r_i**2 + mu * c_dash_2))**2)
                ls_gain = invert(self._h_mat.T @ weights @ self._h_mat) @ self._h_mat.T @ weights
        y = y_ls
        if ls_gain is not None:
            if self._use_theta_noise:
                jac = np.array([
                    [-1, self._center_of_rotation_shift_local[1]],
                    [0, -self._center_of_rotation_shift_local[0]],
                ])
                error_cov_theta \
                    = (np.einsum('a, nab, b -> n',
                                 velocity_hat, theta_cov, velocity_hat)
                       + np.array([np.trace(self._cov[[state_index['velocity'],
                                                       state_index['omega']]][:, [state_index['velocity'], state_index['omega']]]
                                            @ jac.T @ theta_cov[t] @ jac)
                                   for t in range(len(theta_cov))]))
                y_cov = ls_gain \
                        @ (np.eye(len(points_polar))
                           * (self._doppler_standard_deviation ** 2 + error_cov_theta)) \
                        @ ls_gain.T
            else:
                y_cov = ls_gain @ ls_gain.T * self._doppler_standard_deviation ** 2

        return y, y_cov

    def __find_velocity_via_ransac(self, points_polar, velocity_hat, theta_cov):
        # based on C. Doer and G. F. Trommer, “An EKF Based Approach to Radar Inertial Odometry,” in 2020 IEEE
        # International Conference on Multisensor Fusion and Integration for Intelligent Systems (MFI), Sep. 2020,
        # pp. 152–159, doi: 10.1109/MFI49285.2020.9235254.
        n_ransac = 17  # same assumptions as in base paper for minimum number of iterations
        if self._use_theta_noise:
            inlier_threshold = 0.05
        else:
            if self._real_data:
                inlier_threshold = 0.05
            else:
                inlier_threshold = 0.1

        max_inliers = 0  # maximum number of identified inliers so far
        y_ls = velocity_hat  # identified velocity of best run so far
        y_cov = np.eye(len(y_ls)) * 1e6  # no info if no velocity can be identified
        velocity_found = False
        best_inlier_mask = np.zeros(len(points_polar), bool)
        weights = np.ones(len(points_polar))
        for i in range(n_ransac):
            key_points = self._rng.choice(range(len(points_polar)), 3, replace=False)
            # if all rows of h_mat are the same, LS cannot be applied
            if (all(self._h_mat[key_points][0, 0] == self._h_mat[key_points][:, 0])
                    & all(self._h_mat[key_points][0, 1] == self._h_mat[key_points][:, 1])):
                continue
            ls_gain = (invert(self._h_mat[key_points].T @ np.diag(weights[key_points]) @ self._h_mat[key_points])
                       @ self._h_mat[key_points].T @ np.diag(weights[key_points]))
            if self._use_theta_noise:
                current_y_ls = ls_gain @ (points_polar[key_points, measurement_index['doppler']])
                inlier_mask = (abs((points_polar[:, measurement_index['doppler']])
                                   - self._h_mat @ current_y_ls)
                               < inlier_threshold)
            else:
                current_y_ls = ls_gain @ points_polar[key_points, measurement_index['doppler']]
                inlier_mask = abs(points_polar[:, measurement_index['doppler']]
                                  - self._h_mat @ current_y_ls) < inlier_threshold
            if (np.sum(weights[inlier_mask]) > max_inliers) & (np.sum(inlier_mask) > 3):  # at least 4 for covariance
                velocity_found = True
                max_inliers = np.sum(weights[inlier_mask])
                best_inlier_mask = inlier_mask.copy()
                if self._use_theta_noise:
                    err = (self._h_mat[inlier_mask] @ current_y_ls
                           - points_polar[inlier_mask, measurement_index['doppler']])
                else:
                    err = (self._h_mat[inlier_mask] @ current_y_ls
                           - points_polar[inlier_mask, measurement_index['doppler']])
                if any(np.array([np.isclose(np.linalg.eigvals(self._h_mat[inlier_mask].T @ np.diag(weights[inlier_mask]) @ self._h_mat[inlier_mask])[t], 0.0) for t in range(2)])):
                    add_mat = np.eye(2) * 1e-8
                else:
                    add_mat = np.eye(2) * 0.0
                ls_gain = invert(self._h_mat[inlier_mask].T @ np.diag(weights[inlier_mask]) @ self._h_mat[inlier_mask]
                                 + add_mat) @ self._h_mat[inlier_mask].T @ np.diag(weights[inlier_mask])
                y_cov = (err @ err) * ls_gain @ ls_gain.T / (len(err) - 3)
                if any(np.array([np.isclose(np.diag(y_cov)[t], 0.0) for t in range(2)])):
                    y_cov += np.eye(2) * 1e-8
                if self._use_theta_noise:
                    y_ls = ls_gain @ points_polar[inlier_mask, measurement_index['doppler']]
                else:
                    y_ls = ls_gain @ points_polar[inlier_mask, measurement_index['doppler']]
        y = y_ls
        if velocity_found:
            ls_gain = (invert(self._h_mat[best_inlier_mask].T @ np.diag(weights[best_inlier_mask])
                             @ self._h_mat[best_inlier_mask]) @ self._h_mat[best_inlier_mask].T
                       @ np.diag(weights[best_inlier_mask]))
            if self._use_theta_noise:
                jac = np.array([
                    [-1, self._center_of_rotation_shift_local[1]],
                    [0, -self._center_of_rotation_shift_local[0]],
                ])

                error_cov_theta \
                    = (np.einsum('a, nab, b -> n',
                                 velocity_hat, theta_cov[best_inlier_mask], velocity_hat)
                       + np.array([np.trace(self._cov[[state_index['velocity'],
                                                       state_index['omega']]][:, [state_index['velocity'], state_index['omega']]]
                                            @ jac.T @ theta_cov[best_inlier_mask][t] @ jac)
                                   for t in range(np.sum(best_inlier_mask))]))
                y_cov_analytical = ls_gain @ (np.eye(np.sum(best_inlier_mask))
                                              * (self._doppler_standard_deviation ** 2 + error_cov_theta)) @ ls_gain.T
                y_cov = y_cov_analytical
            elif self._real_data:
                y_cov = (invert(self._h_mat[best_inlier_mask].T @ self._h_mat[best_inlier_mask])
                         * self._doppler_standard_deviation**2)

        return y, y_cov, velocity_found

    def __find_velocity_via_baseline(self, points_polar):
        if any(np.array([np.isclose(np.linalg.eigvals(self._h_mat.T @ self._h_mat)[t], 0.0) for t in range(2)])):
            add_mat = np.eye(2) * 1e-8
        else:
            add_mat = np.eye(2) * 0.0
        ls_gain = invert(self._h_mat.T @ self._h_mat + add_mat) @ self._h_mat.T
        y = ls_gain @ points_polar[:, measurement_index['doppler']]
        y_cov = ls_gain @ (np.eye(len(points_polar)) * self._doppler_standard_deviation ** 2) @ ls_gain.T

        return y, y_cov

    def __estimate_velocity(self, points_polar, velocity_hat):
        theta_cov = self.__calculate_angle_of_arrival_noise_parameters(points_polar)
        if self._use_theta_noise:
            self.__debias_h_mat()

        velocity_found = True
        if self._mode in [MODE_GNC, MODE_GNC_THETA]:
            ls_velocity, ls_velocity_covariance = self.__find_velocity_estimate_via_gnc(points_polar, velocity_hat,
                                                                                        theta_cov)
        elif self._mode in [MODE_RANSAC, MODE_RANSAC_THETA]:
            ls_velocity, ls_velocity_covariance, velocity_found = self.__find_velocity_via_ransac(points_polar,
                                                                                                  velocity_hat,
                                                                                                  theta_cov)
        else:
            ls_velocity, ls_velocity_covariance = self.__find_velocity_via_baseline(points_polar)

        return ls_velocity, ls_velocity_covariance, velocity_found

    def __measurement_update(self, velocity_hat, ls_velocity, ls_velocity_covariance):
        jacobian_matrix = np.array([
            [0.0, 0.0, -1.0, 0.0, self._center_of_rotation_shift_local[1]],
            [0.0, 0.0, 0.0, 0.0, -self._center_of_rotation_shift_local[0]]
        ])
        innovation_covariance = jacobian_matrix @ self._cov @ jacobian_matrix.T + ls_velocity_covariance
        innovation_covariance_inv = invert(innovation_covariance)
        gain = self._cov @ jacobian_matrix.T @ innovation_covariance_inv
        self._state += gain @ (ls_velocity - velocity_hat)
        self._cov -= gain @ innovation_covariance @ gain.T

        self._cov = 0.5 * (self._cov + self._cov.T)
        self._state[state_index['angle']] = ((self._state[state_index['angle']] + np.pi) % (2.0 * np.pi)) - np.pi

    def update(self, points_polar, time_stamp):
        self.__predict(time_stamp)

        self.__set_h_matrix(points_polar)

        if self.__no_measurement_update_possible(points_polar):
            return

        velocity_hat = self.__calculate_expected_velocity()
        ls_velocity, ls_velocity_covariance, velocity_found = self.__estimate_velocity(points_polar, velocity_hat)

        if not velocity_found:
            print('No velocity identified in mode ' + self.get_label() + '. Skipping measurement update!')
            return

        self.__measurement_update(velocity_hat, ls_velocity, ls_velocity_covariance)
