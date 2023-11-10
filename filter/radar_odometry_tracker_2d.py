import numpy as np

from filter.abstract_filter import AbstractFilter
from utils import rot_mat, invert
import constants


class RadarOdometryTracker2D(AbstractFilter):
    def __init__(self, time_stamp, ukf=False, polar=False, mode='base', rng=None, use_theta_noise=False, real_data=False,
                 use_ct=False):
        assert mode in ['base', 'weighted_ls', 'ransac_ls']
        self._ukf = ukf  # if False, use ekf
        self._polar = polar  # else, assume Cartesian velocity
        self._mode = mode
        self._rng = rng
        if (self._mode in ['ransac_ls']) & (self._rng is None):
            raise ValueError('RNG needs to be provided when using tracker with Mode: ' + self._mode)

        self._use_theta_noise = use_theta_noise

        if self._polar:
            if real_data:
                self._state = np.array(constants.PRIOR_REAL)[[0, 1, 2, 4, 5]]  # mx, my, vx, yaw, yaw rate
            else:
                self._state = np.array(constants.PRIOR)[[0, 1, 2, 4, 5]]  # mx, my, vx, yaw, yaw rate
            self._cov = np.diag(np.array(constants.PRIOR_COV)[[0, 1, 2, 4, 5]])**2
            self._sigma_q = np.hstack([constants.SIGMA_A[0], constants.SIGMA_W[0]])
            self._m1 = 0
            self._m2 = 1
            self._v1 = 2
            self._v2 = None
            self._al = 3
            self._om = 4
        else:
            if real_data:
                self._state = constants.PRIOR_REAL  # mx, my, vx, vy, yaw, yaw rate
            else:
                self._state = constants.PRIOR  # mx, my, vx, vy, yaw, yaw rate
            self._cov = np.diag(constants.PRIOR_COV)**2
            self._sigma_q = np.hstack([constants.SIGMA_A[:2], constants.SIGMA_W[0]])
            self._m1 = 0
            self._m2 = 1
            self._v1 = 2
            self._v2 = 3
            self._al = 4
            self._om = 5
        self._center_of_rotation_shift = constants.CENTER_OF_ROTATION_SHIFT[:2]
        self._sigma_doppler = constants.SIGMA_DOPPLER
        self._sigma_theta = constants.SIGMA_THETA

        self._last_time = time_stamp

        # for clutter evaluation of RANSAC based approach
        self._missed = 0  # missed detections
        self._false = 0  # false detections (clutter)

        # for hybrid approach
        self._theta_noise_used = 0

        self._real_data = real_data

        self._use_coordinated_turn = use_ct

    def get_state(self):
        return self._state

    def get_covariance(self):
        return self._cov

    def get_nees(self, est_state, angle_index):
        diff = self._state - est_state
        diff[angle_index] = ((diff[angle_index] + np.pi) % (2.0*np.pi)) - np.pi
        return diff @ invert(self._cov) @ diff

    def get_nees_individual(self, est_state, angle_index):
        diff = self._state - est_state
        diff[angle_index] = ((diff[angle_index] + np.pi) % (2.0*np.pi)) - np.pi
        return diff**2 / np.diag(self._cov)

    def get_missed_and_false(self):
        return self._missed, self._false

    def get_theta_noise_used(self):
        return self._theta_noise_used

    def predict(self, time_difference):
        if self._use_coordinated_turn:
            state_aug = np.block([self._state, np.zeros(2)])
            cov_aug = np.zeros((len(self._state)+2, len(self._state)+2))
            cov_aug[:len(self._state), :len(self._state)] = self._cov
            cov_aug[len(self._state):, len(self._state):] = np.diag(self._sigma_q)**2

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
                    [0.5*time_difference**2*np.cos(x_sig[i, self._al]), 0.0],
                    [0.5*time_difference**2*np.sin(x_sig[i, self._al]), 0.0],
                    [time_difference, 0.0],
                    [0.0, 0.5*time_difference**2],
                    [0.0, time_difference]
                ])
                if abs(x_sig[i, self._om]) > 1e-6:
                    x_sig[i, :len(self._state)] = np.array([
                        x_sig[i, self._m1] + 2.0 * x_sig[i, self._v1] / x_sig[i, self._om] * np.sin(0.5*x_sig[i, self._om]*time_difference)
                        * np.cos(x_sig[i, self._al] + 0.5*x_sig[i, self._om]*time_difference),
                        x_sig[i, self._m2] + 2.0 * x_sig[i, self._v1] / x_sig[i, self._om] * np.sin(0.5*x_sig[i, self._om]*time_difference)
                        * np.sin(x_sig[i, self._al] + 0.5*x_sig[i, self._om]*time_difference),
                        x_sig[i, self._v1],
                        x_sig[i, self._al] + time_difference*x_sig[i, self._om],
                        x_sig[i, self._om]
                    ])
                else:
                    x_sig[i, :len(self._state)] = np.array([
                        x_sig[i, self._m1] + time_difference*x_sig[i, self._v1]*np.cos(x_sig[i, self._al]),
                        x_sig[i, self._m2] + time_difference*x_sig[i, self._v1]*np.sin(x_sig[i, self._al]),
                        x_sig[i, self._v1],
                        x_sig[i, self._al] + time_difference*x_sig[i, self._om],
                        x_sig[i, self._om],
                    ])
                x_sig[i, :len(self._state)] += error_mat @ x_sig[i, len(self._state):]
            self._state = np.sum(w_sig[:, None] * x_sig[:, :len(self._state)], axis=0)
            self._cov = np.einsum('x, xa, xb -> ab', w_sig, x_sig[:, :len(self._state)] - self._state[None, :],
                                  x_sig[:, :len(self._state)] - self._state[None, :])
        else:
            if self._polar:
                tran_mat = np.eye(len(self._state))
                tran_mat[self._m1, self._v1] = time_difference * np.cos(self._state[self._al])
                tran_mat[self._m2, self._v1] = time_difference * np.sin(self._state[self._al])
                tran_mat[self._al, self._om] = time_difference

                error_mat = np.array([
                    [0.5*time_difference**2*np.cos(self._state[self._al]), 0.0],
                    [0.5*time_difference**2*np.sin(self._state[self._al]), 0.0],
                    [time_difference, 0.0],
                    [0.0, 0.5*time_difference**2],
                    [0.0, time_difference]
                ])
            else:
                tran_mat = np.eye(len(self._state))
                tran_mat[self._m1, self._v1] = time_difference
                tran_mat[self._m2, self._v2] = time_difference
                tran_mat[self._al, self._om] = time_difference
                error_mat = np.array([
                    [0.5*time_difference**2, 0.0, 0.0],
                    [0.0, 0.5*time_difference**2, 0.0],
                    [time_difference, 0.0, 0.0],
                    [0.0, time_difference, 0.0],
                    [0.0, 0.0, 0.5*time_difference**2],
                    [0.0, 0.0, time_difference]
                ])
            self._state = tran_mat @ self._state
            self._cov = tran_mat @ self._cov @ tran_mat.T + error_mat @ np.diag(self._sigma_q)**2 @ error_mat.T

    def update(self, points_polar, time_stamp, n_clutter):
        # predict to current time step
        time_difference = time_stamp - self._last_time
        self._last_time = time_stamp
        self.predict(time_difference)

        # measurement update
        if len(points_polar) < 3:
            print('Not enough points to solve LS. Skipping measurement update!')
            return
        # get expected velocity
        if self._polar:
            y_hat = -(np.array([self._state[self._v1], 0.0])
                      + self._state[self._om] * rot_mat(0.5 * np.pi) @ self._center_of_rotation_shift)
        else:
            y_hat = -(rot_mat(self._state[self._al]).T @ self._state[[self._v1, self._v2]]
                      + self._state[self._om] * rot_mat(0.5 * np.pi) @ self._center_of_rotation_shift)
        # get measurement velocity from Doppler measurements via Least Squares
        h_mat = np.array([np.cos(points_polar[:, constants.A]), np.sin(points_polar[:, constants.A])]).T
        velocity_found = True

        # if all rows of h_mat are the same, LS cannot be applied
        if all(h_mat[0, 0] == h_mat[:, 0]) & all(h_mat[0, 1] == h_mat[:, 1]):
            print('All rows of H are the same, LS not possible!')
            return

        # preparations for including noise on bearing measurements
        theta_cov = np.array([
            [0.5 + 0.5*np.cos(2*points_polar[:, constants.A])*np.exp(-2*self._sigma_theta**2)-np.cos(points_polar[:, constants.A])**2*np.exp(-self._sigma_theta**2),
             0.5*np.sin(2*points_polar[:, constants.A])*np.exp(-2*self._sigma_theta**2)-np.cos(points_polar[:, constants.A])*np.sin(points_polar[:, constants.A])*np.exp(-self._sigma_theta**2)],
            [0.5*np.sin(2*points_polar[:, constants.A])*np.exp(-2*self._sigma_theta**2)-np.cos(points_polar[:, constants.A])*np.sin(points_polar[:, constants.A])*np.exp(-self._sigma_theta**2),
                0.5 - 0.5*np.cos(2*points_polar[:, constants.A])*np.exp(-2*self._sigma_theta**2)-np.sin(points_polar[:, constants.A])**2*np.exp(-self._sigma_theta**2)]
        ]).transpose((2, 0, 1))
        theta_bias = (np.exp(-0.5*self._sigma_theta**2) - 1.0) * np.array([np.cos(points_polar[:, constants.A]),
                                                                           np.sin(points_polar[:, constants.A])]).T
        if self._mode == 'weighted_ls':
            # based on Y. Zhuang, B. Wang, J. Huai, and M. Li, “4D iRIOM: 4D Imaging Radar Inertial Odometry and
            # Mapping,” IEEE Robotics and Automation Letters, vol. 8, no. 6, pp. 3246–3253, Jun. 2023,
            # doi: 10.1109/LRA.2023.3266669.
            weights = np.diag(np.ones(len(points_polar)))
            y_ls = y_hat  # identified velocity of best run so far
            y_ls_old = y_hat
            y_cov = np.eye(len(y_ls)) * 1e6  # no info if no velocity can be identified
            ls_gain = invert(h_mat.T @ weights @ h_mat) @ h_mat.T @ weights
            if self._use_theta_noise:
                r2_max = np.max((points_polar[:, constants.D] - theta_bias @ y_ls - h_mat @ y_hat)**2)

                jac = np.array([
                    [-1, self._center_of_rotation_shift[1]],
                    [0, -self._center_of_rotation_shift[0]],
                ])

                error_cov_theta \
                    = (np.einsum('a, nab, b -> n',
                                 y_hat, theta_cov, y_hat)
                       + np.array([np.trace(self._cov[[self._v1, self._om]][:, [self._v1, self._om]] @ jac.T @ theta_cov[t] @ jac)
                                   for t in range(len(theta_cov))])
                       + np.einsum('na, ab, bc, dc, nd -> n', theta_bias, jac, self._cov[[self._v1, self._om]][:, [self._v1, self._om]], jac, theta_bias))
                c_dash_2 = 4.0 * (self._sigma_doppler**2 + error_cov_theta)
            else:
                r2_max = np.max((points_polar[:, constants.D] - h_mat @ y_hat)**2)
                c_dash_2 = 4.0 * self._sigma_doppler**2
            mu = 4.0 * r2_max / np.mean(c_dash_2)
            condition = mu > 0.0
            while condition:
                y_ls_old = y_ls
                if self._use_theta_noise:
                    # y_ls = ls_gain @ (points_polar[:, constants.D] - theta_bias @ y_ls_old)
                    # r_i = points_polar[:, constants.D] - theta_bias @ y_ls_old - h_mat @ y_ls
                    y_ls = ls_gain @ (points_polar[:, constants.D] - theta_bias @ y_hat)
                    r_i = points_polar[:, constants.D] - theta_bias @ y_hat - h_mat @ y_ls
                else:
                    y_ls = ls_gain @ points_polar[:, constants.D]
                    r_i = points_polar[:, constants.D] - h_mat @ y_ls

                mu /= 1.4
                condition = mu >= 1.0
                if condition:
                    if self._use_theta_noise:
                        jac = np.array([
                            [-1, self._center_of_rotation_shift[1]],
                            [0, self._center_of_rotation_shift[0]],
                        ])
                        error_cov_theta \
                            = (np.einsum('a, nab, b -> n',
                                         y_hat, theta_cov, y_hat)
                               + np.array([np.trace(self._cov[[self._v1, self._om]][:, [self._v1, self._om]] @ jac.T @ theta_cov[t] @ jac)
                                           for t in range(len(theta_cov))])
                               + np.einsum('na, ab, bc, dc, nd -> n', theta_bias, jac, self._cov[[self._v1, self._om]][:, [self._v1, self._om]], jac, theta_bias))
                        c_dash_2 = 4.0 * (self._sigma_doppler**2 + error_cov_theta)
                        # c_dash_2 = 4.0 * (self._sigma_doppler**2 + np.einsum('a, nab, b -> n', y_ls_old, theta_cov, y_ls_old))
                    else:
                        # err = h_mat @ y_ls - points_polar[:, constants.D]
                        # if len(err) > 3:
                        #     c_dash_2 = 4.0 * (err @ err) / (len(points_polar) - 3)
                        # else:
                        c_dash_2 = 4.0 * self._sigma_doppler**2
                    weights = np.diag((mu * c_dash_2 / (r_i**2 + mu * c_dash_2))**2)
                    ls_gain = invert(h_mat.T @ weights @ h_mat) @ h_mat.T @ weights
            y = y_ls
            if ls_gain is not None:
                # y_cov = invert(h_mat.T @ weights @ h_mat)
                if self._use_theta_noise:
                    # error_cov_theta = np.einsum('a, nab, b -> n',
                    #                             y_ls_old, theta_cov, y_ls_old)
                    jac = np.array([
                        [-1, self._center_of_rotation_shift[1]],
                        [0, -self._center_of_rotation_shift[0]],
                    ])
                    error_cov_theta \
                        = (np.einsum('a, nab, b -> n',
                                     y_hat, theta_cov, y_hat)
                           + np.array([np.trace(self._cov[[self._v1, self._om]][:, [self._v1, self._om]] @ jac.T @ theta_cov[t] @ jac)
                                       for t in range(len(theta_cov))])
                           + np.einsum('na, ab, bc, dc, nd -> n', theta_bias, jac, self._cov[[self._v1, self._om]][:, [self._v1, self._om]], jac, theta_bias))
                    y_cov = ls_gain \
                            @ (np.eye(len(points_polar))
                               * (self._sigma_doppler**2 + error_cov_theta)) \
                            @ ls_gain.T
                else:
                    y_cov = ls_gain @ ls_gain.T * self._sigma_doppler**2
        elif self._mode == 'ransac_ls':
            # based on C. Doer and G. F. Trommer, “An EKF Based Approach to Radar Inertial Odometry,” in 2020 IEEE
            # International Conference on Multisensor Fusion and Integration for Intelligent Systems (MFI), Sep. 2020,
            # pp. 152–159, doi: 10.1109/MFI49285.2020.9235254.
            n_ransac = 17  # same assumptions as in base paper for minimum number of iterations
            # use different thresholds as empirical noise can handle clutter better
            if self._use_theta_noise:
                inlier_threshold = 0.05
            else:
                if self._real_data:
                    inlier_threshold = 0.05
                else:
                    inlier_threshold = 0.1
            # inlier_threshold = 0.01
            # inlier_threshold = 0.1#self._sigma_doppler*10*3

            max_inliers = 0  # maximum number of identified inliers so far
            y_ls = y_hat  # identified velocity of best run so far
            last_y_ls = y_hat
            y_cov = np.eye(len(y_ls)) * 1e6  # no info if no velocity can be identified
            velocity_found = False
            best_key_points = np.zeros(3)
            best_inlier_mask = np.zeros(len(points_polar), bool)
            weights = np.ones(len(points_polar))  # np.exp(-points_polar[:, constants.I])
            for i in range(n_ransac):
                key_points = self._rng.choice(range(len(points_polar)), 3, replace=False)
                # if all rows of h_mat are the same, LS cannot be applied
                if all(h_mat[key_points][0, 0] == h_mat[key_points][:, 0]) & all(h_mat[key_points][0, 1] == h_mat[key_points][:, 1]):
                    continue
                ls_gain = invert(h_mat[key_points].T @ np.diag(weights[key_points]) @ h_mat[key_points]) @ h_mat[key_points].T @ np.diag(weights[key_points])
                if self._use_theta_noise:
                    # current_y_ls = ls_gain @ (points_polar[key_points, constants.D] - theta_bias[key_points] @ y_ls)
                    # inlier_mask = (abs((points_polar[:, constants.D] - theta_bias @ y_ls) - h_mat @ current_y_ls)
                    #                < inlier_threshold)
                    current_y_ls = ls_gain @ (points_polar[key_points, constants.D] - theta_bias[key_points] @ y_hat)
                    inlier_mask = (abs((points_polar[:, constants.D] - theta_bias @ y_hat) - h_mat @ current_y_ls)
                                   < inlier_threshold)
                else:
                    current_y_ls = ls_gain @ points_polar[key_points, constants.D]
                    inlier_mask = abs(points_polar[:, constants.D] - h_mat @ current_y_ls) < inlier_threshold
                if (np.sum(weights[inlier_mask]) > max_inliers) & (np.sum(inlier_mask) > 3):  # at least 4 for covariance
                    velocity_found = True
                    max_inliers = np.sum(weights[inlier_mask])
                    best_key_points = key_points
                    best_inlier_mask = inlier_mask.copy()
                    if self._use_theta_noise:
                        err = h_mat[inlier_mask] @ current_y_ls - (points_polar[inlier_mask, constants.D]
                                                           - theta_bias[inlier_mask] @ y_hat)
                    else:
                        err = h_mat[inlier_mask] @ current_y_ls - points_polar[inlier_mask, constants.D]
                    # err = np.ones(len(err)) * self._sigma_doppler
                    # if all inlier angles are the same due to discretization, small covariance needs to be added to avoid numerical errors
                    if any(np.array([np.isclose(np.linalg.eigvals(h_mat[inlier_mask].T @ np.diag(weights[inlier_mask]) @ h_mat[inlier_mask])[t], 0.0) for t in range(2)])):
                        add_mat = np.eye(2) * 1e-8
                    else:
                        add_mat = np.eye(2) * 0.0
                    ls_gain = invert(h_mat[inlier_mask].T @ np.diag(weights[inlier_mask]) @ h_mat[inlier_mask] + add_mat) @ h_mat[inlier_mask].T @ np.diag(weights[inlier_mask])
                    y_cov = (err @ err) * ls_gain @ ls_gain.T / (len(err) - 3)  # invert(h_mat[inlier_mask].T @ np.diag(weights[inlier_mask]) @ h_mat[inlier_mask] + add_mat)) / (len(err) - 3)
                    if any(np.array([np.isclose(np.diag(y_cov)[t], 0.0) for t in range(2)])):
                        # covariance might get too small if standing still, leading to numerical issues
                        y_cov += np.eye(2) * 1e-8
                    last_y_ls = y_ls
                    if self._use_theta_noise:
                        y_ls = ls_gain @ (points_polar[inlier_mask, constants.D] - theta_bias[inlier_mask] @ y_hat)
                    else:
                        y_ls = ls_gain @ points_polar[inlier_mask, constants.D]
            y = y_ls
            if not velocity_found:
                print('No velocity identified in RANSAC mode. Skipping measurement update!')
            else:
                if any(best_inlier_mask[(len(best_inlier_mask) - n_clutter):]):
                    self._false += np.sum(best_inlier_mask[(len(best_inlier_mask) - n_clutter):])
                if not all(best_inlier_mask[:(len(best_inlier_mask) - n_clutter)]):
                    self._missed += (len(best_inlier_mask) - n_clutter
                                     - np.sum(best_inlier_mask[:(len(best_inlier_mask) - n_clutter)]))
                ls_gain = invert(h_mat[best_inlier_mask].T @ np.diag(weights[best_inlier_mask]) @ h_mat[best_inlier_mask]) @ h_mat[best_inlier_mask].T @ np.diag(weights[best_inlier_mask])
                if self._use_theta_noise:
                    # error_cov_theta = np.einsum('a, nab, b -> n',
                    #                             last_y_ls, theta_cov[best_inlier_mask], last_y_ls)
                    jac = np.array([
                        [-1, self._center_of_rotation_shift[1]],
                        [0, -self._center_of_rotation_shift[0]],
                    ])

                    error_cov_theta \
                        = (np.einsum('a, nab, b -> n',
                                     y_hat, theta_cov[best_inlier_mask], y_hat)
                           + np.array([np.trace(self._cov[[self._v1, self._om]][:, [self._v1, self._om]] @ jac.T @ theta_cov[best_inlier_mask][t] @ jac)
                                       for t in range(np.sum(best_inlier_mask))])
                           + np.einsum('na, ab, bc, dc, nd -> n', theta_bias[best_inlier_mask], jac, self._cov[[self._v1, self._om]][:, [self._v1, self._om]], jac, theta_bias[best_inlier_mask]))
                    y_cov_analytical = ls_gain @ (np.eye(np.sum(best_inlier_mask)) * (self._sigma_doppler**2 + error_cov_theta)) @ ls_gain.T
                    # if not any(best_inlier_mask[(len(best_inlier_mask) - n_clutter):]):
                    if True:#np.trace(y_cov) < np.trace(y_cov_analytical):
                    # if (err @ err) / (len(err) - 3) > (self._sigma_doppler**2 + np.mean(error_cov_theta)):
                        y_cov = y_cov_analytical
                        self._theta_noise_used += 1
                elif self._real_data:
                    y_cov = invert(h_mat[best_inlier_mask].T @ h_mat[best_inlier_mask]) * self._sigma_doppler**2
        else:
            if any(np.array([np.isclose(np.linalg.eigvals(h_mat.T @ h_mat)[t], 0.0) for t in range(2)])):
                add_mat = np.eye(2) * 1e-8
            else:
                add_mat = np.eye(2) * 0.0
            ls_gain = invert(h_mat.T @ h_mat + add_mat) @ h_mat.T
            if self._use_theta_noise:
                y = ls_gain @ (points_polar[:, constants.D] - theta_bias @ y_hat)
                y_cov = ls_gain \
                        @ (np.eye(len(points_polar)) * (self._sigma_doppler**2
                                                        + np.einsum('a, nab, b -> n', y_hat, theta_cov, y_hat))) \
                        @ ls_gain.T
            else:
                y = ls_gain @ points_polar[:, constants.D]
                y_cov = ls_gain @ (np.eye(len(points_polar)) * self._sigma_doppler**2) @ ls_gain.T

        if not velocity_found:
            return

        # calculate jacobian and update state
        if self._ukf:
            raise NotImplementedError('UKF not yet implemented.')
        else:
            cos_alpha = np.cos(self._state[self._al])
            sin_alpha = np.sin(self._state[self._al])
            if self._polar:
                jacobian_matrix = np.array([
                    [0.0, 0.0, -1.0, 0.0, self._center_of_rotation_shift[1]],
                    [0.0, 0.0, 0.0, 0.0, -self._center_of_rotation_shift[0]]
                ])
            else:
                jacobian_matrix = np.array([
                    [0.0, 0.0, -cos_alpha, sin_alpha,
                     self._state[self._v1] * sin_alpha + self._state[self._v2] * cos_alpha,
                     self._center_of_rotation_shift[1]],
                    [0.0, 0.0, -sin_alpha, -cos_alpha,
                     -self._state[self._v1] * cos_alpha + self._state[self._v2] * sin_alpha,
                     -self._center_of_rotation_shift[0]],
                ])
            innovation_covariance = jacobian_matrix @ self._cov @ jacobian_matrix.T + y_cov
            innovation_covariance_inv = invert(innovation_covariance)
            # innovation_covariance_inv = np.eye(len(points_polar)) / self._sigma_doppler**2 \
            #                             - (np.eye(len(points_polar)) / self._sigma_doppler**2) @ jacobian_matrix \
            #                             @ invert(invert(self._cov)
            #                                             + jacobian_matrix.T @ (np.eye(len(points_polar))
            #                                                                    / self._sigma_doppler**2) @ jacobian_matrix) \
            #                             @ jacobian_matrix.T @ (np.eye(len(points_polar)) / self._sigma_doppler**2)
        gain = self._cov @ jacobian_matrix.T @ innovation_covariance_inv
        self._state += gain @ (y - y_hat)
        self._cov -= gain @ innovation_covariance @ gain.T

        self._cov = 0.5 * (self._cov + self._cov.T)
        self._state[self._al] = ((self._state[self._al] + np.pi) % (2.0 * np.pi)) - np.pi
