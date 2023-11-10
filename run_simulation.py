import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm
import time
import os
import fnmatch

from simulator.simulator_2d import Simulator2D
from filter.radar_odometry_tracker_2d import RadarOdometryTracker2D
from visualizer.plotting import plot_2d_points_doppler, plot_ego_pose_2d
import constants
from radar_data_processor import RadarDataProcessor


def main():
    rng_seeder = np.random.default_rng(512)
    time_steps = 25
    runs = 100#0

    paper_mode = True
    plot_nees_individual = False
    real_data = True
    data_path = './data/only_turn_90/'
    radar_config_path = './radar_config.json'

    polar = True
    plot_variance = False
    plot_bias = False

    radar_data_processor = RadarDataProcessor(radar_config_path)

    # simulator settings
    lam_meas = 5
    lam_clutter = 5
    use_coordinated_turn = True
    if real_data:
        area = [-0.5, 1.5, -1.0, 1.0]
        estimate_area = [-1.0, 1.0, -1.5, 0.5]  # turn
        # estimate_area = [-0.5, 1.5, -1.0, 1.0]  # straight line/ line+turn
    else:
        area = [-5, 45, -12.5, 12.5]
        estimate_area = [-5, 45, -12.5, 12.5]

    # evaluation on real data
    only_save = False
    # final_ground_truth = np.array([0.0, -1.0, 0.0, -0.5*np.pi, 0.0])  # turn
    # final_ground_truth = np.array([1.0, 0.0, 0.0, 0.0, 0.0])  # straight line
    # final_ground_truth = np.array([1.0, 0.0, 0.0, -0.5*np.pi, 0.0])  # straight line + turn
    final_ground_truth = np.array([0.0, 0.0, 0.0, -0.5*np.pi, 0.0])  # only turn

    if real_data:
        time_steps = len(fnmatch.filter(os.listdir(data_path), '*.npy'))

    if runs == 1:
        fig, axs = plt.subplots(1, 2)

    simulator = Simulator2D(np.random.default_rng(512), polar=polar, lam_meas=lam_meas, lam_clutter=lam_clutter, area=area, use_ct=use_coordinated_turn)
    error = np.zeros((5, runs, time_steps, 6-polar))
    var_estimates = np.zeros((5, runs, time_steps, 6-polar))
    bias_estimates = np.zeros((5, runs, time_steps, 6-polar))
    nees = np.zeros((5, runs, time_steps))  # normalized estimation squared error
    nees_individual = np.zeros((5, runs, time_steps, 6-polar))
    distance_moved = np.zeros(5)
    yaw_rates_over_time = np.zeros((5, time_steps))
    missed_and_false = np.zeros((2, runs, 2))
    theta_noise_used = np.zeros(runs)
    no_detections = 0
    colors = ['red', 'blue', 'green', 'cyan', 'lightgreen']
    labels = ['base', 'RANSAC', 'RANSAC (AoA noise)', 'GNC', 'GNC (AoA noise)']
    markers = ['*', '+', '+', 'x', 'x']
    barplot_hatch = ['*', '+', '+', 'x', 'x']

    for r in tqdm(range(runs)):
        rng_seed = rng_seeder.integers(0, 9999)
        if real_data:
            last_time, _ = np.load(os.path.join(data_path, f'{0:03d}.npy'), allow_pickle=True)
        else:
            last_time = 0.0

        tracker = RadarOdometryTracker2D(last_time, ukf=False, polar=polar, mode='base',
                                         real_data=real_data, use_ct=use_coordinated_turn)
        tracker_ransac = RadarOdometryTracker2D(last_time, ukf=False, polar=polar, mode='ransac_ls',
                                                rng=np.random.default_rng(rng_seed), real_data=real_data, use_ct=use_coordinated_turn)
        tracker_ransac_theta = RadarOdometryTracker2D(last_time, ukf=False, polar=polar, mode='ransac_ls',
                                                      use_theta_noise=True, rng=np.random.default_rng(rng_seed),
                                                      real_data=real_data, use_ct=use_coordinated_turn)
        tracker_wls = RadarOdometryTracker2D(last_time, ukf=False, polar=polar, mode='weighted_ls',
                                             real_data=real_data, use_ct=use_coordinated_turn)
        tracker_wls_theta = RadarOdometryTracker2D(last_time, ukf=False, polar=polar, mode='weighted_ls',
                                                   use_theta_noise=True, real_data=real_data, use_ct=use_coordinated_turn)

        time_stamp = time.perf_counter() if real_data else 0.0

        for i in range(time_steps):
            if real_data:
                time_stamp, frame_data = np.load(os.path.join(data_path, f'{i:03d}.npy'), allow_pickle=True)
                points = radar_data_processor.identify_detections_in_polar_coordinates_from_frame_data(frame_data)
                true_state = final_ground_truth
                n_clutter = 0
                no_detections += (len(points) < 3)
            else:
                points, n_clutter = simulator.simulate_timestep(time_stamp-last_time, i)
                true_state = simulator.get_state()
            tracker.update(points, time_stamp, n_clutter)
            tracker_ransac.update(points, time_stamp, n_clutter)
            tracker_ransac_theta.update(points, time_stamp, n_clutter)
            tracker_wls.update(points, time_stamp, n_clutter)
            tracker_wls_theta.update(points, time_stamp, n_clutter)
            ego_state = tracker.get_state()
            ego_state_ransac = tracker_ransac.get_state()
            ego_state_ransac_theta = tracker_ransac_theta.get_state()
            ego_state_wls = tracker_wls.get_state()
            ego_state_wls_theta = tracker_wls_theta.get_state()
            points_cartesian = np.array([np.cos(points[:, constants.A]) * points[:, constants.R],
                                         np.sin(points[:, constants.A]) * points[:, constants.R],
                                         np.zeros(len(points)),
                                         points[:, constants.D],
                                         points[:, constants.I]]).T
            if runs == 1:
                plot_ego_pose_2d(np.array([ego_state, ego_state_ransac, ego_state_ransac_theta, ego_state_wls, ego_state_wls_theta]), radar_data_processor.get_max_range_m(), radar_data_processor.get_max_angle_degree(),
                                 fig, axs[1], angle_index=(3 if polar else 4), ground_truth=true_state, extent=estimate_area,
                                 state_color=colors)
                plot_2d_points_doppler(points_cartesian, radar_data_processor.get_max_range_m(), radar_data_processor.get_max_angle_degree(), fig, axs[0], i, ego_state=None,
                                       progressed_time=time_stamp-last_time, extent=area)
            if not real_data:
                last_time = time_stamp
                time_stamp = time.perf_counter() if real_data else time_stamp + 1.0
            error[0, r, i] = calculate_squared_error(true_state, ego_state, angle_index=(3 if polar else 4))
            error[1, r, i] = calculate_squared_error(true_state, ego_state_ransac, angle_index=(3 if polar else 4))
            error[2, r, i] = calculate_squared_error(true_state, ego_state_ransac_theta, angle_index=(3 if polar else 4))
            error[3, r, i] = calculate_squared_error(true_state, ego_state_wls, angle_index=(3 if polar else 4))
            error[4, r, i] = calculate_squared_error(true_state, ego_state_wls_theta, angle_index=(3 if polar else 4))
            var_estimates[0, r, i] = np.diag(tracker.get_covariance())
            var_estimates[1, r, i] = np.diag(tracker_ransac.get_covariance())
            var_estimates[2, r, i] = np.diag(tracker_ransac_theta.get_covariance())
            var_estimates[3, r, i] = np.diag(tracker_wls.get_covariance())
            var_estimates[4, r, i] = np.diag(tracker_wls_theta.get_covariance())
            bias_estimates[0, r, i] = calculate_error(true_state, ego_state, angle_index=(3 if polar else 4))
            bias_estimates[1, r, i] = calculate_error(true_state, ego_state_ransac, angle_index=(3 if polar else 4))
            bias_estimates[2, r, i] = calculate_error(true_state, ego_state_ransac_theta, angle_index=(3 if polar else 4))
            bias_estimates[3, r, i] = calculate_error(true_state, ego_state_wls, angle_index=(3 if polar else 4))
            bias_estimates[4, r, i] = calculate_error(true_state, ego_state_wls_theta, angle_index=(3 if polar else 4))
            nees[0, r, i] = tracker.get_nees(true_state, angle_index=(3 if polar else 4))
            nees[1, r, i] = tracker_ransac.get_nees(true_state, angle_index=(3 if polar else 4))
            nees[2, r, i] = tracker_ransac_theta.get_nees(true_state, angle_index=(3 if polar else 4))
            nees[3, r, i] = tracker_wls.get_nees(true_state, angle_index=(3 if polar else 4))
            nees[4, r, i] = tracker_wls_theta.get_nees(true_state, angle_index=(3 if polar else 4))
            nees_individual[0, r, i] = tracker.get_nees_individual(true_state, angle_index=(3 if polar else 4))
            nees_individual[1, r, i] = tracker_ransac.get_nees_individual(true_state, angle_index=(3 if polar else 4))
            nees_individual[2, r, i] = tracker_ransac_theta.get_nees_individual(true_state, angle_index=(3 if polar else 4))
            nees_individual[3, r, i] = tracker_wls.get_nees_individual(true_state, angle_index=(3 if polar else 4))
            nees_individual[4, r, i] = tracker_wls_theta.get_nees_individual(true_state, angle_index=(3 if polar else 4))

            yaw_rates_over_time[0, i] += ego_state[-1]
            yaw_rates_over_time[1, i] += ego_state_ransac[-1]
            yaw_rates_over_time[2, i] += ego_state_ransac_theta[-1]
            yaw_rates_over_time[3, i] += ego_state_wls[-1]
            yaw_rates_over_time[4, i] += ego_state_wls_theta[-1]
            if i == (time_steps-1):
                distance_moved[0] += np.linalg.norm(ego_state[:2])
                distance_moved[1] += np.linalg.norm(ego_state_ransac[:2])
                distance_moved[2] += np.linalg.norm(ego_state_ransac_theta[:2])
                distance_moved[3] += np.linalg.norm(ego_state_wls[:2])
                distance_moved[4] += np.linalg.norm(ego_state_wls_theta[:2])

        # simulator.plot_trajectory()
        simulator.reset()
        # if not real_data:
        #     for t in range(len(error)):
        #         error[t, r, 1:, :2] = (np.sqrt(error[t, r, 1:, :2]) - np.sqrt(error[t, r, :-1, :2]))**2
        #         # var_estimates[t, r, 1:, :2] = (np.sqrt(var_estimates[t, r, 1:, :2])
        #         #                                - np.sqrt(var_estimates[t, r, :-1, :2]))**2
        #         # error[t, r, 1:, 4-polar] -= error[t, r, :-1, 4-polar]

        missed_and_false[0, r] = np.array(tracker_ransac.get_missed_and_false()) / time_steps
        missed_and_false[1, r] = np.array(tracker_ransac_theta.get_missed_and_false()) / time_steps
        theta_noise_used[r] = tracker_ransac_theta.get_theta_noise_used() / time_steps

    print('No theta')
    print(np.mean(missed_and_false[0], axis=0))
    print('Theta')
    print(np.mean(missed_and_false[1], axis=0))
    print(f'Theta noise used in {100*np.mean(theta_noise_used)} percent of time steps.')
    print(f'Not enough detections in {100*no_detections/(runs*time_steps)} percent of time steps.')

    print(f'Baseline moved {distance_moved[0] / runs}m')
    print(f'RANSAC moved {distance_moved[1] / runs}m')
    print(f'RANSAC+theta moved {distance_moved[2] / runs}m')
    print(f'GNC moved {distance_moved[3] / runs}m')
    print(f'GNC+theta moved {distance_moved[4] / runs}m')

    #fig_yaw, ax_yaw = plt.subplots(1, 1)
    #for i in range(len(yaw_rates_over_time)):
    #    ax_yaw.plot(np.arange(0, time_steps), yaw_rates_over_time[i] / runs, color=colors[i], label=labels[i])
    #plt.show()

    if real_data:
        plt.style.use(f"./stylesheets/paperstyle.mplstyle")
        fig_s, ax_s = plt.subplots(1, 1)
        width = 0.8 / len(error)
        if plot_nees_individual:
            error_name = 'NEES'
            ax_s.bar(np.array(range(error.shape[-1])) - 0.4,
                     np.mean(nees_individual[0, :, -1], axis=0), width, align='edge',
                     color=colors[0], label=labels[0], hatch=barplot_hatch[0])
            ax_s.bar(np.array(range(error.shape[-1])) - 0.4 + width,
                     np.mean(nees_individual[1, :, -1], axis=0), width, align='edge',
                     color=colors[1], label=labels[1], hatch=barplot_hatch[1])
            ax_s.bar(np.array(range(error.shape[-1])) - 0.4 + 2*width,
                     np.mean(nees_individual[2, :, -1], axis=0), width, align='edge',
                     color=colors[2], label=labels[2], hatch=barplot_hatch[2])
            ax_s.bar(np.array(range(error.shape[-1])) - 0.4 + 3*width,
                     np.mean(nees_individual[3, :, -1], axis=0), width, align='edge',
                     color=colors[3], label=labels[3], hatch=barplot_hatch[3])
            ax_s.bar(np.array(range(error.shape[-1])) - 0.4 + 4*width,
                     np.mean(nees_individual[4, :, -1], axis=0), width, align='edge',
                     color=colors[4], label=labels[4], hatch=barplot_hatch[4])
        else:
            error_name = 'RMSE'
            ax_s.bar(np.array(range(error.shape[-1])) - 0.4, np.sqrt(np.mean(error[0, :, -1], axis=0)), width,
                     align='edge', color=colors[0], label=labels[0], hatch=barplot_hatch[0])
            ax_s.bar(np.array(range(error.shape[-1])) - 0.4 + width, np.sqrt(np.mean(error[1, :, -1], axis=0)),
                     width, align='edge', color=colors[1], label=labels[1], hatch=barplot_hatch[1])
            ax_s.bar(np.array(range(error.shape[-1])) - 0.4 + 2*width, np.sqrt(np.mean(error[2, :, -1], axis=0)),
                     width, align='edge', color=colors[2], label=labels[2], hatch=barplot_hatch[2])
            ax_s.bar(np.array(range(error.shape[-1])) - 0.4 + 3*width, np.sqrt(np.mean(error[3, :, -1], axis=0)),
                     width, align='edge', color=colors[3], label=labels[3], hatch=barplot_hatch[3])
            ax_s.bar(np.array(range(error.shape[-1])) - 0.4 + 4*width, np.sqrt(np.mean(error[4, :, -1], axis=0)),
                     width, align='edge', color=colors[4], label=labels[4], hatch=barplot_hatch[4])
        if polar:
            ax_s.set_xticks(np.array(range(error.shape[-1])), np.array(["$m_1$", "$m_2$", "$v$", "$\\alpha$", "$\\omega$"]))
        else:
            ax_s.set_xticks(np.array(range(error.shape[-1])), np.array(['m1', 'm2', 'v1', 'v2', 'alpha', 'omega']))

        ax_s.set_ylabel(error_name)
        fig_s.tight_layout()
        # ax_s.legend(loc='upper left')
        ax_s.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        if paper_mode:
            fig_s.savefig('./figs_odometry/real_individual_error_' + error_name)
        else:
            ax_s.set_title('Error over trajectory')
        plt.show()

        return

    plt.style.use(f"./stylesheets/paperstyle.mplstyle")
    fig_s, ax_s = plt.subplots(1, 1)
    fig_e11, ax_e11 = plt.subplots(1, 1)
    fig_e12, ax_e12 = plt.subplots(1, 1)
    fig_e21, ax_e21 = plt.subplots(1, 1)
    fig_e22, ax_e22 = plt.subplots(1, 1)
    fig_l, ax_l = plt.subplots(1, 1)
    axs_e = np.array([[ax_e11, ax_e12], [ax_e21, ax_e22]])
    for i in range(len(error)):
        width = 0.8 / len(error)
        if plot_nees_individual:
            ax_s.bar(np.array(range(error.shape[-1])) - 0.4 + i*width, np.mean(nees_individual[i], axis=(0, 1)), width,
                     align='edge', color=colors[i], label=labels[i], hatch=barplot_hatch[i])
        else:
            ax_s.bar(np.array(range(error.shape[-1])) - 0.4 + i*width, np.sqrt(np.mean(error[i], axis=(0, 1))), width,
                     align='edge', color=colors[i], label=labels[i], hatch=barplot_hatch[i])

        if plot_nees_individual:
            axs_e[0, 0].plot(np.arange(0, time_steps), np.mean(np.sum(nees_individual[i, :, :, :2], axis=2), axis=0),
                             color=colors[i], label=labels[i], marker=markers[i])
        else:
            axs_e[0, 0].plot(np.arange(0, time_steps), np.sqrt(np.mean(np.sum(error[i, :, :, :2], axis=2), axis=0)),
                             color=colors[i], label=labels[i], marker=markers[i])
        if plot_variance:
            axs_e[0, 0].plot(np.arange(0, time_steps), np.sqrt(np.mean(np.sum(var_estimates[i, :, :, :2], axis=2), axis=0)),
                             color=colors[i], linestyle='--', marker=markers[i])
        if plot_bias:
            axs_e[0, 0].plot(np.arange(0, time_steps), np.linalg.norm(np.mean(bias_estimates[i, :, :, :2], axis=0), axis=1),
                             color=colors[i], linestyle=':', marker=markers[i])
        if polar:
            if plot_nees_individual:
                axs_e[0, 1].plot(np.arange(0, time_steps), np.mean(nees_individual[i, :, :, 2], axis=0),
                                 color=colors[i], label=labels[i], marker=markers[i])
                axs_e[1, 0].plot(np.arange(0, time_steps), np.mean(nees_individual[i, :, :, 3], axis=0), color=colors[i],
                                 label=labels[i], marker=markers[i])
                axs_e[1, 1].plot(np.arange(0, time_steps), np.mean(nees_individual[i, :, :, 4], axis=0), color=colors[i],
                                 label=labels[i], marker=markers[i])
            else:
                axs_e[0, 1].plot(np.arange(0, time_steps), np.sqrt(np.mean(error[i, :, :, 2], axis=0)),
                                 color=colors[i], label=labels[i], marker=markers[i])
                axs_e[1, 0].plot(np.arange(0, time_steps), np.sqrt(np.mean(error[i, :, :, 3], axis=0)), color=colors[i],
                                 label=labels[i], marker=markers[i])
                axs_e[1, 1].plot(np.arange(0, time_steps), np.sqrt(np.mean(error[i, :, :, 4], axis=0)), color=colors[i],
                                 label=labels[i], marker=markers[i])
            if plot_variance:
                axs_e[0, 1].plot(np.arange(0, time_steps), np.sqrt(np.mean(var_estimates[i, :, :, 2], axis=0)),
                                 color=colors[i], linestyle='--', marker=markers[i])
                axs_e[1, 0].plot(np.arange(0, time_steps), np.sqrt(np.mean(var_estimates[i, :, :, 3], axis=0)),
                                 color=colors[i], linestyle='--', marker=markers[i])
                axs_e[1, 1].plot(np.arange(0, time_steps), np.sqrt(np.mean(var_estimates[i, :, :, 4], axis=0)),
                                 color=colors[i], linestyle='--', marker=markers[i])
            if plot_bias:
                axs_e[0, 1].plot(np.arange(0, time_steps), np.mean(bias_estimates[i, :, :, 2], axis=0),
                                 color=colors[i], linestyle=':', marker=markers[i])
                axs_e[1, 0].plot(np.arange(0, time_steps), np.mean(bias_estimates[i, :, :, 3], axis=0),
                                 color=colors[i], linestyle=':', marker=markers[i])
                axs_e[1, 1].plot(np.arange(0, time_steps), np.mean(bias_estimates[i, :, :, 4], axis=0),
                                 color=colors[i], linestyle=':', marker=markers[i])
        else:
            axs_e[0, 1].plot(np.arange(0, time_steps), np.sqrt(np.mean(np.sum(error[i, :, :, 2:4], axis=2), axis=0)),
                             color=colors[i], label=labels[i], marker=markers[i])
            axs_e[1, 0].plot(np.arange(0, time_steps), np.sqrt(np.mean(error[i, :, :, 4], axis=0)), color=colors[i],
                             label=labels[i], marker=markers[i])
            axs_e[1, 1].plot(np.arange(0, time_steps), np.sqrt(np.mean(error[i, :, :, 5], axis=0)), color=colors[i],
                             label=labels[i], marker=markers[i])

        if plot_nees_individual:
            ax_l.plot(np.arange(0, time_steps), np.mean(nees[i, :, :], axis=0), color=colors[i],
                      label=labels[i], marker=markers[i])
        else:
            ax_l.plot(np.arange(0, time_steps), np.sqrt(np.mean(np.sum(error[i, :, :, :], axis=2), axis=0)),
                      color=colors[i], label=labels[i], marker=markers[i])

    if polar:
        ax_s.set_xticks(np.array(range(error.shape[-1])), np.array(["$m_1$", "$m_2$", "$v$", "$\\alpha$", "$\\omega$"]))
    else:
        ax_s.set_xticks(np.array(range(error.shape[-1])), np.array(['m1', 'm2', 'v1', 'v2', 'alpha', 'omega']))

    if plot_nees_individual:
        error_name = 'NEES'
    else:
        error_name = 'RMSE'
    ax_s.set_ylabel(error_name)

    axs_e[0, 0].set_xlabel('time step')
    axs_e[0, 0].set_ylabel(error_name)

    axs_e[0, 1].set_xlabel('time step')
    axs_e[0, 1].set_ylabel(error_name)

    axs_e[1, 0].set_xlabel('time step')
    axs_e[1, 0].set_ylabel(error_name)

    axs_e[1, 1].set_xlabel('time step')
    axs_e[1, 1].set_ylabel(error_name)

    fig_s.tight_layout()
    # ax_s.legend(loc='upper right')
    ax_s.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    fig_e11.tight_layout()
    ax_e11.legend(loc='upper left')
    ax_e11.set_ylim(0, 12.5)
    ax_e11.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    fig_e12.tight_layout()
    ax_e12.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax_e12.set_ylim(0, 0.4)
    fig_e21.tight_layout()
    ax_e21.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax_e21.set_ylim(0, 1.9)
    fig_e22.tight_layout()
    ax_e22.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax_e22.set_ylim(0, 0.9)
    fig_l.tight_layout()
    ax_l.legend(loc='upper left')
    ax_l.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    fig_s.savefig('./figs_odometry/individual_error_' + error_name)
    fig_e11.savefig('./figs_odometry/position_' + error_name)
    fig_e12.savefig('./figs_odometry/velocity_' + error_name)
    fig_e21.savefig('./figs_odometry/orientation_' + error_name)
    fig_e22.savefig('./figs_odometry/yawrate_' + error_name)
    fig_l.savefig('./figs_odometry/overall_' + error_name)
    if not paper_mode:
        ax_s.set_title('Error over trajectory')
        axs_e[0, 0].set_title('Positional error over time')
        axs_e[0, 1].set_title('Velocity error over time')
        axs_e[1, 0].set_title('Orientation error over time')
        axs_e[1, 1].set_title('Yaw rate error over time')
        ax_l.set_title(error_name)
        ax_e12.legend(loc='upper left')
        ax_e21.legend(loc='upper left')
        ax_e22.legend(loc='upper left')
    plt.show()


def calculate_squared_error(true_state, ego_state, angle_index):
    return calculate_error(true_state, ego_state, angle_index)**2


def calculate_error(true_state, ego_state, angle_index):
    error = true_state - ego_state
    error[angle_index] = ((error[angle_index] + np.pi) % (2.0*np.pi)) - np.pi
    return error


if __name__ == "__main__":
    main()
