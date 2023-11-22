import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow
from matplotlib.ticker import FormatStrFormatter

from array_indices import measurement_index

ERROR_FIGURE_PATH = './error_figures'
TRAJECTORY_FIGURE_PATH = './trajectory_figures'


class Plotter:
    def __init__(self, real_data, paper_mode, scenario_name, display_legend_on_barplot=False):
        self._real_data = real_data
        self._paper_mode = paper_mode
        self._scenario_name = scenario_name
        self._ax = None
        self._display_legend_on_barplot = display_legend_on_barplot
        self._error_figure_path, self._trajectory_figure_path = self.make_paths()

    def make_paths(self):
        if not os.path.isdir(ERROR_FIGURE_PATH):
            os.mkdir(ERROR_FIGURE_PATH)
        error_figure_path = os.path.join(ERROR_FIGURE_PATH, self._scenario_name)
        if not os.path.isdir(error_figure_path):
            os.mkdir(error_figure_path)
        if not os.path.isdir(TRAJECTORY_FIGURE_PATH):
            os.mkdir(TRAJECTORY_FIGURE_PATH)
        trajectory_figure_path = os.path.join(TRAJECTORY_FIGURE_PATH, self._scenario_name)
        if not os.path.isdir(trajectory_figure_path):
            os.mkdir(trajectory_figure_path)
        return error_figure_path, trajectory_figure_path

    def initialize_plotting_of_timesteps(self):
        fig, self._ax = plt.subplots(1, 2)
        fig.set_size_inches(16.0, 8.0)

    def plot_2d_points_doppler(self, points_polar, max_range_m, max_angle_degrees, step, progressed_time=None,
                               extent=None):
        assert self._ax is not None
        points = np.array([np.cos(points_polar[:, measurement_index['bearing']])
                           * points_polar[:, measurement_index['range']],
                           np.sin(points_polar[:, measurement_index['bearing']])
                           * points_polar[:, measurement_index['range']],
                           points_polar[:, measurement_index['doppler']],
                           points_polar[:, measurement_index['intensity']]]).T

        self._ax[0].clear()

        # xy-plane
        if extent is None:
            extent = [0,
                      max_range_m,
                      -np.sin(np.deg2rad(max_angle_degrees))*max_range_m,
                      np.sin(np.deg2rad(max_angle_degrees))*max_range_m]
        self._ax[0].set_xlabel("x (m)")
        self._ax[0].set_ylabel("y (m)")
        self._ax[0].set_aspect("auto")
        self._ax[0].set_xlim(extent[:2])
        self._ax[0].set_ylim(extent[2:])

        self._ax[0].scatter(points[:, measurement_index['m1']], points[:, measurement_index['m2']],
                            c='black', s=2.0)

        points_d = points[abs(points[:, measurement_index['doppler']]) != 0.0]
        for i in range(len(points_d)):
            self._ax[0].add_patch(Arrow(points_d[i, measurement_index['m1']], points_d[i, measurement_index['m2']],
                                        points_d[i, measurement_index['m1']] * points_d[i, measurement_index['doppler']]
                                        / np.linalg.norm(points_d[i, [measurement_index['m1'], measurement_index['m2']]]),
                                        points_d[i, measurement_index['m2']] * points_d[i, measurement_index['doppler']]
                                        / np.linalg.norm(points_d[i, [measurement_index['m1'], measurement_index['m2']]]),
                                        color='red', width=0.1))

        title = f'Step: {step:03d}'
        if progressed_time is not None:
            title += f' Time: {progressed_time:.3f}'
        self._ax[0].set_title(title)
        self._ax[0].set_aspect('equal')

        plt.draw()
        plt.savefig(os.path.join(self._trajectory_figure_path, f'{step:03d}' + '.png'))
        plt.pause(1e-3)

    def plot_ego_pose_2d(self, trackers, max_range_m, max_angle_degrees, angle_index, ground_truth=None, extent=None):
        assert self._ax is not None
        ego_state = np.array([tracker.get_state() for tracker in trackers])
        state_color = np.array([tracker.get_color() for tracker in trackers])
        self._ax[1].clear()
        if extent is None:
            self._ax[1].set_xlim([0, max_range_m])
            self._ax[1].set_ylim([-np.sin(max_angle_degrees)*max_range_m, np.sin(max_angle_degrees)*max_range_m])
        else:
            self._ax[1].set_xlim(extent[:2])
            self._ax[1].set_ylim(extent[2:])

        if len(ego_state.shape) > 1:
            assert len(ego_state) == len(state_color)
        else:
            ego_state = np.atleast_2d(ego_state)
            state_color = np.atleast_1d(state_color)

        for i in range(len(ego_state)):
            pose_vector = np.array([np.cos(ego_state[i, angle_index]), np.sin(ego_state[i, angle_index])])
            self._ax[1].quiver(ego_state[i, 0], ego_state[i, 1], pose_vector[0], pose_vector[1], color=state_color[i],
                               scale=3.0)
        if ground_truth is not None:
            pose_vector_gt = np.array([np.cos(ground_truth[angle_index]), np.sin(ground_truth[angle_index])])
            self._ax[1].quiver(ground_truth[0], ground_truth[1], pose_vector_gt[0], pose_vector_gt[1], color='black',
                               alpha=0.5, scale=3.0)
        self._ax[1].set_aspect('equal')

    def plot_errors(self, trackers):
        error = []
        for tracker in trackers:
            error.append(tracker.get_error())
        error = np.array(error)

        if self._real_data:
            plt.style.use(f"./stylesheets/paperstyle.mplstyle")
            fig_s, ax_s = plt.subplots(1, 1)
            width = 0.8 / len(trackers)
            for i in range(len(trackers)):
                ax_s.bar(np.array(range(error.shape[-1])) - 0.4 + i*width, np.sqrt(np.mean(error[i, :, -1], axis=0)),
                         width, align='edge', color=trackers[i].get_color(), label=trackers[i].get_label(),
                         hatch=trackers[i].get_marker())
            ax_s.set_xticks(np.array(range(error.shape[-1])),
                            np.array(["$m_1$", "$m_2$", "$v$", "$\\alpha$", "$\\omega$"]))

            ax_s.set_ylabel('RMSE')
            fig_s.tight_layout()
            ax_s.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            if self._display_legend_on_barplot:
                ax_s.legend(loc='upper left')
            if self._paper_mode:
                fig_s.savefig(os.path.join(self._error_figure_path, 'real_individual_error_RMSE_' + self._scenario_name))
            else:
                ax_s.set_title('Error over trajectory')
            plt.show()
        else:
            plt.style.use(f"./stylesheets/paperstyle.mplstyle")
            fig_s, ax_s = plt.subplots(1, 1)
            fig_e11, ax_e11 = plt.subplots(1, 1)
            fig_e12, ax_e12 = plt.subplots(1, 1)
            fig_e21, ax_e21 = plt.subplots(1, 1)
            fig_e22, ax_e22 = plt.subplots(1, 1)
            fig_l, ax_l = plt.subplots(1, 1)
            axs_e = np.array([[ax_e11, ax_e12], [ax_e21, ax_e22]])
            time_steps = error.shape[2]
            for i in range(len(trackers)):
                width = 0.8 / len(trackers)
                ax_s.bar(np.array(range(error.shape[-1])) - 0.4 + i*width, np.sqrt(np.mean(error[i], axis=(0, 1))),
                         width, align='edge', color=trackers[i].get_color(), label=trackers[i].get_label(),
                         hatch=trackers[i].get_marker())

                axs_e[0, 0].plot(np.arange(0, time_steps), np.sqrt(np.mean(np.sum(error[i, :, :, :2], axis=2), axis=0)),
                                 color=trackers[i].get_color(), label=trackers[i].get_label(),
                                 marker=trackers[i].get_marker())
                axs_e[0, 1].plot(np.arange(0, time_steps), np.sqrt(np.mean(error[i, :, :, 2], axis=0)),
                                 color=trackers[i].get_color(), label=trackers[i].get_label(),
                                 marker=trackers[i].get_marker())
                axs_e[1, 0].plot(np.arange(0, time_steps), np.sqrt(np.mean(error[i, :, :, 3], axis=0)),
                                 color=trackers[i].get_color(), label=trackers[i].get_label(),
                                 marker=trackers[i].get_marker())
                axs_e[1, 1].plot(np.arange(0, time_steps), np.sqrt(np.mean(error[i, :, :, 4], axis=0)),
                                 color=trackers[i].get_color(), label=trackers[i].get_label(),
                                 marker=trackers[i].get_marker())

                ax_l.plot(np.arange(0, time_steps), np.sqrt(np.mean(np.sum(error[i, :, :, :], axis=2), axis=0)),
                          color=trackers[i].get_color(), label=trackers[i].get_label(), marker=trackers[i].get_marker())

            ax_s.set_xticks(np.array(range(error.shape[-1])),
                            np.array(["$m_1$", "$m_2$", "$v$", "$\\alpha$", "$\\omega$"]))
            ax_s.set_ylabel('RMSE')

            axs_e[0, 0].set_xlabel('time step')
            axs_e[0, 0].set_ylabel('RMSE')

            axs_e[0, 1].set_xlabel('time step')
            axs_e[0, 1].set_ylabel('RMSE')

            axs_e[1, 0].set_xlabel('time step')
            axs_e[1, 0].set_ylabel('RMSE')

            axs_e[1, 1].set_xlabel('time step')
            axs_e[1, 1].set_ylabel('RMSE')

            fig_s.tight_layout()
            if self._display_legend_on_barplot:
                ax_s.legend(loc='upper right')
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

            fig_s.savefig(os.path.join(self._error_figure_path, 'individual_error_RMSE_' + self._scenario_name),
                          bbox_inches='tight')
            fig_e11.savefig(os.path.join(self._error_figure_path, 'position_RMSE_' + self._scenario_name),
                            bbox_inches='tight')
            fig_e12.savefig(os.path.join(self._error_figure_path, 'velocity_RMSE_' + self._scenario_name),
                            bbox_inches='tight')
            fig_e21.savefig(os.path.join(self._error_figure_path, 'orientation_RMSE_' + self._scenario_name),
                            bbox_inches='tight')
            fig_e22.savefig(os.path.join(self._error_figure_path, 'yawrate_RMSE_' + self._scenario_name),
                            bbox_inches='tight')
            fig_l.savefig(os.path.join(self._error_figure_path, 'overall_RMSE_' + self._scenario_name),
                          bbox_inches='tight')
            if not self._paper_mode:
                ax_s.set_title('Error over trajectory')
                axs_e[0, 0].set_title('Positional error over time')
                axs_e[0, 1].set_title('Velocity error over time')
                axs_e[1, 0].set_title('Orientation error over time')
                axs_e[1, 1].set_title('Yaw rate error over time')
                ax_l.set_title('RMSE')
                ax_e12.legend(loc='upper left')
                ax_e21.legend(loc='upper left')
                ax_e22.legend(loc='upper left')
            plt.show()
