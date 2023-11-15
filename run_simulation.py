import numpy as np
from tqdm import tqdm
import os
import fnmatch

from simulator.simulator_2d import Simulator2D
from filter.radar_odometry_tracker_2d import RadarOdometryTracker2D
from visualizer.plotter import Plotter
from radar_data_processor import RadarDataProcessor
import scenario_parameters

SCENARIO_NO_CLUTTER = 0
SCENARIO_CLUTTER = 1
SCENARIO_STRAIGHT = 2
SCENARIO_TURN = 3
SCENARIO_NAME = ['noClutter', 'clutter', 'realStraight', 'realTurn']


def get_scenario_parameters_from_scenario(scenario):
    if scenario == SCENARIO_NO_CLUTTER:
        this_scenario_parameters = scenario_parameters.scenario_no_clutter_parameters
    elif scenario == SCENARIO_CLUTTER:
        this_scenario_parameters = scenario_parameters.scenario_clutter_parameters
    elif scenario == SCENARIO_STRAIGHT:
        this_scenario_parameters = scenario_parameters.scenario_straight_parameters
    elif scenario == SCENARIO_TURN:
        this_scenario_parameters = scenario_parameters.scenario_turn_parameters
    else:
        raise ValueError('Invalid scenario ID')

    return this_scenario_parameters


def prepare_trackers(initial_timestep, this_scenario_parameters, rng_seed, real_data):
    trackers = []
    for mode in RadarOdometryTracker2D.get_valid_modes():
        trackers.append(RadarOdometryTracker2D(initial_timestep, this_scenario_parameters, mode=mode,
                                               rng=np.random.default_rng(rng_seed), real_data=real_data))
    return trackers


def calculate_squared_error(true_state, ego_state, angle_index):
    return calculate_error(true_state, ego_state, angle_index)**2


def calculate_error(true_state, ego_state, angle_index):
    error = true_state - ego_state
    error[angle_index] = ((error[angle_index] + np.pi) % (2.0*np.pi)) - np.pi
    return error


def main(scenario, plot_single_timestep, plot_final_trajectory, radar_config_path, paper_mode):
    this_scenario_parameters = get_scenario_parameters_from_scenario(scenario)
    rng_seeder = np.random.default_rng(512)
    runs = 1 if plot_single_timestep else this_scenario_parameters.get('runs')

    radar_data_processor = RadarDataProcessor(radar_config_path)
    simulator = Simulator2D(np.random.default_rng(512), this_scenario_parameters)

    data_path = this_scenario_parameters.get('data_path')
    if data_path is not None:
        real_data = True
        time_steps = len(fnmatch.filter(os.listdir(data_path), '*.npy'))
        initial_timestamp, _ = np.load(os.path.join(data_path, f'{0:03d}.npy'), allow_pickle=True)
    else:
        real_data = False
        time_steps = this_scenario_parameters.get('time_steps')
        initial_timestamp = 0.0

    plotter = Plotter(real_data, paper_mode, SCENARIO_NAME[scenario],
                      display_legend_on_barplot=(scenario in [SCENARIO_NO_CLUTTER, SCENARIO_STRAIGHT]))
    if plot_single_timestep:
        plotter.initialize_plotting_of_timesteps()
    error = np.zeros((len(RadarOdometryTracker2D.get_valid_modes()), runs, time_steps, 5))

    for r in tqdm(range(runs)):
        rng_seed = rng_seeder.integers(0, 9999)
        current_timestamp = initial_timestamp
        last_timestamp = initial_timestamp

        trackers = prepare_trackers(last_timestamp, this_scenario_parameters, rng_seed, real_data)

        for i in range(time_steps):
            if real_data:
                current_timestamp, frame_data = np.load(os.path.join(data_path, f'{i:03d}.npy'), allow_pickle=True)
                points = radar_data_processor.identify_detections_in_polar_coordinates_from_frame_data(frame_data)
                true_state = this_scenario_parameters.get('final_ground_truth_state')
            else:
                points = simulator.simulate_timestep(current_timestamp-last_timestamp)
                true_state = simulator.get_state()

            ego_states = []
            current_errors = []
            for tracker in trackers:
                tracker.update(points, current_timestamp)
                ego_states.append(tracker.get_state())
                current_errors.append(calculate_squared_error(true_state, tracker.get_state(), angle_index=3))
            error[:, r, i] = np.array(current_errors)

            if plot_single_timestep:
                plotter.plot_ego_pose_2d(trackers, radar_data_processor.get_max_range_m(),
                                         radar_data_processor.get_max_angle_degree(), angle_index=3,
                                         ground_truth=true_state, extent=this_scenario_parameters.get('area'))
                plotter.plot_2d_points_doppler(points, radar_data_processor.get_max_range_m(),
                                               radar_data_processor.get_max_angle_degree(), step=i,
                                               progressed_time=current_timestamp-last_timestamp,
                                               extent=this_scenario_parameters.get('area'))
            if not real_data:
                last_timestamp = current_timestamp
                current_timestamp += 1.0

        if plot_final_trajectory:
            simulator.plot_trajectory()
        simulator.reset()

        if r == (runs-1):
            plotter.plot_errors(error, trackers)


if __name__ == "__main__":
    scenario = SCENARIO_TURN
    plot_single_timestep = False
    plot_final_trajectory = False
    radar_config_path = './radar_config.json'
    paper_mode = True

    main(scenario, plot_single_timestep, plot_final_trajectory, radar_config_path, paper_mode)
