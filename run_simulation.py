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


def prepare_trackers(initial_timestep, this_scenario_parameters, rng_seeder, real_data, runs, time_steps):
    trackers = []
    rng_seed = rng_seeder.integers(0, 9999)
    for mode in RadarOdometryTracker2D.get_valid_modes():
        trackers.append(RadarOdometryTracker2D(initial_timestep, this_scenario_parameters, mode=mode,
                                               rng=np.random.default_rng(rng_seed), runs=runs,  time_steps=time_steps,
                                               real_data=real_data))
    return trackers


def reset_trackers(trackers, rng_seeder):
    rng_seed = rng_seeder.integers(0, 9999)
    for tracker in trackers:
        tracker.reset(np.random.default_rng(rng_seed))


def main(scenario, plot_single_run, plot_final_trajectory, radar_config_path, paper_mode):
    this_scenario_parameters = get_scenario_parameters_from_scenario(scenario)
    rng_seeder = np.random.default_rng(512)
    runs = 1 if plot_single_run else this_scenario_parameters.get('runs')

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
    if plot_single_run:
        plotter.initialize_plotting_of_timesteps()

    trackers = prepare_trackers(initial_timestamp, this_scenario_parameters, rng_seeder, real_data, runs, time_steps)

    for r in tqdm(range(runs)):
        current_timestamp = initial_timestamp
        last_timestamp = initial_timestamp

        for i in range(time_steps):
            if real_data:
                current_timestamp, frame_data = np.load(os.path.join(data_path, f'{i:03d}.npy'), allow_pickle=True)
                points = radar_data_processor.identify_detections_in_polar_coordinates_from_frame_data(frame_data)
                true_state = this_scenario_parameters.get('final_ground_truth_state')
            else:
                points = simulator.simulate_timestep(current_timestamp-last_timestamp)
                true_state = simulator.get_state()

            ego_states = []
            for tracker in trackers:
                tracker.update(points, current_timestamp)
                ego_states.append(tracker.get_state())
                tracker.calculate_squared_error(true_state, r, i)

            if plot_single_run:
                plotter.plot_ego_pose_2d(trackers, radar_data_processor.get_max_range_m(),
                                         radar_data_processor.get_max_angle_degree(), angle_index=3,
                                         ground_truth=true_state, extent=this_scenario_parameters.get('estimate_area'))
                plotter.plot_2d_points_doppler(points, radar_data_processor.get_max_range_m(),
                                               radar_data_processor.get_max_angle_degree(), step=i,
                                               progressed_time=current_timestamp-last_timestamp,
                                               extent=this_scenario_parameters.get('measurement_area'))
            if not real_data:
                last_timestamp = current_timestamp
                current_timestamp += 1.0

        if not real_data:
            if plot_final_trajectory:
                simulator.plot_trajectory()
            simulator.reset()
        reset_trackers(trackers, rng_seeder)

        if r == (runs-1):
            plotter.plot_errors(trackers)


if __name__ == "__main__":
    scenario = SCENARIO_STRAIGHT
    plot_single_run = True
    plot_final_trajectory = False
    radar_config_path = './radar_config.json'
    paper_mode = True

    main(scenario, plot_single_run, plot_final_trajectory, radar_config_path, paper_mode)
