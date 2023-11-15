import numpy as np

general_parameters = {
    'center_of_rotation_shift_local': np.array([0.2, 0.0]),
    'velocity_standard_deviation': 0.1,
    'yaw_rate_standard_deviation': 0.01,
    'clutter_doppler_standard_deviation': 1.0,
    'theta_standard_deviation': 0.05*np.pi,
    'range_standard_deviation': 0.05,
    'doppler_standard_deviation': 0.01,
}

simulation_parameters = {
    'prior': np.array([0.0, 0.0, 0.5, 0.0, 0.05]),
    'prior_covariance': np.diag([0.001, 0.001, 0.001, 0.001, 0.001])**2,
    'expected_number_of_measurements': 5,
    'area': np.array([-5, 45, -12.5, 12.5]),
    'time_steps': 25,
    'runs': 1000,
}
simulation_parameters.update(general_parameters)


real_world_parameters = {
    'prior': np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    'prior_covariance': np.diag([0.001, 0.001, 0.001, 0.001, 0.001])**2,
    'expected_number_of_clutter': 5,
    'runs': 100,
}
real_world_parameters.update(general_parameters)


scenario_no_clutter_parameters = {
    'expected_number_of_clutter': 0,
    'data_path': None,
}
scenario_no_clutter_parameters.update(simulation_parameters)


scenario_clutter_parameters = {
    'expected_number_of_clutter': 5,
    'data_path': None,
}
scenario_clutter_parameters.update(simulation_parameters)


scenario_straight_parameters = {
    'area': np.array([-0.5, 1.5, -1.0, 1.0]),
    'data_path': './data/straight_line_100cm/',
    'final_ground_truth_state': np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
}
scenario_straight_parameters.update(real_world_parameters)


scenario_turn_parameters = {
    'area': np.array([-1.0, 1.0, -1.5, 0.5]),
    'data_path': './data/only_turn_90/',
    'final_ground_truth_state': np.array([0.0, 0.0, 0.0, -0.5*np.pi, 0.0]),
}
scenario_turn_parameters.update(real_world_parameters)
