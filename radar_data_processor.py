import json
import numpy as np
from scipy import signal

SPEED_OF_LIGHT = 299792458


def calculate_range_doppler_image(raw_data: np.ndarray):
    normalized_data = raw_data - np.average(raw_data)
    range_fft_per_chirp = calculate_range_frequency_spectrum(normalized_data)
    return calculate_range_doppler_image_from_range_frequency_spectrum(range_fft_per_chirp)


def calculate_range_frequency_spectrum(normalized_data):
    range_fft_per_chirp = calculate_windowed_fft_transform(normalized_data)
    # ignore negative spectrum
    return 2.0 * range_fft_per_chirp[:, :normalized_data.shape[1]]


def calculate_range_doppler_image_from_range_frequency_spectrum(range_fft_per_chirp):
    range_fft_per_chirp_transpose = np.transpose(range_fft_per_chirp)
    fft_range_doppler = calculate_windowed_fft_transform(range_fft_per_chirp_transpose)
    return np.fft.fftshift(fft_range_doppler, (1,))


def calculate_windowed_fft_transform(data):
    sample_size = data.shape[1]
    windowed_data = np.multiply(data, signal.blackmanharris(sample_size).reshape(1, sample_size))
    windowed_padded_data = np.pad(windowed_data, ((0, 0), (0, sample_size)), 'constant')
    return np.fft.fft(windowed_padded_data) / sample_size


class RadarDataProcessor:
    def __init__(self, radar_data_parameter_file):
        with open(radar_data_parameter_file) as file:
            radar_data_parameters: dict = json.load(file)
        # radar settings
        self._number_of_chirps = radar_data_parameters.get('number_of_chirps')
        self._number_of_samples_per_chirp = radar_data_parameters.get('number_of_samples_per_chirp')
        self._number_of_receiver_antennas = radar_data_parameters.get('number_of_receiver_antennas')
        self._vertical_antenna_is_active = bool(radar_data_parameters.get('vertical_antenna_is_active'))
        self._bandwidth = radar_data_parameters.get('bandwidth')
        self._post_chirp_delay = radar_data_parameters.get('post_chirp_delay')

        # calculate derived parameters
        self._max_range_m = SPEED_OF_LIGHT*0.5*self._number_of_samples_per_chirp*0.5/self._bandwidth
        self._max_speed = ((SPEED_OF_LIGHT / (58 * 1e9 + 0.5*self._bandwidth)) * 0.25
                           / (self._post_chirp_delay + 0.08 * 1e-3))

        # parameters for angle estimation
        self._max_angle_degree = radar_data_parameters.get('max_angle_degree')
        self._number_of_beams = radar_data_parameters.get('number_of_beams')
        self._receiver_antenna_map_horizontal = radar_data_parameters.get('receiver_antenna_map_horizontal')
        self._receiver_antenna_map_vertical = radar_data_parameters.get('receiver_antenna_map_vertical')
        # assumes spacing of 0.5*lambda between antenna
        self._weights_channel_beam = np.array([np.exp(1j * 2 * np.pi * i * 0.5 * np.sin(np.deg2rad(np.linspace(-self._max_angle_degree,
                                                                                                               self._max_angle_degree,
                                                                                                               self._number_of_beams))))
                                               for i in range(self._number_of_receiver_antennas-self._vertical_antenna_is_active)])

        # parameters for identifying detections in the radar data
        self._cfar_window_for_velocity = radar_data_parameters.get('cfar_window_for_velocity')
        self._cfar_window_for_range = radar_data_parameters.get('cfar_window_for_range')
        self._cfar_scaling = radar_data_parameters.get('cfar_scaling')
        self._minimum_detection_range_m = radar_data_parameters.get('minimum_detection_range_m')

        # for final filtering
        self._negative_log_intensity_cutoff = radar_data_parameters.get('negative_log_intensity_cutoff')

    def get_max_range_m(self):
        return self._max_range_m

    def get_max_angle_degree(self):
        return self._max_angle_degree

    def identify_detections_in_polar_coordinates_from_frame_data(self, frame_data):
        radar_frame_range_velocity_channel = self.__calculate_radar_frame_per_channel_from_frame_data(frame_data)
        return self.__identify_detections_in_polar_coordinates(radar_frame_range_velocity_channel)

    def __calculate_radar_frame_per_channel_from_frame_data(self, frame_data):
        radar_frame_range_velocity_channel = np.zeros((self._number_of_samples_per_chirp, 2*self._number_of_chirps,
                                                       self._number_of_receiver_antennas), dtype=complex)
        for i in range(self._number_of_receiver_antennas):
            radar_frame_range_velocity_channel[:, :, i] = calculate_range_doppler_image(frame_data[i])
        return radar_frame_range_velocity_channel

    def __identify_detections_in_polar_coordinates(self, radar_frame_range_velocity_channel, keep_elevation=False):
        detection_intensities_per_channel, detection_ranges, detection_dopplers\
            = self.__identify_detections_intensity_per_channel_range_and_doppler_using_cfar(radar_frame_range_velocity_channel)
        detections_in_polar = np.array([
            self.__identify_horizontal_angles_of_detections(detection_intensities_per_channel),
            self.__identify_vertical_angles_of_detections(detection_intensities_per_channel),
            detection_ranges,
            detection_dopplers,
            np.mean(abs(detection_intensities_per_channel), axis=0),
            ]).T
        filtered_detections_in_polar = self.__filter_detections_by_intensity(detections_in_polar)
        return filtered_detections_in_polar if keep_elevation else filtered_detections_in_polar[:, [0, 2, 3, 4]]

    def __identify_detections_intensity_per_channel_range_and_doppler_using_cfar(self, radar_frame_range_velocity_channel):
        abs_data = np.sum(abs(radar_frame_range_velocity_channel), axis=2)
        # identify Doppler detection per range
        max_doppler_ids = np.argmax(abs_data, axis=1)
        max_doppler_intensity = np.max(abs_data, axis=1)
        # filter points based on intensity
        point_mask = max_doppler_intensity >= self._cfar_scaling * np.array([np.mean(np.hstack([
            abs_data[np.maximum(int(i-0.5*self._cfar_window_for_range), 0):np.minimum(int(i+0.5*self._cfar_window_for_range),
                                                                                      self._number_of_samples_per_chirp-1),
            np.maximum(int(max_doppler_ids[i]-0.5*self._cfar_window_for_velocity), 0):(max_doppler_ids[i]-3)],
            abs_data[np.maximum(int(i-0.5*self._cfar_window_for_range), 0):np.minimum(int(i+0.5*self._cfar_window_for_range),
                                                                                      self._number_of_samples_per_chirp-1),
            (max_doppler_ids[i]+4):np.minimum(int(max_doppler_ids[i]+0.5*self._cfar_window_for_velocity), 2*self._number_of_chirps)]
        ])) for i in range(len(max_doppler_intensity))])
        point_mask[:int(self._minimum_detection_range_m/self._max_range_m*len(point_mask))] = False  # remove too close detections
        max_doppler_ids = max_doppler_ids[point_mask]

        detection_intensities_per_channel = np.take_along_axis(radar_frame_range_velocity_channel[point_mask], max_doppler_ids[:, None, None], axis=1).reshape(-1, self._number_of_receiver_antennas).T
        detection_ranges = np.arange(0, self._number_of_samples_per_chirp)[point_mask] * self._max_range_m / self._number_of_samples_per_chirp
        detection_dopplers = max_doppler_ids * self._max_speed / self._number_of_chirps - self._max_speed
        return detection_intensities_per_channel, detection_ranges, detection_dopplers

    def __identify_horizontal_angles_of_detections(self, detection_intensities_per_channel):
        return self.__calculate_detected_angle_per_range(detection_intensities_per_channel, self._receiver_antenna_map_horizontal)

    def __identify_vertical_angles_of_detections(self, detection_intensities_per_channel):
        if self._vertical_antenna_is_active:
            return self.__calculate_detected_angle_per_range(detection_intensities_per_channel, self._receiver_antenna_map_vertical)
        else:
            return np.zeros(detection_intensities_per_channel.shape[1])

    def __calculate_detected_angle_per_range(self, detection_intensities_per_channel, antenna_map):
        beam_intensities_per_range = np.sqrt(abs(np.transpose(self._weights_channel_beam[:len(antenna_map)])
                                                 @ detection_intensities_per_channel[antenna_map])**2
                                             / len(antenna_map)).T
        beam_ids_per_range = np.argmax(beam_intensities_per_range, axis=1)
        return np.deg2rad(beam_ids_per_range / self._number_of_beams * 2 * self._max_angle_degree - self._max_angle_degree)

    def __filter_detections_by_intensity(self, detections_in_polar):
        intensities = detections_in_polar[:, -1]
        return detections_in_polar[-np.log(intensities) < self._negative_log_intensity_cutoff]
