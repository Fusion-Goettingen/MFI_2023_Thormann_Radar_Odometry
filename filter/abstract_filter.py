from abc import ABC, abstractmethod

import numpy as np


class AbstractFilter(ABC):
    @abstractmethod
    def get_state(self):
        """
        :return: current state
        """

    @abstractmethod
    def update(self, frame_data, time_stamp):
        """
        Update the state with the current measurements (and predict if necessary)
        :param frame_data:  current frame as antennas x chirps x samples per chirp
        :param time_stamp:  time stamp of current frame
        :return:
        """
        return np.zeros((0, 4))
