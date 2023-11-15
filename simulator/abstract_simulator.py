from abc import ABC, abstractmethod


class AbstractSimulator(ABC):
    @abstractmethod
    def propagate(self, time_difference):
        pass

    @abstractmethod
    def generate_data(self):
        return None

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    def simulate_timestep(self, time_difference,):
        self.propagate(time_difference)
        return self.generate_data()
