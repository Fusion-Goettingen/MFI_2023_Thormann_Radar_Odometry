from abc import ABC, abstractmethod


class AbstractSimulator(ABC):
    @abstractmethod
    def propagate(self, time_difference, step_id=None):
        pass

    @abstractmethod
    def generate_data(self):
        return None

    def simulate_timestep(self, time_difference, step_id=None):
        self.propagate(time_difference, step_id)
        return self.generate_data()
