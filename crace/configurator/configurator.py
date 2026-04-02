from abc import ABC, abstractmethod


class Configurator(ABC):

    def __init__(self, scenario):
        self.scenario = scenario

    @abstractmethod
    def run(self):
        """
        Create and send experiments to work. Iterate until the end conditions
        are met
        :return:
        """
        pass

    @abstractmethod
    def recover(self):
        pass

    @abstractmethod
    def check(self):
        pass


