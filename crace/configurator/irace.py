import copy
from crace.containers import Scenario, CraceOptions


class Irace:
    def __init__(self, scenario: Scenario):
        self.options = scenario.options
        self.parameters = scenario.parameters
        self.instances = scenario.instances
        self.configurations = copy.deepcopy(scenario.initial_configurations)

        # calculate important variables

    def recover(self, options: CraceOptions):
        pass

    def search(self):
        pass

    def get_elite_configurations(self):
        pass
