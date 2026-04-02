import random
import asyncio
import logging

import crace.execution as execution
from crace.race.race import Race, RaceCheck
from crace.containers.scenario import Scenario
from crace.containers.instances import Instances
from crace.containers.parameters import Parameters
from crace.containers.crace_options import CraceOptions

asyncio_logger = logging.getLogger("asyncio")


class Crace:
    def __init__(self, scenario: Scenario):
        self.options = scenario.options
        self.parameters = scenario.parameters
        self.instances = scenario.instances
        self.initial_configurations = scenario.initial_configurations

        random.seed((self.options.seed.value * 7 - 1234567) % 7654321)

        if self.options.readLogsInCplot.value < 1: self.options.print_crace_header()
        if not self.options.readlogs.value:
            print_options(self.options, self.parameters, self.instances, full=True)

        self.race = Race(self.initial_configurations, self.instances, self.parameters, self.options)

        # create master execution handler
        if not self.options.readlogs.value:
            self.master, self._pool = execution.start_execution(self.options, self.race.race_callback, self.race.experiments)
            self.race.set_pool(self._pool)

    def recover(self, options: CraceOptions):
        pass

    def run(self):
        """
        Runs the configurator and returns the result of the race
        :return:
        """
        print("# Running crace configuration procedure")
        execution_logger = logging.getLogger('execution')
        loop = asyncio.get_event_loop()
        future = asyncio.ensure_future(self.race.asynchronous_race())
        asyncio_logger.debug("Ensured asynchronous_race future")
        loop.run_until_complete(future)
        asyncio_logger.debug("asynchronous_race future is complete")
        execution_logger.info(future.result())
        shutdown_future = asyncio.ensure_future(execution.stop_execution_async(self.master, self._pool))
        asyncio_logger.debug("Requesting stop of execution")
        loop.run_until_complete(shutdown_future)
        asyncio_logger.debug("Execution stopped")
        return future.result()

    def read_log_files(self):
        """
        Read log files
        """
        data = self.race.read_log_files()
        return data

    def get_elite_configurations(self):
        return self.race.get_elite_configurations()

class Check:
    def __init__(self, scenario: Scenario):
        self.options = scenario.options
        self.parameters = scenario.parameters
        self.instances = scenario.instances
        self.initial_configurations = scenario.initial_configurations

        random.seed((self.options.seed.value * 7 - 1234567) % 7654321)

        if self.options.readLogsInCplot.value < 1: self.options.print_crace_header()
        print_options(self.options, self.parameters, self.instances, full=False)

        self.race = RaceCheck(self.initial_configurations, self.instances, self.parameters, self.options)

        # create master execution handler
        self.master, self._pool = execution.start_execution(self.options, self.race.race_callback, self.race.experiments)
        self.race.set_pool(self._pool)

    def run(self):
        """
        Runs the configurator and returns the result of the race
        :return:
        """
        print("# Checking the provided scenario files..")
        loop = asyncio.get_event_loop()
        future = asyncio.ensure_future(self.race.asynchronous_race())
        loop.run_until_complete(future)
        shutdown_future = asyncio.ensure_future(execution.stop_execution_checking(self.master, self._pool))
        loop.run_until_complete(shutdown_future)
        return future.result()

def print_options(options: CraceOptions, parameters: Parameters, instances: Instances, full=True):
    # FIXME: check what we want to display here
    if full:
        options.print_all()
        print('#------------------------------------------------------------------------------')
    else:
        options.print_selects()
        print('#------------------------------------------------------------------------------')
