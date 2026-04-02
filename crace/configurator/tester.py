import os
import time
import copy
import asyncio
import logging
import crace.execution as execution

from crace.utils import get_loop
from crace.errors import CraceError
from crace.containers.scenario import Scenario
from crace.containers.experiments import Experiments

asyncio_logger = logging.getLogger("asyncio")


class Tester:
    def __init__(self, scenario: Scenario):
        print("\n\n# Initializing tester ")
        self.options = scenario.options
        self.parameters = scenario.parameters
        self.instances = scenario.instances
        if scenario.initial_configurations is None:
            self.initial_configurations = None
        else:
            self.initial_configurations = copy.deepcopy(scenario.initial_configurations)
            self.configurations = self.initial_configurations.get_alive()

        self.target_eval = False
        # if self.options.targetEvaluator.value is not None:
        #     self.target_eval = True

        self.log_folder_test = os.path.join(self.options.logDir.value, "test")
        if not os.path.exists(self.log_folder_test):
            os.mkdir(self.log_folder_test)
        self.experiments = Experiments(log_folder=self.log_folder_test,
                                       budget_digits=self.options.boundDigits.value,
                                       log_level=self.options.logLevel.value)

        self.debug_level = self.options.debugLevel.value
        self._scheduled_jobs = []
        self._to_eval_jobs = []
        self.n_experiments = 0

        # self._result_future = asyncio.Future()
        loop = get_loop()
        self._result_future = loop.create_future()

        # create master execution handler
        self.master, self._pool = execution.start_execution(self.options, self.test_callback, self.experiments, True)

        self.clock = time.time()

    def do_test(self, config: dict = None):
        terminated = None
        if self.options.recoveryDir.is_set():
            terminated = self.check_status(config)
        if terminated:
            print('# Testing is finished in previous run.')
            return

        print("# Start testing ... ")

        if config is not None:
            self.configurations = config
        elif self.configurations is None:
            raise CraceError("There are no configurations assigned to test.")

        print("# Configurations: ", len(self.configurations))
        if self.debug_level >= 0:
            for co in self.configurations:
                print(f" {co.id}: {co.cmd}")

        print("# Instances: ", self.instances.tsize)
        if self.debug_level >= 2:
            for i in range(self.instances.tsize):
                print("  {}: {}".format(i+1, self.instances.tnames[i]))

        if self.options.testNbElites.value != len(self.configurations):
            self.configurations = self.configurations[0:self.options.testNbElites.value]
            print(f"# Test {self.options.testNbElites.value} configuration(s): {','.join([str(x.id) for x in self.configurations])}")
        if self.debug_level >= 2 and self.options.testNbElites.value > len(self.configurations):
            print(f"# Only {len(self.configurations)} elite configurations are returned.")

        asyncio_logger.debug("do_test")
        loop = asyncio.get_event_loop()
        future = asyncio.ensure_future(self.execute_test())
        asyncio_logger.debug("do_test: ensured future")
        loop.run_until_complete(future)
        asyncio_logger.debug("do_test: future is complete")
        asyncio_logger.info(future.result())
        shutdown_future = asyncio.ensure_future(execution.stop_execution_async(self.master, self._pool))
        asyncio_logger.debug("do_test: requesting stop")
        loop.run_until_complete(shutdown_future)
        asyncio_logger.debug("do_test: stopped")

        print("Stopped")

    async def execute_test(self):
        asyncio_logger.debug("execute_test")
        assert self.configurations is not None, "configurations must be provided"
        self._add_experiments(self.configurations, self.instances.tinstances, {0:self.options.boundMax.value})
        asyncio_logger.debug("execute_test: Before await")
        # do we need the following line?
        await self._result_future
        return "finished"

    def _add_experiments(self, configurations: list, instances: list, bound: dict):
        asyncio_logger.debug("_add_experiments: called")

        if configurations is dict:
            configurations = configurations.values()

        exp_ids, exps = self.experiments.new_experiments(configurations, instances, bound)
        if self.debug_level >= 2:
            print("# Adding test experiments config: {} \n# instances: {} \n# exp ids: {} ".format(
                [x.id for x in configurations],
                [x.instance_id for x in instances],
                exp_ids))
        self._pool.submit_experiments(exps)
        self._scheduled_jobs.extend(exp_ids)
        asyncio_logger.debug("_add_experiments: finished")

    def _complete_experiment(self, experiment, result, result_time):
        print("# Experiment: {:<7d} config: {:<7d} instance: {:<7d} result: {:<20f}time: {:.5f}".format(experiment.experiment_id,
                                                                                                        experiment.configuration_id,
                                                                                                        experiment.instance_id,
                                                                                                        result,result_time))
        if not self.target_eval:
            result_experiment = self.experiments.complete_experiment(
                experiment_id=experiment.experiment_id,
                budget=experiment.budget,
                budget_format=experiment.budget_format,
                solution_quality=result,
                exp_time=result_time,
                start_time=experiment.start_time,
                end_time=experiment.end_time)
        else:
            result_experiment = self.experiments.complete_experiment_time(
                experiment_id=experiment.experiment_id,
                exp_time=result_time,
                start_time=experiment.start_time,
                end_time=experiment.end_time)
            self._to_eval_jobs.append(experiment.experiment_id)

        if experiment.experiment_id in self._scheduled_jobs:
            index = self._scheduled_jobs.index(experiment.experiment_id)
            del self._scheduled_jobs[index]
        return result_experiment

    def _evaluate_experiment(self, experiment):
        """
        Call to target evaluator when an a set of experiments is done.

        :param experiment: experiment object
        :param solution_quality:

        :return: None
        """
        if self.target_eval:
            # FIXME: put this into a function
            exps = self.experiments.get_waiting_experiments_by_instance_id(experiment.instance_id)
            alive_ids = self.configurations.keys()
            evaluations = self._pool.evaluate_experiments(exps, alive_ids)
            for e, solution_quality in evaluations:
                result_experiment = self.experiments.complete_experiment_quality(e.experiment_id, solution_quality)
                index = self._to_eval_jobs.index(e.experiment_id)
                del self._to_eval_jobs[index]

    def _is_instance_fully_executed(self, instance_id):
        """
        Checks if an instance is fully executed. A fully executed instance is defined as
        an instance for which there are no experiments scheduled.

        :param instance_id: instance id

        :return: Boolean indicating if instance is complete
        """
        for experiment_id in self._scheduled_jobs:
            experiment = self.experiments.get_experiment_by_id(experiment_id, as_dataframe=False)
            if instance_id == experiment.instance_id:
                return False
        return True

    def test_callback(self, experiment, solution_quality, solution_time):
        complete_experiment = self._complete_experiment(experiment, solution_quality, solution_time)
        self.n_experiments = self.n_experiments + 1

        if self._is_instance_fully_executed(experiment.instance_id):
            self._evaluate_experiment(experiment)

        if len(self._scheduled_jobs) == 0:
            self._result_future.set_result("Future is done!")

    def check_status(self, config):
        """
        check if test is finished already
        """
        tsize = self.instances.tsize
        n_elite = self.options.testNbElites.value
        exps_fin = os.path.join(self.options.recoveryDir.value, 'test/exps_fin.log')
        if os.path.exists(exps_fin) and os.path.getsize(exps_fin) > 0:
            with open(exps_fin, newline='') as f:
                line_count = sum(1 for line in f)
            return True if line_count == tsize * n_elite else False
        else:
            return False