import logging
import traceback

import collections as co

from crace.errors import CraceExecutionError
from crace.execution.target_evaluator import TargetEvaluator
from crace.execution.target_runner import parse_single_line_output, parse_single_line_output_with_target_evaluator

execution_logger = logging.getLogger("execution")
logger = logging.getLogger('asyncio')


class ExecutionPool:
    """
    The ExperimentPool is used to schedule the execution of experiments.
    """

    def __init__(self, chainMap, executioner, callback, experiments, worker_num: int, strict_pool: bool, target_evaluator=None, bound_out: bool=False):
        """
        Creates a new ExperimentPool

        :param queue: A queue, or queue-like object, that is used to store the experiments that need to be executed.
        :param executioner: The executioner that takes care of actually executing the instances.
        :param callback: A callback function that will be called after each experiment is done.
        :param target_evaluator: An optional parameter. The target evaluator that is to be used in evaluate_experiments.
        """
        self.waiting_queue = chainMap # storage all the submitted experiments
        #                               key is something related to the index of worker
        #                               values of each item are a list of experiment object
        self.queue_config = co.ChainMap()  # storage all the configuration ids of submitted experiments
        #                               is a mapping for waiting_queue
        #                               values of each item are a list of integers
        for key in range(1, worker_num+1):
            self.waiting_queue[key] = co.deque()
            self.queue_config[key] = co.deque()

        self.executioner = executioner
        self.experiment_finished_callback = callback
        self.experiments = experiments
        self.capping = experiments.capping
        self.output_parser = parse_single_line_output
        self.parallel = worker_num
        self.strict_pool = strict_pool
        if target_evaluator:
            self.evaluator = TargetEvaluator(target_evaluator)
            self.output_parser = parse_single_line_output_with_target_evaluator
        self.bound_out = bound_out
        self.is_open = True

        self.running_configs = []

    def close(self):
        """
        Closes the ExperimentPool. No new experiments can be submitted to it,
        all submitted experiments will be cancelled.
        """
        if not self.is_open:
            execution_logger.warning("The ExperimentPool is already closed.")
        self.is_open = False
        execution_logger.info("The ExperimentPool is closed.")
        self.waiting_queue.clear()
        self.queue_config.clear()

    def _submit_experiment(self, experiment, challenge=None):
        """
        Appends a single experiment to the queue of pending experiments.

        :param experiment: The experiment to be appended.
        :raises: A RuntimeError exception if the pool was already closed.
        """
        if not self.is_open:
            raise RuntimeError("The pool is closed.")

        # find the queue has more (configurations, experiments)
        if challenge:
            key = [k for k,v in sorted(self.queue_config.items(), key=lambda x: (len(set(x[1])), len(x[1])), reverse=True)][0]
            self.waiting_queue[key].appendleft(experiment)
            self.queue_config[key].appendleft(experiment.configuration_id)

        # find the queue has fewer (configurations, experiments)
        else:
            key = [k for k,v in sorted(self.queue_config.items(), key=lambda x: (len(set(x[1])), len(x[1])), reverse=False)][0]
            self.waiting_queue[key].append(experiment)
            self.queue_config[key].append(experiment.configuration_id)

    def submit_experiments(self, experiments, challenges: list=[]):
        """
        Appends an iterable of experiments to the queue of pending experiments.
        In order for the experiments to be retrieved, the executioner needs to call get_next_experiment.

        :param experiments: An iterable of experiments that will be added to the queue of pending experiments.
        :param challenge: A list of configurations having the most evaluations
        :raises: A RuntimeError exception if the pool was already closed.
        """
        # seperate and assign all submitted experiments based on config_id
        # exp_configs: storage the seperated experiments
        #   key: config_id
        #   value: experiments based on config_id
        if not self.is_open:
            raise RuntimeError("The pool is closed.")

        # separate new experiments based on their configuration_id
        exp_configs = {}
        for exp in experiments:
            if exp.configuration_id not in exp_configs.keys():
                exp_configs[exp.configuration_id] = []
            exp_configs[exp.configuration_id].append(exp)

        # update waiting queues based on the configuration ids
        for id, exps in exp_configs.items():
            # find the queue has more (configurations, experiments) if it is from a challenge
            if len(challenges) > 0 and id in challenges:
                key = [k for k,v in sorted(self.queue_config.items(),
                                           key=lambda x: (len(set(x[1])), len(x[1])), reverse=True)][0]
                # add new experiments to the top in the provided order
                self.waiting_queue[key].extendleft(exps[::-1])
                self.queue_config[key].extendleft([id]*len(exps))

            # find the queue has fewer (configurations, experiments) if it is not from a challenge
            else:
                key = [k for k,v in sorted(self.queue_config.items(),
                                           key=lambda x: (len(set(x[1])), len(x[1])), reverse=False)][0]
                # add new experiments to the bottom in the provided order
                self.waiting_queue[key].extend(exps)
                self.queue_config[key].extend([id]*len(exps))

            execution_logger.info("Submitting experiments with ids {} to queue {}".format([e.experiment_id for e in exps], key))

    def cancel_experiment(self, experiment):
        """
        Cancels the scheduled experiment.

        If it is still in the pending queue, then it will be removed from it without execution.
        If it was already in execution, then the executioner will be notified and should handle the cancellation.

        :param experiment: The experiment to be cancelled.
        """
        # find the queue includes experiment
        # from self.queue_config: based on config_id
        #   queue_config is a mapping for waiting_queue
        # if experiment is in waiting_queue (scheduled jobs), remove it
        # else: experiment is in running_job and it should be cancel via calling executioner
        found = False
        for k, v in self.waiting_queue.items():
            if experiment in v:
                self.queue_config[k].rotate(-self.waiting_queue[k].index(experiment))
                self.queue_config[k].popleft()
                self.queue_config[k].rotate(self.waiting_queue[k].index(experiment))
                self.waiting_queue[k].remove(experiment)
                execution_logger.info("Cancelling scheduled experiment %s (configuration %s).",
                                    experiment.experiment_id, experiment.configuration_id)
                return

        # if the provided experiment is not in the waiting queue
        # it should in the running list
        if not found:
            execution_logger.info("Cancelling running experiment %s (configuration %s).",
                                experiment.experiment_id, experiment.configuration_id)
            self.executioner.cancel_job(experiment)

    def cancel_experiments(self, configuration_ids, experiments):
        """
        Cancel a list of experiments
        """
        for experiment in experiments:
            self.cancel_experiment(experiment)

    def has_next_experiment(self):
        """
        Checks if there is currently an experiment available to be sent to a worker.

        If this method returns True, then the next call to get_next_experiment will return an experiment and not None.
        If this method returns False, then the next call to get_next_experiment will return None.

        :return: True, if there is an experiment available, False otherwise.
        """
        if not self.is_open:
            return False
        if any(self.waiting_queue[x] for x in self.waiting_queue.keys()):
            return True
        return False

    def get_next_experiment(self, worker=1):
        """
        Retrieves the first element of the pending experiments.

        :param worker: obtained free worker
        :return: Returns the first element of the pending experiments or None if there are no pending experiments.
        :raises: A RuntimeError if the pool was already closed.
        """
        # when the firstly selected waiting queue is empty, select the next configuration
        # from another queue with more (configs, exps)
        if not self.is_open:
            raise RuntimeError("The pool is closed.")

        # next_index: the index of exp should be selected in second queue, it is the first exp on different config.
        next_index = 0
        tries = self.parallel
        self.running_configs = self.get_running_config_ids()
        next_experiment = None

        # check the key(queue) of the next experiment until it is a good choice
        # or there is no good choice
        for _ in range(tries):
            if not self.waiting_queue[worker]:
                # if current sub-queue has no experiment, turn to another one having the most (configs, expes)
                worker = [k for k,v in sorted(self.queue_config.items(), key=lambda x: (len(set(x[1])), len(x[1])), reverse=True)][0]

            if (not self.strict_pool
                or self.waiting_queue[worker][next_index].configuration_id not in self.running_configs
                # if not capping/strict_pool, return the top of the queue(worker)
                # if capping/strict_pool, choice should be:
                #  a. FIXME(may cause error): exp from new instance or new config: budget == bound_max
                #  b. config differs from any of the running ones
                ):
                if next_index == 0:    # queue(worker) is a good choice
                    next_experiment = self.waiting_queue[worker].popleft()
                    self.queue_config[worker].popleft()
                else:
                    next_experiment = self.waiting_queue[worker][next_index]
                    self.waiting_queue[worker].remove(next_experiment)
                    self.queue_config[worker].rotate(-next_index)
                    self.queue_config[worker].popleft()
                    self.queue_config[worker].rotate(next_index)
                break

            else:
                # this is a capping problem
                # exps on workers must from different configs
                waiting_configs = [config for x in self.queue_config.values() for config in x]
                not_run = sorted(list(set(waiting_configs) - set(self.running_configs)))
                for k, v in self.queue_config.items():   # prefer to select the old config in list not_run
                    if not_run[0] in v:
                        worker = k
                        next_index = self.queue_config[worker].index(not_run[0])

        if next_experiment is not None:
            # if capping experiment has priori time budget, update the bound based on finished exps
            if self.capping and next_experiment.budget != next_experiment.bound_max:
                next_experiment = self.experiments.calculate_post_bounds(next_experiment)

            # update the running configuration list
            self.running_configs = self.get_running_config_ids() + [next_experiment.configuration_id]

        else:
            raise CraceExecutionError("Error next experiment: {}. ".format(next_experiment))
        return next_experiment

    def experiment_finished(self, experiment, full_line, cancelled, return_code, output_stdout, output_stderr, exp_time,
                            start_time=None, end_time=None):
        """
        This method is called by the executioner after it finishes the execution of one experiment.

        :param experiment: The experiment that just finished
        :param full_line: Experiment execution time
        :param cancelled: A boolean indicating if the experiment was cancelled.
        :param return_code: The return code of the target runner.
        :param output_stdout: The output from stdout as an utf-8 encoded string
        :param output_stderr: The output from stderr as an utf-8 encoded string
        :param exp_time: The time it took to execute the experiment
        :param start_time: The time in which the experiment execution started
        :param end_time: The time in which the experiment execution ended
        """
        execution_logger.debug("Experiment %s finished with return code %s and it was%s cancelled.",
                              experiment.experiment_id, return_code, "" if cancelled else " not")

        # if it was cancelled, then we don't need to do anything else
        if cancelled: return

        # target algorithm terminated with a non-zero exit code
        if return_code != 0:
            execution_logger.error("Error executing: %s. ", full_line)
            execution_logger.error("Error stdout: %s", output_stdout)
            execution_logger.error("Error stderr: %s", output_stderr)
            raise CraceExecutionError("Error executing: {} \nError stdout: {} \nError stderr: {} ".format(full_line,
                                                           output_stdout, output_stderr))

        # consume stdout to produce the result
        try:
            quality_score, time_score = self.output_parser(output_stdout)
        except (TypeError, ValueError):
            execution_logger.error("Error executing: %s", full_line)
            execution_logger.error("Error: output of experiment %d is not a number: %s " % (experiment.experiment_id, output_stdout))
            raise CraceExecutionError("Error executing: {}. Output is not a number: {}".format(full_line,
                                                           output_stdout))

        experiment.set_start_time(start_time)
        experiment.set_end_time(end_time)

        if time_score is None:
            # especially when obtaining only one result from the target algorithm
            time_score = end_time - start_time

        try:
            # if the experiment_finished_callback was set, then call it with the results
            if self.experiment_finished_callback:
                self.experiment_finished_callback(experiment, quality_score, time_score)

        except Exception as e:
            print("\nERROR: There was an error while executing crace callback: ")
            print(e)
            logger.error("There was an error while executing crace callback: %s", e)
            logger.error("Full traceback:\n%s", traceback.format_exc())
            raise CraceExecutionError(f"Error callback: experiment {experiment.experiment_id} "
                                      f"configuration {experiment.configuration_id} instance {experiment.instance_id}")

    def evaluate_experiments(self, experiments, alive_ids):
        """
        Runs the target evaluator on a set of experiments.

        :param experiments: The experiments to be evaluated
        :param alive_ids: A list of alive configuration ids
        :return: A list of tuples (experiment, quality) for all evaluated experiments.
        """
        evaluations = []
        for experiment in experiments:
            quality = self.evaluator.evaluate_experiment(experiment, alive_ids)
            evaluations.append((experiment, quality))
        return evaluations
    
    def get_running_experiment_ids(self):
        """
        Gets experiments ids of the running experiments
        """
        return self.executioner.get_running_experiment_ids()

    def get_running_config_ids(self):
        """
        Get configuration ids of the running experiments
        """
        return self.executioner.get_running_config_ids()
