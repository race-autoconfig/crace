"""target_evaluator module

This module contains the following classes:
    * TargetEvaluatorBase - a base class to define custom target evaluators for crace
    * TargetEvaluator - the default target evaluator for crace
"""

import logging
import subprocess

from abc import ABC, abstractmethod

from crace.errors import CraceExecutionError
from crace.containers.experiments import ExperimentEntry

execution_logger = logging.getLogger("execution")


class TargetEvaluatorBase(ABC):
    """
    An abstract base class to define target evaluators for crace.
    """

    @abstractmethod
    def evaluate_experiment(self, experiment: ExperimentEntry, all_experiment_ids):
        """
        Execute the experiment.

        Will return the information about the process.

        :param experiment: The experiment to be executed.
        :param all_experiment_ids: The ids of all experiments for this instance.
        :return: A tuple (return_code, stdout, stderr, start_time, end_time)
        return_code is 0 if the execution was successful and can take any other integer value if it was not
        stdout and stderr are utf-8 encoded strings of the output produced by running the experiment
        start_time and end_time are the system times at which the target runner was started and ended
        """


class TargetEvaluator(TargetEvaluatorBase):
    """
    This is the default target evaluator of crace.
    """

    def __init__(self, target_evaluator_file: str):
        """
        Constructor for the target evaluator.

        :param target_evaluator_file: A string containing the path to the target evaluator executable.
        """
        self.target = target_evaluator_file

    def evaluate_experiment(self, experiment: ExperimentEntry, alive_ids):
        """
        Evaluate the experiment.

        Blocks until the evaluation is finished. Will return the information about the evaluation.
        This method raises an CraceExecutionError, if the target evaluator script returns a non-zero return code.

        :param experiment: The experiment to be executed.
        :param alive_ids: The ids of all alive configurations.
        :return: Returns stdout, an utf-8 encoded string of the output produced by the target executable
        """
        execution_logger.debug("Starting evaluation of experiment %s", experiment.experiment_id)
        configuration_id = experiment.configuration_id
        instance_id = experiment.instance_id

        # experiment.cmd_line[:4] are the first 4 entries of the cmd_line: configuration_id, instance_id, seed, instance
        cmd_args = self.target + " " + experiment.cmd_line + " " + str(len(alive_ids)) + \
                   " " + " ".join([str(x) for x in alive_ids])

        p = subprocess.Popen([cmd_args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        p.wait()
        stdout, stderr = p.communicate()

        if p.returncode != 0:
            execution_logger.error("Target evaluator did not exit with status 0.")
            execution_logger.error("stdout: {}".format(stdout))
            execution_logger.error("stderr: {}".format(stderr))
            raise CraceExecutionError("Target Evaluator exited with status {}".format(p.returncode))
        stdout = stdout.decode("utf-8")

        try:
            stdout = float(stdout)
        except:
            raise CraceExecutionError("Wrong type of output:" + stdout)

        return stdout
