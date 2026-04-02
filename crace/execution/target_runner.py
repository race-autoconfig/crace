"""target_runner module

This module contains the following classes:
    * TargetRunnerBase - a base class to define custom target runners for crace
    * TargetRunner - the default target runner for crace
"""
import time
import math
import asyncio
import logging

from abc import ABC, abstractmethod
from crace.errors import CraceExecutionError
from crace.containers.experiments import ExperimentEntry

execution_logger = logging.getLogger("execution")


def parse_single_line_output(stdout: str):
    """
    Parses the output of stdout of the target runner.

    This method expects a single line as stdout and will fail if more than one line is provided.

    :param stdout: The output of the target runner.
    :return:
    """
    stdout = stdout.strip()
    if "\n" in stdout:
        execution_logger.warning("The target runner provided more than one line.")

    output = stdout.split()
    if len(output) > 2 or len(output) < 1:
        raise ValueError("Output: ", stdout, " is not correct.")

    if len(output) == 2:
        quality_string, time_string = stdout.split()
        if "inf" in quality_string or "Inf" in quality_string:
            quality = math.inf
        else:
            quality = float(quality_string)
        reported_time = float(time_string)

    else:
        if "inf" in stdout or "Inf" in stdout:
            quality = math.inf
        else:
            quality = float(stdout)
        reported_time = None
    return quality, reported_time


def parse_single_line_output_with_target_evaluator(stdout: str):
    """
    Parses the output of stdout of the target runner, when a target evaluator is also present.

    This method expects a single line as stdout and will fail if more than one line is provided.

    :param stdout: The output of the target runner.
    :return: quality, time
    """
    stdout = stdout.strip()

    if "\n" in stdout:
        execution_logger.warning("The target runner provided more than one line.")

    output = stdout.split()

    if len(output) > 2 or len(output) < 1:
        raise ValueError("Output: ", stdout, " is not correct.")

    if len(output) == 2:
        quality_string, time_string = stdout.split()
        if "inf" in quality_string or "Inf" in quality_string:
            quality = math.inf
        else:
            quality = float(quality_string)
        reported_time = float(time_string)

    else:
        quality = float(stdout)
        reported_time = None
    return quality, reported_time


class TargetRunnerBase(ABC):
    """
    An abstract base class to define target runners for crace.
    """

    @abstractmethod
    async def run_experiment(self, experiment: ExperimentEntry):
        """
        Execute the experiment.

        Will return the information about the process.

        :param experiment: The experiment to be executed.
        :return: A tuple (return_code, stdout, stderr, start_time, end_time)
        return_code is 0 if the execution was successful and can take any other integer value if it was not
        stdout and stderr are utf-8 encoded strings of the output produced by running the experiment
        start_time and end_time are the system times at which the target runner was started and ended
        """

    @abstractmethod
    def cancel_experiment(self):
        """
        Cancel the currently running experiment.

        If an experiment is currently running, it will be cancelled and :meth:`run_experiment` will return with a
        non-zero return_code.

        If no experiment is currently running, this method will not have any effect.

        :return: None
        """


class TargetRunner(TargetRunnerBase):
    """
    This is the default target runner of crace.

    It launches an asynchronous subprocess to run the experiment.
    """

    def __init__(self, exec_cmd: str, target_runner_file: str, debug_level=0):
        """
        Initialize the target runner.

        :param target_runner_file: A string containing the path (relative or absolute) to the target executable.
        """
        self.exec_cmd = exec_cmd
        self.target = target_runner_file
        self._process = None
        self._experiment = None
        self._is_done = False
        self.debug_level = debug_level

    async def run_experiment(self, experiment: ExperimentEntry):
        """
        Execute the experiment in an asynchronous subprocess.

        Will return the information about the process.

        :param experiment: The experiment to be executed.
        :return: A tuple (return_code, stdout, stderr, start_time, end_time)
        return_code is the return code of the target executable
        stdout and stderr are utf-8 encoded strings of the output produced by the target executable
        start_time and end_time are the system times at which the target runner was started and ended
        """
        # prepare the execution
        self._experiment = experiment
        command_line = experiment.get_exec_line()

        # execute the experiment
        execution_logger.debug("# Starting execution of experiment %s", self._experiment.experiment_id)
        execution_logger.debug("# Executing exp %s: %s", str(experiment.experiment_id), self.target + " " + command_line)
        start_time = time.time()

        if self.exec_cmd is not None:
            full_line = self.exec_cmd + " " + self.target + " " + command_line
            self._process = await asyncio.create_subprocess_exec(self.exec_cmd, self.target, *command_line.split(),
                                                                stdout=asyncio.subprocess.PIPE,
                                                                stderr=asyncio.subprocess.PIPE)
        else:
            full_line = self.target + " " + command_line
            self._process = await asyncio.create_subprocess_exec(self.target, *command_line.split(),
                                                                stdout=asyncio.subprocess.PIPE,
                                                                stderr=asyncio.subprocess.PIPE)

        stdout_data, stderr_data = await self._process.communicate()
        self._is_done = True
        end_time = time.time()
        execution_logger.debug("Finished execution of experiment %s", self._experiment.experiment_id)

        # collect the results
        return_code = self._process.returncode
        stdout = stdout_data.decode("utf-8")
        stderr = stderr_data.decode("utf-8")

        execution_logger.debug("Return code of the experiment was %s", return_code)
        if self.debug_level >= 4:
            if return_code == 0:
                print("# Result:", stdout, " : ", self.target, " ", command_line)
            else:
                print("# Error:", stderr, " : ", self.target, " ", command_line)

        # reset the data related to this experiment
        self._process = None
        self._experiment = None

        # return the result of the experiment
        return full_line, return_code, stdout, stderr, start_time, end_time

    def cancel_experiment(self):
        """
        Cancel the currently running experiment.

        If an experiment is currently running, it will be cancelled and :meth:`TargetRunner.run_experiment` will return
        with a non-zero return_code.

        If no experiment is currently running, this method will not have any effect.

        :return: None
        """
        # If the process is already finished, dont do nothing
        if self._process is None or self._is_done:
            return

        execution_logger.debug("TargetRunner is trying to terminate its process %s running experiment %s",
                               self._process.pid,
                               self._experiment.experiment_id)
        try:
            self._process.terminate()
        except AttributeError:
            # TODO implement error handling
            CraceExecutionError("There was an exception terminating the process")
            pass
