"""concurrent module

This module contains classes for the different execution masters and workers using concurrency.

This module contains the following classes:
    * ConcurrentExecutionMaster - an execution master class for crace using concurrency
    * ConcurrentExecutionWorker - an execution worker class for crace using concurrency
"""

import sys
import logging
import asyncio
import concurrent.futures

from crace.errors import CraceExecutionError
from crace.execution.target_runner import TargetRunner
from crace.execution.execution_base import ExecutionMasterBase, ExecutionWorkerBase

cancel = []

execution_logger = logging.getLogger("execution")


class ConcurrentExecutionMaster(ExecutionMasterBase):
    """
    The ConcurrentExecutionMaster implements the behavior of the execution master for crace using concurrency.

    That is, the process waits for an experiment and sends it to an available worker.
    Once the worker finishes execution of the job, it will send the results back to the master.
    All communication is handled over the concurrency module.
    """

    def __init__(self, number_workers, exec_cmd, target_runner_file, max_retries, debug_level=0):
        execution_logger.debug("Starting execution master.")
        super().__init__()

        self.debug_level = debug_level
        # Define a set of workers and discard the main process
        self.number_workers = number_workers
        self.target_runner_file = target_runner_file

        self.workers = set()
        self.process_executor = concurrent.futures.ProcessPoolExecutor()

        for _ in range(number_workers):
            target_runner = TargetRunner(exec_cmd, self.target_runner_file, debug_level)
            worker = ConcurrentExecutionWorker(target_runner, max_retries, self.process_executor, debug_level)
            self._add_worker(worker)
        # this is a dictionary that keeps the experiment_id as key and a tuple (worker_id, experiment) as key
        self.running_jobs = {}

    def get_running_experiment_ids(self):
        """
        Gets experiments ids of the running experiments
        """
        return list(self.running_jobs.keys())

    def get_running_config_ids(self):
        """
        Gets config ids of the running experiments
        """
        config_ids = []
        for x in self.running_jobs.keys():
            _, exp = self.running_jobs[x]
            config_ids.append(exp.configuration_id)
        return config_ids

    def _has_free_worker(self):
        """
        Checks if there is currently a worker available to take the next experiment.

        If this method returns True, then the next call to _get_free_worker will return a reference and not None.
        If this method returns False, then the next call to _get_free_worker will return None.

        :return: True, if there is a worker available, False otherwise.
        """
        if self.workers:
            return True
        return False

    def _get_free_worker(self):
        """
        Returns a worker object, that is currently available to take the next experiment.

        Returns None if there is no free worker currently.

        :return: A worker that is currently available.
                 None, if no worker is available.
        """
        if self.workers:
            return self.workers.pop()
        return None

    def _add_worker(self, worker):
        self.workers.add(worker)

    def _send_job_to_worker(self, experiment, worker):
        """
        This method passes an experiment to a worker for execution.

        It does not return the score or any information about the execution status.

        :param experiment: The experiment to be executed. It should be passed as the namedtuple Experiment.
        :param worker: The worker that the experiment is executed on.
        :return: True if the job was sent
        """
        global cancel
        if experiment.experiment_id not in cancel:
            experiment_id = experiment.experiment_id
            execution_logger.debug("Master sending experiment %s to worker", experiment_id)
            self.running_jobs[experiment_id] = (worker, experiment)
            worker.execute_experiment(experiment, self.job_finished)
            return True
        else:
            return False

    def cancel_job(self, experiment):
        """
        Cancel the execution of this experiment.

        :param experiment: The experiment to be cancelled.
        :return: None
        """
        try:
            worker = self.running_jobs[experiment.experiment_id][0]
            worker.cancel_job()
            del self.running_jobs[experiment.experiment_id]
            execution_logger.debug("Master cancelled experiment %s", experiment.experiment_id)
        except KeyError:
            global cancel
            cancel.append(experiment.experiment_id)

    def job_finished(self, worker, experiment, job_report):
        """
        Method called when a job is finished from a worker.

        :param worker: The worker that executed the job (in the form of the MPI rank)
        :param experiment: The experiment that was executed by the worker
        :param job_report: The additional information that the worker provided about the execution of the job.
                           This contains the following information:
                           * (bool) if the job was cancelled
                           * (int) the return code of the last execution of the target runner
                           * (str) the stdout of the last execution of the target runner
                           * (str) the stderr of the last execution of the target runner
                           * (time) start time of the experiment
                           * (time) end time of the experiment
                           * (time) the system time duration of the last execution of the target runner
                           * (int) the number of times the target runner was retried
        :return: None
        """
        # process the job_report
        full_line = job_report[0]
        cancelled = job_report[1]
        return_code = job_report[2]
        output_stdout = job_report[3]
        output_stderr = job_report[4]
        start_time = job_report[5]
        end_time = job_report[6]
        target_runner_time = job_report[7]
        num_retries = job_report[8]

        execution_logger.debug("Master received finished experiment %s", experiment.experiment_id)
        execution_logger.debug("The experiment was%s cancelled. The target runner had to retry %s times. "
                               "The last return code was %s", "" if cancelled else " not", num_retries, return_code)

        if self.debug_level >= 6:
            print("# Experiment {} finished in master".format(experiment.experiment_id))

        # return worker to available workers
        self.workers.add(worker)
        try:
            self.experiment_pool.experiment_finished(experiment, full_line, cancelled, return_code, output_stdout,
                                                     output_stderr, target_runner_time, start_time, end_time)
            if experiment.experiment_id in self.running_jobs.keys():
                del self.running_jobs[experiment.experiment_id]

        except Exception as e:
            print("\nERROR: There was an error while executing job: ")
            print(e)
            raise e

    def shutdown_workers(self):
        """
        Gracefully shuts down all workers.

        At the moment, does not do anything.

        :return: None
        """
        # TODO: Is there something we have to do here?


class ConcurrentExecutionWorker(ExecutionWorkerBase):
    """
    The ConcurrentExecutionWorker implements the behavior of the worker processes for crace using concurrency.
    """
    def __init__(self, target, max_retries, pool_executor, debug_level=0):
        super().__init__(target, max_retries)
        self.debug_level = debug_level
        self.pool_executor = pool_executor
        self.experiment_finished_callback = None
        self.is_cancelled = False
        self.current_id = None

    async def run_job(self, experiment):
        """
        Asynchronously executes an experiment.

        This method will retry the execution, if it failed less than self.max_retries times.

        :param experiment: The experiment to be executed
        :return: A tuple including the following information:
                 * return_code: the return code of the last execution of the target runner
                 * stdout: the stdout of the last execution of the target runner
                 * stderr: the stderr of the last execution of the target runner
                 * target_runner_time: the duration of the last execution of the target runner
                 * num_retries: the number of retries
        """
        execution_logger.debug("Worker to execute experiment %s", experiment.experiment_id)
        self.current_id = experiment.experiment_id

        current_retries = 0

        full_line, return_code, stdout, stderr, start_time, end_time = await self.target.run_experiment(experiment)

        # retry execution if it's terminated with a non-zero exit code
        while not self.is_cancelled and return_code != 0 and return_code != 15 and current_retries < self.max_retries:
            current_retries += 1
            if self.debug_level >=1:
                print("# Retry {} experiment {}".format(current_retries,experiment.experiment_id))
            full_line, return_code, stdout, stderr, start_time, end_time = await self.target.run_experiment(experiment)

        # raise when it's still terminated with a non-zero exit code
        # after retries
        if current_retries >= self.max_retries and self.max_retries > 0:
            print("\nERROR: There was an error while executing the target algorithm")
            execution_logger.error("Experiment {} exceeded the number of retries".format(experiment.experiment_id))
            execution_logger.error(f"Error executing: {full_line}\nOutput: {stdout}\nError: {stderr}")
            raise CraceExecutionError("Error: Experiment {} exceeded the number of retries".format(experiment.experiment_id))

        total_time = end_time - start_time

        execution_logger.debug("Worker finished execution of experiment %s with return code %s",
                               experiment.experiment_id, return_code)

        # Solve results via race_callback
        try:
            self.experiment_finished_callback(self, experiment,
                                              (full_line, self.is_cancelled, return_code, stdout, stderr,
                                               start_time, end_time, total_time, current_retries))
        except Exception:
            sys.tracebacklimit = 0
            raise

    def execute_experiment(self, experiment, callback):
        """
        Schedule the experiment for execution and return immediately.

        :param experiment: The experiment to be executed.
        :param callback: The callback that will be called after execution of the experiment is finished.
        :return: None
        """
        self.experiment_finished_callback = callback
        self.is_cancelled = False
        # FIXME: difficult to catch errors here
        # print error and exit instead of raising in the neareast lower level
        task = asyncio.get_event_loop().create_task(self.run_job(experiment))
        task.add_done_callback(lambda t: self._handle_task_result(t, experiment))

    @staticmethod
    def _handle_task_result(task, experiment):
        """
        Raise error when a task is failed
        
        :param task: current task
        :param experiment: current experiment
        """
        try:
            task.result()
        except Exception as e:
            loop = asyncio.get_running_loop()
            loop.call_soon(loop.stop)
            raise e

    def cancel_job(self):
        """
        Cancel the currently running experiment.

        If no experiment is currently running, this method has no effect.

        :return: None
        """
        execution_logger.debug("Worker to cancel experiment %s", self.current_id)
        self.target.cancel_experiment()
        self.is_cancelled = True
