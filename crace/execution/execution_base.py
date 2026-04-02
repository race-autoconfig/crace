"""execution_base module

This module contains base classes for the different execution masters and workers.

This module contains the following classes:
    * ExecutionMasterBase - a base class to define execution master classes for crace
    * ExecutionWorkerBase - a base class to define execution worker classes for crace
"""

import asyncio
import logging
from abc import ABC, abstractmethod

execution_logger = logging.getLogger("execution")


class ExecutionMasterBase(ABC):
    """
    An abstract base class to define execution masters.

    An execution master is used to control the consumption of experiments from the ExperimentPool.
    It also needs to keep track which worker is currently executing which experiment.

    Subclass this class to implement how the ExecutionMaster and the ExecutionWorkers communicate.
    """

    def __init__(self):
        """
        Initialize the ExecutionMaster.
        """
        
        self.experiment_pool = None
        self.debug_level = 0
        self.is_running = True

    async def stop(self):
        self.is_running = False

    async def run(self):
        """
        This method runs the master, trying to distribute experiments to the workers, until the master is stopped.
        Do not override this method when subclassing.
        :return: None
        """
        while self.experiment_pool.is_open and self.is_running:
            # distribute experiments to workers
            try:
                await self._distribute_job()
            except Exception as e:
                print(f'There was an error while executing jobs:')
                print(e)
                try:
                    loop = asyncio.get_running_loop()
                except AttributeError:
                    loop = asyncio.get_event_loop()
                loop.call_soon(loop.stop)
                raise e
            await asyncio.sleep(0)

    async def _distribute_job(self):
        """
        Distributes one experiment from the ExperimentPool to a worker.

        This method will asynchronously wait until an experiment is available or the pool is closed.

        :return: None
        """

        while self.experiment_pool.is_open:
            if not self._has_free_worker():
                execution_logger.debug("Waiting for available worker on master.")
                await asyncio.sleep(1)  # wait a second and try again
                continue

            if not self.experiment_pool.has_next_experiment():
                execution_logger.debug("Waiting for available experiment on master.")
                await asyncio.sleep(1)  # wait a second and try again
                continue

            worker = self._get_free_worker()
            exp = self.experiment_pool.get_next_experiment(worker) if isinstance(worker, int) else \
                self.experiment_pool.get_next_experiment()
            self._send_job_to_worker(exp, worker)
            break

        else:
            execution_logger.debug("Master is no longer trying to distribute an experiment, "
                                  "because the experiment pool is closed.")
        await asyncio.sleep(0)

    @abstractmethod
    def _has_free_worker(self):
        """
        Checks if there is currently a worker available to take the next experiment.

        If this method returns True, then the next call to _get_free_worker will return a reference and not None.
        If this method returns False, then the next call to _get_free_worker will return None.

        :return: True, if there is a worker available, False otherwise.
        """

    @abstractmethod
    def _get_free_worker(self):
        """Returns a reference to a worker, that is currently available to take the next experiment.

        This may not directly return the ExperimentWorker object but an ID that represents how to reach the worker.
        Returns None if there is no free worker currently.

        :return: A worker that is currently available (either as object reference or by an identifier).
                 None, if no worker is available.
        """

    @abstractmethod
    def _add_worker(self, worker):
        """
        Adds a worker to master
        """

    @abstractmethod
    def _send_job_to_worker(self, experiment, worker):
        """
        This method passes an experiment to a worker for execution.

        It should not return the score or any information about the execution status.
        Use the experiment_finished function of the ExperimentPool instead.

        :param experiment: The experiment to be executed.
        It should be passed as the namedtuple Experiment.
        :param worker: The worker that the experiment is executed on.
                       Can be either an object reference or an identifier of the worker.
        :return: None
        """

    @abstractmethod
    def job_finished(self, worker, experiment, job_report):
        """
        Callback when a job is finished from a worker.

        :param worker: The worker that executed the job (either as object reference or an identifier)
        :param experiment: The experiment that was executed by the worker
        :param job_report: The additional information that the worker provided about the execution of the job.
                           The details of this parameter will depend on the implementation of the ExecutionMaster
                           and ExecutionWorker.
        :return: None
        """

    @abstractmethod
    def cancel_job(self, experiment):
        """
        Cancel the execution of this experiment.

        :param experiment: The experiment to be cancelled.
        :return: None
        """

    @abstractmethod
    def shutdown_workers(self):
        """
        Gracefully shuts down all workers.

        :return: None
        """

    @abstractmethod
    def get_running_experiment_ids(self):
        """
        Gets experiments ids of the running experiments
        """

    @abstractmethod
    def get_running_config_ids(self):
        """
        Gets config ids of the running experiments
        """


class ExecutionWorkerBase(ABC):
    """
    An abstract base class to define execution workers.

    An execution worker is used to execute experiments and collect the data about them.

    Subclass this class to implement how the ExecutionMaster and the ExecutionWorkers communicate.
    """

    def __init__(self, target_runner, max_retries):
        """
        Initializes the execution worker.

        :param target_runner: The target runner used for this worker.
        :param max_retries: The maximum number of retries allowed, should the target runner fail.
        """
        self.target = target_runner
        self.max_retries = max_retries

    @abstractmethod
    async def run_job(self, experiment):
        """
        Asynchronously executes an experiment.

        :param experiment: The experiment to be executed
        :return: A tuple including all collected information about the execution of the experiment
        """

    @abstractmethod
    def cancel_job(self):
        """
        Cancel the currently running experiment.

        If no experiment is currently running, this method has no effect.

        :return: None
        """
