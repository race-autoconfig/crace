"""mpi module

This module contains classes for the different execution masters and workers using MPI.

This module contains the following classes:
    * MPIExecutionMaster - an execution master class for crace using MPI
    * MPIExecutionWorker - an execution worker class for crace using MPI
"""

import os
import sys
import time
import socket
import asyncio
import logging

from abc import ABC

from crace.utils import get_loop
from crace.execution.target_runner import TargetRunner
from crace.execution.execution_base import ExecutionMasterBase, ExecutionWorkerBase

MPI = None

mpi_logger = logging.getLogger("mpi")
execution_logger = logging.getLogger("execution")


def _import_mpi(quiet=False):
    """
    Imports the mpi4py package if this module is used

    :param quiet: If True raise an error when the package is not installed
    :return: None
    """
    global MPI
    try:
        import mpi4py.MPI
        MPI = mpi4py.MPI
    except ImportError as import_error:
        if not quiet:
            raise ImportError("To run this program with MPI you need the library mpi4py installed.\n"
                              "Please make sure that mpi4py is properly installed on your system.") from import_error


def start_worker(exec_dir, exec_cmd, target_runner_file, max_retries, log_folder, debug_level=0, log_level=0):
    """
    Creates an MPIExecutionWorker object and runs its main loop.

    Call this method from the MPI process that is supposed to take on the role of a worker.
    NOTE: This method blocks until the worker is terminated (as there is no need for the worker to do anything else).

    :param target_runner_file: The target runner that is used to evaluate the experiments
    :param max_retries: If the execution of the target runner fails, max_retries indicates the number of times that
        the experiment is retried, before the failure is reported to the master. This leads to a maximum of
        max_retries + 1 executions of the experiment (one time for the original failure and up to max_retries retries).
    :return: None
    """
    _import_mpi()
    os.chdir(exec_dir)
    target_runner = TargetRunner(exec_cmd, target_runner_file, debug_level)

    # TODO: Massively improve this
    from crace.utils import setup_logger
    if log_level >= 4: setup_logger('mpi', f'{log_folder}/mpi_worker_{MPI.COMM_WORLD.Get_rank()}.log', level=0)
    worker = MPIExecutionWorker(target_runner, max_retries, debug_level=debug_level)
    worker.run()


def start_master(number_of_workers, exec_cmd, target_runner, max_retries, debug_level, log_level):
    """
    Creates an MPIExecutionMaster object, schedules its execution, and returns it.

    Call this method from the MPI process that is supposed to take on the role of the master.

    :return: The MPIExecutionMaster object, created by this method
    """

    _import_mpi()
    master = MPIExecutionMaster(number_of_workers, exec_cmd, target_runner, max_retries, debug_level)
    event_loop = asyncio.get_event_loop()
    event_loop.create_task(master.listen())  # make the master listen to incoming messages
    event_loop.create_task(master.run())  # make the master distribute jobs to the workers

    return master


class MPIProcess(ABC):
    """
    An abstract base class that provides access to the MPI communicator.
    """

    TERMINATE_JOB = "TERMINATE_JOB"
    SHUTDOWN_WORKER = "SHUTDOWN_WORKER"
    SEND_JOB = "SEND_JOB"
    INIT = "INIT"

    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    TERMINATED = "TERMINATED"

    def __init__(self):
        self.master = 0
        self.active = True

    def _disconnect(self):
        status = MPI.Status()
        request = self.comm.irecv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
        time.sleep(1)
        completed, message = request.test(status)

        while completed:
            print(f"Ignoring queued message on rank {self.comm.Get_rank()}")
            request = self.comm.irecv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
            time.sleep(1)
            completed, message = request.test(status)

        print(f"Ignored all queued message on rank {self.comm.Get_rank()}. Now disconnecting")
        rank = self.comm.Get_rank()
        self.comm.Disconnect()
        print(f"Rank {rank} disconnected")


class MPIExecutionMaster(MPIProcess, ExecutionMasterBase):
    """
    The MPIExecutionMaster implements the behavior of the MPI master processes for crace.

    That is, the process waits for an experiment and sends it to an available worker.
    Once the worker finishes execution of the job, it will send the results back to the master.
    All communication is handled over MPI.
    """

    mpi_master_id = 0

    def __init__(self, number_of_workers, exec_cmd, target_runner, max_retries, debug_level):
        # TODO: Understand multiple inheritance and improve this
        super().__init__()
        ExecutionMasterBase.__init__(self)  # TODO: Understand why this is not part of the super() call

        self.debug_level = debug_level
        self.comm = MPI.COMM_WORLD

        # Define a set of workers and discard the main process
        mpi_logger.debug("Master sees comm with size %s", str(self.comm.Get_size()))

        self.workers = set(range(self.comm.Get_size()))
        # self.workers = set(range(number_of_workers))
        self.workers.discard(self.comm.Get_rank())
        self.size = self.comm.Get_size() - 1
        if self.size == 1:
            raise ValueError

        self.id = MPIExecutionMaster.mpi_master_id
        MPIExecutionMaster.mpi_master_id += 1

        for worker in self.workers:
            self.comm.send([MPIProcess.INIT], dest=worker)

        self.running_jobs = {}  # this is a dictionary that keeps
        #                         the experiment_id as key and
        #                         a tuple (worker_id, experiment) as key
        self.cancelled_jobs = []  # this is a list that keeps the currently
        #                           cancelled jobs for cross-referencing with
        #                           jobs that might finish before the signal
        #                           is received

    async def stop(self):
        mpi_logger.debug(f"Stopping master {self.id}")

        # Drain all of the message queue before returning. 
        # Otherwise the next master will receive messages for this one.
        while self.running_jobs:
            mpi_logger.debug(f"Master {self.id} is waiting for {len(self.running_jobs.keys())} jobs to finish before shutting down.")
            await asyncio.sleep(0)
        await super().stop()
        await asyncio.sleep(0)  # Give all other asyncio tasks the chance to finish

        mpi_logger.debug(f"Master {self.id} is running: {self.is_running}")
        mpi_logger.debug(f"Master {self.id} stopped.")


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
        Returns a reference to a worker, that is currently available to take the next experiment.

        Returns None if there is no free worker currently.

        :return: A worker that is currently available (represented by its MPI rank).
                 None, if no worker is available.
        """
        if self.workers:
            return self.workers.pop()
        return None

    def _send_job_to_worker(self, experiment, worker):
        """
        This method passes an experiment to a worker for execution.

        It does not return the score or any information about the execution status.

        :param experiment: The experiment to be executed. It should be passed as the namedtuple Experiment.
        :param worker: The worker that the experiment is executed on, identified by its MPI rank.
        :return: None
        """
        experiment_id = experiment.experiment_id
        self.running_jobs[experiment_id] = (worker, experiment)
        mpi_logger.debug("Sending experiment %s from master to worker %s", str(experiment_id), worker)
        self.comm.send([MPIProcess.SEND_JOB, experiment], dest=worker, tag=experiment.experiment_id)

    def _send_termination(self, worker, exp_id):
        """
        Sends a request to terminate the currently running job on the worker

        :param worker: The MPI rank of the worker which should terminate its running job
        :return: None
        """
        self.comm.send([MPIProcess.TERMINATE_JOB, exp_id], dest=worker)

    def cancel_job(self, experiment):
        """
        Cancel the execution of this experiment.

        :param experiment: The experiment to be cancelled.
        :return: None
        """
        if experiment.experiment_id not in self.running_jobs:
            mpi_logger.debug("Tried to cancel experiment %s, but it was not in the list of running experiments.",
                             str(experiment.experiment_id))
            return

        worker, experiment = self.running_jobs[experiment.experiment_id]
        mpi_logger.debug("Cancelling experiment %s on worker %s", str(experiment.experiment_id), worker)
        self._send_termination(worker, experiment.experiment_id)
        self.cancelled_jobs.append(experiment.experiment_id)

    def _send_shutdown(self, worker):
        """
        Sends a request to shutdown the worker

        :param worker: The MPI rank of the worker which should shut down
        :return: None
        """
        self.comm.isend([MPIProcess.SHUTDOWN_WORKER], dest=worker)

    def shutdown_worker(self, worker):
        """
        Gracefully shut down a single worker.

        :param worker: The MPI rank of the worker to be shut down.
        :return: None
        """
        mpi_logger.debug("Shutting down worker %s", worker)
        self._send_shutdown(worker)

    def shutdown_workers(self):
        """
        Gracefully shuts down all workers.

        :return: None
        """
        print("# Shutting down workers")
        for worker in self.workers:
            self.shutdown_worker(worker)
        for worker, _ in self.running_jobs.values():
            self.shutdown_worker(worker)
        print("# Signals send")


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
                           * (time) the system time duration of the last execution of the target runner
                           * (int) the number of times the target runner was retried
        :return: None
        """
        # return the worker to the pool
        self.workers.add(worker)

        mpi_logger.debug("Job report: %s", job_report)

        # process the job_report
        full_line = job_report[1]
        cancelled = job_report[0]
        exp_id = job_report[2]
        return_code = job_report[3]
        output_stdout = job_report[4]
        output_stderr = job_report[5]
        target_runner_time = job_report[6]
        start_time = job_report[7]
        end_time = job_report[8]
        num_retries = job_report[9]

        mpi_logger.debug("Master received finished experiment %s from worker %s", str(experiment.experiment_id), worker)
        mpi_logger.debug("The experiment was%s cancelled. "
                         "The target runner had to retry %s times. "
                         "The last return code was %s", "" if cancelled else " not", str(num_retries), str(return_code))

        try:
            self.experiment_pool.experiment_finished(experiment, full_line, cancelled, return_code, output_stdout,
                                                 output_stderr, target_runner_time, start_time, end_time)

        except Exception as e:
            print("\nERROR: There was an error while executing job: ")
            print(e)
            sys.exit(1)

    async def _async_wait_for_message(self):
        """
        This method waits asynchronously for a new message over MPI.

        :return: A tuple containing the keyword and the payload of the message.
        """
        status = MPI.Status()
        request = self.comm.irecv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)

        completed, message = request.test(status)

        while self.is_running and not completed:
            mpi_logger.debug("Master %s is still waiting to receive a message", str(self.id))
            await asyncio.sleep(1)

            if self.is_running:
                completed, message = request.test(status)
        keyword, payload = message[0], message[1]
        return keyword, payload

    def _check_for_message(self):
        """
        This method performs a non-blocking check for a message over MPI.

        :return: A tuple containing if a boolean indicating if a message was received,
        the keyword and the payload of the message.
        """
        completed = self.comm.iprobe()
        if completed:
            message = self.comm.recv()
            keyword, payload = message[0], message[1]
        else:
            keyword = None
            payload = None
        return completed, keyword, payload

    async def listen(self):
        """
        Listen and react to any incoming messages

        :return: None
        """
        while self.is_running:
            mpi_logger.debug("Master %s is waiting to receive a message", str(self.id))
            completed, keyword, payload = self._check_for_message()

            if completed:
                mpi_logger.debug("Received message %s with keyword %s on master", payload, keyword)
                mpi_logger.debug(f"Experiment ID received on master {self.id}: {payload[1]}")
                mpi_logger.debug(f"Running jobs on master {self.id}: {list(self.running_jobs.keys())}")

                # TODO: This is necessary because of the bug in self.stop(). See there for more info.
                if payload[1] in self.running_jobs:
                    worker, experiment = self.running_jobs.pop(payload[1])
                    cancelled = (keyword == MPIProcess.TERMINATED) or (experiment.experiment_id in self.cancelled_jobs)
                    self.job_finished(worker, experiment, [cancelled] + list(payload))
                else:
                    mpi_logger.warning(f"Experiment {payload[1]} is not in running jobs."
                                       f"It will be ignored.")

            else:
                mpi_logger.debug(f"Master {self.id} did not receive a message.")
                await asyncio.sleep(1)

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

    def _add_worker(self, worker):
        """
        Adds a worker to master
        """
        pass


class MPIExecutionWorker(ExecutionWorkerBase, MPIProcess):
    """
    The MPIExecutionWorker implements the behavior of the MPI worker processes for crace.

    The MPIExecutionWorker waits for an incoming message sending an experiment.
    Once the experiment is received the worker will execute it using the target runner.
    While the experiment is running, the worker will listen to a terminate command.
    If the experiment is terminated (either successfully or by the termination command), the worker will return the
    results back to the MPIExecutionMaster process.
    """

    def __init__(self, target_runner, max_retries, debug_level=0):
        """
        Initialize an MPIExecutionWorker

        :param target_runner: The target runner used for this worker.
        :param max_retries: The maximum number of retries allowed, should the target runner fail.
        """
        # TODO: Understand multiple inheritance and improve this
        ExecutionWorkerBase.__init__(self, target_runner, max_retries)
        MPIProcess.__init__(self)

        self.debug_level = debug_level
        self.comm = MPI.COMM_WORLD

        mpi_logger.debug("Worker %s sees comm of size %s", str(self.comm.Get_rank()), str(self.comm.Get_size()))

        self.running = False
        self.terminated_jobs = []
        self.current_experiment_id = None

    def initialize_worker(self):
        """
        Use this to initialize the worker every time a new master is starting.
        :return:
        """
        self.terminated_jobs.clear()
        self.current_experiment_id = None

    def cancel_job(self):
        """
        Cancel the currently running experiment.

        If no experiment is currently running, this method has no effect.

        :return: None
        """
        # check, if target_runner is actually running the experiment
        if self.current_experiment_id in self.terminated_jobs:
            self.target.cancel_experiment()
        # self.job_terminated = True

    def _shutdown(self):
        """
        Request to shutdown this worker.

        This will cancel any running experiment and set the running flag to be False.
        As a result, this worker will not accept anymore incoming messages.
        NOTE: This method will return quickly, but the worker might not be finished instantaneously.

        :return: None
        """
        if self.debug_level >=1:
            print(f"Worker {self.comm.Get_rank()} shutting down.")
        mpi_logger.debug("Worker %s is shutting down.", str(self.comm.Get_rank()))

        # set a flag to prevent the async_listening from scheduling itself again
        self.running = False

        # terminate job if there is one running
        self.cancel_job()

    async def run_job(self, experiment):
        """
        Asynchronously executes an experiment.

        This method will retry the execution, if it failed less than self.max_retries times.

        :param experiment: The experiment to be executed
        :return: A tuple including the following information:
                 * return_code: the return code of the last execution of the target runner
                 * stdout: the stdout of the last execution of the target runner
                 * stderr: the stderr of the last execution of the target runner
                 * start_time: start time of the experiment
                 * end_time of the experiment
                 * target_runner_time: the duration of the last execution of the target runner
                 * num_retries: the number of retries
        """
        current_retries = 0

        full_line, return_code, stdout, stderr, start_time, end_time = await self.target.run_experiment(experiment)

        while (self.current_experiment_id not in self.terminated_jobs
               and return_code != 0
               and current_retries < self.max_retries
              ):
            current_retries += 1

            if self.debug_level >=1:
                print("# Retry {} experiment {}".format(current_retries, experiment.experiment_id))

            full_line, return_code, stdout, stderr, start_time, end_time = await self.target.run_experiment(experiment)

        total_time = end_time - start_time

        return full_line, return_code, stdout, stderr, start_time, end_time, total_time, current_retries

    async def _async_execute_job(self, mpi_tag, experiment):
        """
        Asynchronously executes the job and sends the results back to the MPIExecutionMaster.

        :param mpi_tag: The MPI tag that can be used to identify this conversation.
        :param experiment: The experiment to be executed.
        :return: None
        """
        self.current_experiment_id = experiment.experiment_id

        full_line, return_code, stdout, stderr, start_time, end_time, target_runner_time, num_retries = await self.run_job(experiment)

        # Correct handling of return values here
        if experiment.experiment_id in self.terminated_jobs:
            experiment_status = MPIProcess.TERMINATED
        elif return_code == 0:
            experiment_status = MPIProcess.SUCCEEDED
        else:
            experiment_status = MPIProcess.FAILED

        self.current_experiment_id = None

        mpi_logger.debug(f"Worker {self.comm.Get_rank()} sends results of experiment {experiment.experiment_id}.")

        self.comm.ssend([experiment_status,
                        (full_line, experiment.experiment_id, return_code, stdout, stderr, target_runner_time,
                         start_time, end_time, num_retries)],
                        self.master, mpi_tag)

    async def _async_wait_for_message(self):
        """
        This method waits asynchronously for a new message over MPI.

        :return: A tuple containing the tag of this converstaion, the keyword and the payload of the message.
        """
        status = MPI.Status()
        request = self.comm.irecv(source=self.master, tag=MPI.ANY_TAG)

        completed, message = request.test(status)
        while not completed:
            completed, message = request.test(status)
            await asyncio.sleep(0)

        mpi_logger.debug("Worker %s got a message!\n %s", str(self.comm.Get_rank()), message)
        return status.tag, message[0], message[1:]

    async def _async_listen(self):
        """
        This method makes the worker listen and react asynchronously to input from the master.

        The asynchronicity is used to be able to check the status of the execution of an experiment and listen to
        incoming messages without the need for parallelism (e.g. threading, multiprocessing, distributed computation).

        The worker expects three different kind of messages
        (denoted by their keyword, the first element of the received message)
            * SEND_JOB: the rest of the message will be interpreted as an experiment that will execute on this worker
            * TERMINATE_JOB: the currently running experiment will be terminated
            * SHUTDOWN_WORKER: the worker will terminate the running experiment and shutdown.
              It will not react to any further messages.

        :return: None
        """
        event_loop = asyncio.get_event_loop()

        mpi_logger.debug("Worker %s is waiting to receive a message", str(self.comm.Get_rank()))

        mpi_tag, keyword, message = await self._async_wait_for_message()

        mpi_logger.debug("Worker %s received a message with keyword %s", str(self.comm.Get_rank()), keyword)

        if keyword == MPIProcess.INIT:
            self.initialize_worker()

        elif keyword == MPIProcess.TERMINATE_JOB:
            experiment_id = message[0]
            self.terminated_jobs.append(experiment_id)
            asyncio.get_event_loop().call_soon(self.cancel_job)

        elif keyword == MPIProcess.SHUTDOWN_WORKER:
            asyncio.get_event_loop().call_soon(self._shutdown)

        elif keyword == MPIProcess.SEND_JOB:
            experiment = message[0]  # it seems that message is wrapped once in an array. This removes the outer wrap
            mpi_logger.debug(f"Worker {self.comm.Get_rank()} has received experiment "
                             f"{experiment.experiment_id} ({experiment.configuration_id}, {experiment.instance_id}).")
            asyncio.get_event_loop().create_task(self._async_execute_job(mpi_tag, experiment))

        else:
            mpi_logger.warning("Unknown keyword %s. Message %s is ignored.", keyword, message)

        if self.running:
            # resubmit this method to the event loop
            event_loop.create_task(self._async_listen())
        else:
            await asyncio.sleep(1)
            # stop the loop
            asyncio.get_event_loop().stop()


    def run(self):
        """
        Runs this worker until an exception is caught or the worker is asked to shutdown.

        This method is blocking until the worker shuts down.

        :return: None
        """
        mpi_logger.debug("Running worker %s on %s", self.comm.Get_rank(), socket.gethostname())

        loop = get_loop()

        loop.create_task(self._async_listen())

        self.running = True

        # This will block until the loop is stopped, then finish and exit the method
        loop.run_forever()  # TODO: Catch exceptions and inform the master about failure
        loop.close()
        print(f"Loop stopped on {self.comm.Get_rank()}")
        mpi_logger.debug("Worker %s is finished.", str(self.comm.Get_rank()))

        MPI.Finalize()
        print(f"Disconnected")

def broadcast_termination_signal():
    comm = MPI.COMM_WORLD
    for worker in range(comm.Get_size()):
        mpi_logger.debug("Shutting down worker %s", worker)
        comm.isend([MPIProcess.SHUTDOWN_WORKER], dest=worker)


if __name__ == "__main__":
    mpi_logger.debug("Worker args:")
    mpi_logger.debug(sys.argv)
    target_runner_file = sys.argv[1]
    max_target_runner_retries = sys.argv[2]
    log_level = sys.argv[3]
    start_worker(target_runner_file, int(max_target_runner_retries), int(log_level))

