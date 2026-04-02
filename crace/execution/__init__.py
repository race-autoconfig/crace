"""
This module provides methods and classes to execute experiments.
"""
import asyncio
import collections

import crace.execution.mpi
from crace.execution.execution_pool import ExecutionPool
from crace.execution.local_concurrent import ConcurrentExecutionMaster


def start_execution(options, experiment_finished_callback, experiments, test: bool=False):
    # create master execution handler
    master = start_master(options)
    worker_num = options.parallel.value if options.parallel.value > 1 else 1
    adaptive = options.domType.value in ("adaptive", "adaptive-dom", "adaptive-dominance", "adaptive-Dominance")
    strict_pool = True if (adaptive or options.testExperimentSize.value > 1) and not test else False
    bound_out = True if options.capping.value and options.boundAsTimeout.value else False

    # create and set pool
    # if options.targetEvaluator.value is None:
    if options.debugLevel.value >= 3:
        print("# Initializing execution pool")

    pool = ExecutionPool(chainMap=collections.ChainMap(),
                         executioner=master,
                         callback=experiment_finished_callback,
                         experiments=experiments,
                         worker_num=worker_num,
                         strict_pool=strict_pool,
                         bound_out=bound_out)

    master.experiment_pool = pool
    return master, pool


def stop_execution_sync(master, pool):
    print("Before closing")
    pool.close()
    print("Pool closed")
    master.stop()
    print("Master stopped")
    print("Shutdown complete")


async def stop_execution_async(master, pool):
    await asyncio.sleep(0)
    # check if the closing is the same if we close self._pool
    print("Before closing")
    pool.close()
    print("Pool closed")
    await master.stop()
    print("Master stopped")
    await asyncio.sleep(0)  # TODO: Check that the executioner did shut down
    print("Shutdown complete")

async def stop_execution_checking(master, pool):
    await asyncio.sleep(0)
    # check if the closing is the same if we close self._pool
    pool.close()
    await master.stop()
    await asyncio.sleep(0)  # TODO: Check that the executioner did shut down
    print("#\n# Checking complete")

def start_master(options):
    """
    Call this method from the MPI process that is supposed to take on the role of the master.
    :return: The MPIMaster object, created by this method
    """
    if options.parallel.value == 0:
        n_parallel = 1
    else:
        n_parallel = options.parallel.value

    target_runner_file = options.targetRunner.value
    max_retries = options.targetRunnerRetries.value

    if options.mpi.value:
        if options.debugLevel.value >= 5:
            print("# Starting MPI execution master")
        master = crace.execution.mpi.start_master(number_of_workers=n_parallel,
                                                  exec_cmd=options.targetRunnerLauncher.value,
                                                  target_runner=target_runner_file,
                                                  max_retries=max_retries,
                                                  debug_level=options.debugLevel.value,
                                                  log_level=options.logLevel.value)

    else:
        if options.debugLevel.value >= 5:
            print("# Starting execution master")
        master = ConcurrentExecutionMaster(number_workers=n_parallel,
                                           exec_cmd=options.targetRunnerLauncher.value,
                                           target_runner_file=target_runner_file,
                                           max_retries=max_retries,
                                           debug_level=options.debugLevel.value)

    if options.debugLevel.value >= 5:
        print("# Running master")
    asyncio.ensure_future(master.run())
    return master

