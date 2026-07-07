"""
This file is used as a wrapper when running crace with MPI on a cluster.
For example, on the IRIDIA cluster, the openmpi4-gnu9 and slurm are available.

When all crace processes are launched simultaneously, it's normal for each process
attempting to load the scenario to understand the options, resulting in printing
the scenario information many times.

This script is supposed to be a workaround.
Via package mpi4py, only the node with rank value 0, which is set as the master,
will print the scenario information.
Besides, using option '--bind-cores', it's feasible to bind each worker to a specific
physical node on the cluster.
Note: this option is unavailable on macOS system.
"""

import os
import sys
import inspect

current_file = os.path.normpath(__file__)
CRACE_HOME = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

if CRACE_HOME not in map(os.path.abspath, sys.path):
    sys.path.insert(0, os.path.abspath(CRACE_HOME))

import crace.errors as CE
import crace.execution.mpi
from crace.scripts.main import crace_cmdline
from crace.containers.scenario import Scenario
from crace.scripts.utils import _check_mpi_env, _get_binding_flag, _enforce_single_thread_binding, _print_affinity_info

def start_mpi(args=None, cli: bool=False):
    """
    start crace with mpi4py
    """
    MPI, comm, rank = _check_mpi_env()

    if cli:
        arguments = args
    else:
        arguments = sys.argv[1:]  # command line arguments as a list

    try:

        if _get_binding_flag(arguments):
            _enforce_single_thread_binding(MPI, comm, rank)    # load option bind-cores from arguments
        _print_affinity_info(comm, rank)        # print affinity information in order

    except Exception as e:
        if rank == 0:
            print(f"\nERROR: There was an error while pining processors: {repr(e)}")
        MPI.COMM_WORLD.Abort(1)

    try:
        if rank == 0:
            # print("Master")
            crace_cmdline(arguments=arguments, console=False, cli=cli)
            crace.execution.mpi.broadcast_termination_signal()
        elif rank > 0:
            # print("Worker")
            # TODO: Suppress output here
            # TODO: Suppress creating of log files here
            scenario = Scenario.from_input(arguments, silent=True)
            crace.execution.mpi.start_worker(exec_dir=scenario.options.execDir.value,
                                            exec_cmd=scenario.options.targetRunnerLauncher.value,
                                            target_runner_file=scenario.options.targetRunner.value,
                                            max_retries=scenario.options.targetRunnerRetries.value,
                                            log_folder=scenario.options.logDir.value,
                                            debug_level=scenario.options.debugLevel.value,
                                            log_level=scenario.options.logLevel.value)
    except (SystemExit, KeyboardInterrupt, Exception) as e:
        if any(isinstance(e, cls) for cls in [x[1] for x in inspect.getmembers(CE, inspect.isclass)]):
            pass
        else:
            print(f"\nERROR: There was an error while executing crace(mpi) on rank {rank}: {repr(e)}")
            MPI.COMM_WORLD.Abort(1)
            sys.exit(1)

if __name__ == "__main__":
    """
    Allowed to be called only by this file
    """
    start_mpi()
