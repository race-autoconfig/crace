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
import socket

current_file = os.path.normpath(__file__)
CRACE_HOME = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

if CRACE_HOME not in map(os.path.abspath, sys.path):
    sys.path.insert(0, os.path.abspath(CRACE_HOME))

import crace.execution.mpi
from crace.scripts.main import crace_cmdline
from crace.containers.scenario import Scenario
from crace.scripts.utils import _get_binding_flag, _enforce_single_thread_binding, _get_affinity

def start_mpi(args=None, cli: bool=False):
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    except Exception as e:
        print(f"Error initializing MPI: {e}")
        MPI.COMM_WORLD.Abort(1)
        raise e

    # print(rank)
    if cli:
        arguments = args
    else:
        arguments = sys.argv[1:]  # command line arguments as a list

    binding_flag = _get_binding_flag(arguments)
    if binding_flag: _enforce_single_thread_binding()

    # print rank information
    # ---------- basic info ----------
    pid = os.getpid()
    hostname = socket.gethostname()
    cluster = os.environ.get("SLURM_CLUSTER_NAME", "UNKNOWN")
    jobid = os.environ.get("SLURM_JOB_ID", "N/A")
    partition = os.environ.get("SLURM_JOB_PARTITION", "N/A")

    # ---------- /proc info ----------
    cpus_allowed_list, cpus_allowed_mask, source= _get_affinity()

    # ---------- ordered output ----------
    comm.Barrier()
    print(
        f"\n===== Rank {rank} =====\n"
        # f"Cluster          : {cluster}\n"
        f"Source           : {source}\n"
        f"Node             : {hostname}\n"
        f"JobID            : {jobid}\n"
        # f"Partition        : {partition}\n"
        f"PID              : {pid}\n"
        f"Cpus_allowed_list: {cpus_allowed_list}\n"
        f"Cpus_allowed     : {cpus_allowed_mask}\n",
        flush=True,
    )
    comm.Barrier()

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

if __name__ == "__main__":
    """
    Allowed to be called only by this file
    """
    start_mpi()