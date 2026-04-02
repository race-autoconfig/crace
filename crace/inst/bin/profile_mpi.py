"""
This file is used as a wrapper when running crace with MPI on a cluster.
On the IRIDIA cluster, the spawning of MPI processes is not working correctly.
When calling all crace processes at the same time, they are trying to load the scenario to understand the configuration,
resulting in printing the scenario many times.

This script is supposed to be a workaround.
It can be called many times in parallel and as it assumes that the user runs MPI, it can directly suppress the scenario
output on the workers.

This is a version of crace/scripts/mpi.py for executing crace with cProfile.Profile().
"""

import os
import sys
import socket

CRACE_HOME = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if CRACE_HOME not in map(os.path.abspath, sys.path):
    sys.path.insert(0, os.path.abspath(CRACE_HOME))

import profile_main
import crace.execution.mpi as cmpi
from crace.containers.scenario import Scenario
from crace.scripts.utils import _get_binding_flag, _enforce_single_thread_binding, _get_affinity


if __name__ == "__main__":
    
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    except Exception as e:
        print(f"Error initializing MPI: {e}")
        MPI.COMM_WORLD.Abort(1)
        raise e

    # print(rank)
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
        profile_main.start_profile(arguments)
        cmpi.broadcast_termination_signal()
    elif rank > 0:
        # print("Worker")
        # TODO: Suppress output here
        # TODO: Suppress creating of log files here
        scenario = Scenario.from_input(arguments, silent=True)
        cmpi.start_worker(exec_dir=scenario.options.execDir.value,
                                         target_runner_file=scenario.options.targetRunner.value,
                                         max_retries=scenario.options.targetRunnerRetries.value,
                                         log_folder=scenario.options.logDir.value,
                                         debug_level=scenario.options.debugLevel.value,
                                         log_level=scenario.options.logLevel.value)
