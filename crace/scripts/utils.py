import os
import sys
import socket
import crace.errors as CE

def _check_scenario_parameters(scenario):
    """
    check if there is a conflict bebetween the values provided by options
    and other scenario parameters.
    """
    parameters_to_check = {
        "CraceOptions": ('scenario.options', scenario.options),
        "Parameters": ('scenario.parameters', scenario.parameters),
        "Instances": ('scenario.instances', scenario.instances),
        "Configurations": ('scenario.initial_configurations',scenario.initial_configurations)
    }

    print(f"# Calling crace using Scenario object..")
    print(f"# Checking the consistency of scenario options provided by Scenario object..")

    # check execDir
    exec_dirs = {}
    for obj, param in parameters_to_check.items():
        if obj == 'Configurations' and not param[1]:
            continue
        if obj == 'CraceOptions':
            e = param[1].execDir.value
        else:
            e = param[1].exec_dir
        if e not in exec_dirs.keys(): exec_dirs[e] = []
        exec_dirs[e].append(obj)
    if len(exec_dirs.keys()) > 1:
        print("\nERROR: There was an error while loading scenario:")
        formatted_info = ',\n '.join(f"{k}: {v}" for k, v in exec_dirs.items())
        print(f"Discordant value of option execDir: \n {formatted_info}")
        sys.tracebacklimit = 0
        raise CE.OptionError(f"Discordant value of option execDir: {exec_dirs.items()}")

    # check options
    debug_level = scenario.options.debugLevel.value

    # check parameterFile
    if not scenario.parameters.parameters_file and scenario.options.parameterFile.value:
        if debug_level > 0: print(f"# Discordant value of provided parameters: set as parameters_text")
        scenario.options.parameterFile.set_value(None)

    elif (scenario.parameters.parameters_file and
         scenario.parameters.parameters_file != scenario.options.parameterFile.value):
        if debug_level > 0: print(f"Discordant value of provided parameters: set as {scenario.parameters.parameters_file}")
        scenario.options.parameterFile.set_value(scenario.parameters.parameters_file)

    # check training instances
    if (not scenario.instances.instances_dir and not scenario.instances.instances_file and
        (scenario.options.trainInstancesDir.value or scenario.options.trainInstancesFile.value)):
        if debug_level > 0: print(f"# Discordant value of provided training instances: set as instances_list")
        scenario.options.trainInstancesDir.set_value(None)
        scenario.options.trainInstancesFile.set_value(None)

    else:
        if scenario.instances.instances_dir != scenario.options.trainInstancesDir.value:
            scenario.options.trainInstancesDir.set_value(scenario.instances.instances_dir)
            if debug_level > 0: print(f"Discordant value of provided training instances dir: set as {scenario.instances.instances_dir}")
        if scenario.instances.instances_file != scenario.options.trainInstancesFile.value:
            if debug_level > 0: print(f"Discordant value of provided training instances file: set as {scenario.instances.instances_file}")
            scenario.options.trainInstancesFile.set_value(scenario.instances.instances_file)

    # check test instances
    if (not scenario.instances.tinstances_dir and not scenario.instances.tinstances_file and
        (scenario.options.testInstancesDir.value or scenario.options.testInstancesFile.value)):
        if debug_level > 0: print(f"# Discordant value of provided training instances: set as instances_list")
        scenario.options.testInstancesDir.set_value(None)
        scenario.options.testInstancesFile.set_value(None)

    else:
        if scenario.instances.tinstances_dir != scenario.options.testInstancesDir.value:
            scenario.options.testInstancesDir.set_value(scenario.instances.tinstances_dir)
            if debug_level > 0: print(f"Discordant value of provided test instances dir: set as {scenario.instances.tinstances_dir}")
        if scenario.instances.tinstances_file != scenario.options.testInstancesFile.value:
            if debug_level > 0: print(f"Discordant value of provided test instances file: set as {scenario.instances.tinstances_file}")
            scenario.options.testInstancesFile.set_value(scenario.instances.tinstances_file)

    # check configurationsFile
    if not scenario.initial_configurations and scenario.options.configurationsFile.value:
        if debug_level > 0: print(f"# Discordant value of provided configurations: set as None")
        scenario.options.configurationsFile.set_value(None)

    elif (scenario.initial_configurations and
        scenario.initial_configurations.configurations_file != scenario.options.configurationsFile.value):
            if debug_level > 0: print(f"Discordant value of provided configurations: set as {scenario.initial_configurations.configurations_file}")
            scenario.options.configurationsFile.set_value(scenario.initial_configurations.configurations_file)

def _check_mpi_env(check_flag=1):
    """
    check mpi environment
    1. if mpi4py is installed
    2. if mpi implementation installed
    3. if the script launched by mpi launcher
    """
    error_msg, MPI, comm, rank = None, None, None, None

    if check_flag == 1:
        try:
            import mpi4py
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            mpi4py_version = mpi4py.__version__
            check_flag += 1
        except Exception as e:
            error_msg = f"\nERROR: There was an error while importing mpi4py: {repr(e)}"

    if check_flag == 2:
        try:
            mpilib_version = MPI.Get_library_version()
            check_flag += 1
        except Exception as e:
            error_msg = f"\nERROR: There was an error while checking MPI runtime: {repr(e)}"

    if check_flag == 3:
        mpirun_indicators = ["OMPI_COMM_WORLD_SIZE",
                             "OMPI_COMM_WORLD_RANK",
                             "PMI_RANK",
                             "PMI_SIZE",
                             "SLURM_PROCID",
                             "SLURM_NTASKS"]
        if not any(k in os.environ for k in mpirun_indicators):
            error_msg = f"\nERROR: The script is not launched by an MPI launcher. Please use 'mpirun' or 'srun' to launch the script."

    if error_msg:
        print(error_msg)
        if MPI: MPI.COMM_WORLD.Abort(1)
        sys.exit(1)

    if rank == 0:
        print('#------------------------------------------------------------------------------', flush=True)
        print(f"# Python version: {sys.version.split()[0]}", flush=True)
        print(f"# Installed mpi4py version: {mpi4py_version}", flush=True)
        print(f"# Detected MPI library version: {mpilib_version}", flush=True)

    return MPI, comm, rank

def _enforce_single_thread_binding(MPI, comm, rank):
    """
    enforce current process bind to the first cpu in the avalible cpu list
    especially to disable SMT when using mpirun --bind-to none
    """
    # skip on non-linux
    if not sys.platform.startswith("linux"):
        if rank == 0: print(f"# [Binding] skip: unsupported on platform {sys.platform}")
        return

    try:
        # 1. obtain local rank id from MPI implementation
        local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
        local_rank = local_comm.Get_rank()

        # 2. obtain all alowed cpu for current job
        #    if SMT is enabled, it includes physical and logical cpus
        affinity_mask = os.sched_getaffinity(0)

        # 3. sort all allowed cpus
        #    if SMT is enabled, physical cpus would be sorted at the top
        #    e.g.: [0, 1, 2, 3, ..., 32, 33, 34, 35]
        available_cpus = sorted(list(affinity_mask))

        # 4. find out the specific cpu based on the rank id
        if local_rank < len(available_cpus):
            target_cpu = available_cpus[local_rank]

            # 5. assign rank to the selected cpu
            os.sched_setaffinity(0, {target_cpu})

            # print(f"# [Python-Bind] Rank {local_rank} pinned to CPU {target_cpu} (dropped others)", flush=True)
        else:
            raise RuntimeError(f"# [Python-Bind] Warning: Not enough CPUs for Rank {local_rank}")

    except Exception as e:
        if not rank: print(f"# [Python-Bind] Failed to enforce binding: {e}", file=sys.stderr)
        raise e

def _get_binding_flag(arguments):
    """
    load option bind-cores from arguments
    """
    label = None
    idx = None
    if "--bind-cores" in arguments:
        idx = arguments.index("--bind-cores")
    elif "-b" in arguments:
        idx = arguments.index("-b")
    if idx is not None:
        if idx + 1 < len(arguments):
            val = arguments[idx + 1].lower()
            if val in ("true", "1", "yes", "on"):
                label = True
            elif val in ("false", "0", "no", "off"):
                label = False
            else:
                raise ValueError(f"Invalid value for --bind-cores: {val}")
            del arguments[idx+1]
        else:
            raise ValueError("--bind-cores provided without value")
        del arguments[idx]
    return label

def _get_affinity():
    """
    get affinity information
    """
    cpu_list = "N/A"
    cpu_mask = "N/A"
    platform = "unknown"

    # 1) psutil cross-platform
    try:
        import psutil
        proc = psutil.Process()
        if hasattr(proc, "cpu_affinity"):
            l = proc.cpu_affinity()
            cpu_list = l
            cpu_mask = hex(sum(1 << x for x in l))
            platform = "psutil"
            return cpu_list, cpu_mask, platform
    except Exception:
        pass

    # 2) Linux /proc/self/status (best binding info)
    if sys.platform.startswith("linux") and os.path.exists("/proc/self/status"):
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("Cpus_allowed_list"):
                        cpu_list = line.strip().split(":", 1)[1].strip()
                    elif line.startswith("Cpus_allowed:"):
                        cpu_mask = line.strip().split(":", 1)[1].strip()
            platform = "proc"
            return cpu_list, cpu_mask, platform
        except Exception:
            pass

    # 3) Linux sched_getaffinity
    if hasattr(os, "sched_getaffinity"):
        try:
            l = sorted(list(os.sched_getaffinity(0)))
            cpu_list = l
            cpu_mask = hex(sum(1 << x for x in l))
            platform = "sched"
            return cpu_list, cpu_mask, platform
        except Exception:
            pass

    # 4) macOS fallback: use logical CPU count
    if sys.platform == "darwin":
        try:
            import psutil
            n = psutil.cpu_count()
            cpu_list = list(range(n))
            cpu_mask = hex(sum(1 << x for x in cpu_list))
            platform = "darwin"
            return cpu_list, cpu_mask, platform
        except Exception:
            pass

    # 5) Windows fallback: logical CPU count
    if sys.platform.startswith("win"):
        try:
            import psutil
            n = psutil.cpu_count()
            cpu_list = list(range(n))
            cpu_mask = hex(sum(1 << x for x in cpu_list))
            platform = "windows"
            return cpu_list, cpu_mask, platform
        except Exception:
            pass

    return cpu_list, cpu_mask, platform

def _print_affinity_info(comm, rank):
    """
    print the affinity information
    """
    size = comm.Get_size()

    # ---------- basic info ----------
    pid = os.getpid()
    hostname = socket.gethostname()
    cluster = os.environ.get("SLURM_CLUSTER_NAME", "N/A")
    jobid = os.environ.get("SLURM_JOB_ID", "N/A")
    partition = os.environ.get("SLURM_JOB_PARTITION", "N/A")

    # ---------- /proc info ----------
    cpus_allowed_list, cpus_allowed_addr, platform= _get_affinity()

    msg_dict = {
        "Rank": rank,
        "Platform": platform,
        "Cluster": cluster,
        "Partition": partition,
        "Node": hostname,
        "JobID": jobid,
        "PID": pid,
        "Cpus_allowed_list": cpus_allowed_list,
        "Cpus_allowed_addr": cpus_allowed_addr,
    }

    # ---------- ordered output ----------
    all_msg = comm.gather(msg_dict, root=0)
    if rank == 0:
        print('#------------------------------------------------------------------------------', flush=True)
        print(f"# MPI Affinity Information:", flush=True)
        for msg in sorted(all_msg, key=lambda x: x["Rank"]):
            for k in msg:
                if k == "Rank":
                    print(f"#\n#  {k}: {msg.get(k)}", flush=True)
                elif msg_dict.get(k) not in ("NA", "N/A", "UNKNOWN", None):
                    print(f"#   {k:<18}: {msg.get(k)}", flush=True)
