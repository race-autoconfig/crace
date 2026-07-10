import os
import sys
import socket

def check_mpi_env(check_flag=1):
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

def enforce_single_thread_binding(MPI, comm, rank):
    """
    enforce current process bind to the first cpu in the avalible cpu list
    especially to disable SMT when using mpirun --bind-to none
    """
    error_msg = None

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
        # 5. assign rank to the selected cpu
        if local_rank < len(available_cpus):
            target_cpu = available_cpus[local_rank]
            os.sched_setaffinity(0, {target_cpu})

        # # 4. only one cpu is available, no binding
        elif len(available_cpus) == 1:
            pass

        else:
            error_msg = f"# [Python-Bind] Warning: Not enough CPUs for Rank {local_rank}"

    except Exception as e:
        error_msg = f"# [Python-Bind] Failed to enforce binding: {e}"
    
    err_dict = {"Rank": rank, "Error": error_msg}    
    all_err = comm.gather(err_dict, root=0)

    if rank == 0:
        for err in sorted(all_err, key=lambda x: x["Rank"]):
            if err["Error"]: raise RuntimeError(err["Error"])

def get_binding_flag(arguments):
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

def print_affinity_info(mpi: bool, MPI=None, test=False, sel_rank=None,):
    """
    print the affinity information

    mpi: whether enable mpi 
    comm: MPI communicator (optional, default is None: no MPI used)
    rank: MPI rank (optional, default is None: no MPI used)
    test: whether in test phase (optional, default is False)
    sel_rank: selected ranks to print when testParallel is enabled
    """
    msg_dict = {}

    if not mpi:
        # no mpi: before training or testing
        #   from scripts.main: print_affinity_info(mpi=False)
        #   from tester:       print_affinity_info(mpi=False, test=True, sel_rank=sel_rank)
        msg_dict = _get_affinity_info()
        _print_affinity_msg([msg_dict], msg_dict)

    elif not test:
        # mpi and called before training
        #   from scripts.mpi: print_affinity_info(mpi=True, MPI=MPI)
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        tmp = _get_affinity_info(rank)
        msg_dict.update(tmp)

        all_msg = comm.gather(msg_dict, root=0)
        if rank == 0:
            _print_affinity_msg(all_msg, msg_dict)

    else:
        # mpi and called before testing
        #   from tester: print_affinity_info(mpi=True, test=True, sel_rank=sel_rank)
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        print(f"# CPU Affinity Information - selected ranks: {', '.join([str(x) for x in sel_rank])}", flush=True)

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


def _get_affinity_info(rank=None):
    """
    get the affinity information of the current process
    """
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

    return msg_dict

def _print_affinity_msg(all_msg, msg_dict):
    """
    print the affinity information in a formatted way
    """
    print('#------------------------------------------------------------------------------', flush=True)
    print(f"# CPU Affinity Information:", flush=True)
    for msg in sorted(all_msg, key=lambda x: x["Rank"]):
        for k in msg:
            if k in ["Rank", "Size"]:
                print(f"#\n#  {k}: {msg.get(k)}", flush=True)
            elif msg_dict.get(k) not in ("NA", "N/A", "UNKNOWN", None):
                print(f"#   {k:<18}: {msg.get(k)}", flush=True)
