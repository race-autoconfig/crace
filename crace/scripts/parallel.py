# scripts/crace-parallel.py

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from crace.utils.const import WIDTH

def error(msg):
    print(f"! ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def check_crace():
    """check crace command exists"""
    from shutil import which

    if which("crace") is None:
        print("ERROR: 'crace' command not found, is 'crace' installed in current environment?\n")
        print("Python location:")
        print("  python :", shutil.which("python"))
        print("  python3:", shutil.which("python3"))
        print("\nPATH:")
        for p in os.environ["PATH"].split(os.pathsep):
            print(" ", p)
        sys.exit(127)

    print("NOTE: calling python using the global environment")



def run_crace(execdir, seed, params):
    pid = os.getpid()

    stdout = execdir / f"crace-{pid}.stdout"
    stderr = execdir / f"crace-{pid}.stderr"

    with open(stdout, "w") as out, open(stderr, "w") as err:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            ["crace", "--exec-dir", str(execdir), "--seed", str(seed)] + params,
            stdout=out,
            stderr=err,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )

    return proc


def parse_arguments(argv=None):
    """
    Parse the command line arguments and return them.
    """
    parser = argparse.ArgumentParser(
        description="""
        Execute parallel crace runs. Arguments are separated by '--': arguments before '--' configure parallel crace runs, while arguments after '--' are forwarded to crace.
        """,
        formatter_class=lambda prog: CustomFormatter(
            prog, width=WIDTH, max_help_position=50),
    )

    parser.add_argument("-n", "--repetitions",
                        default=1,
                        dest="repetitions",
                        help="The number of parallel crace runs",)

    parser.add_argument("-p", "--exec-dir",
                        default=os.getcwd(),
                        dest="execdir",
                        help="The directory path for saving the parallel crace results",)

    parser.add_argument("-t", "--expname",
                        default="exp", 
                        dest="expname",
                        help="The prefix of the folder name to save the parallel crace results",)

    parser.add_argument("-s", "--seeds",
                        nargs='+',
                        default=1234567,
                        dest="seeds",
                        help="A list of seeds used for parallel crace",)

    args, extra = parser.parse_known_args(argv)

    if extra:
        error(f"unknown arguments: {extra}")

    return args


def crace_parallel(args=None):
    
    argv = args.copy()

    try:
        idx = argv.index("--")
        args = argv[:idx]
        extra = argv[idx+1:]
    except ValueError:
        args = argv
        extra = []

    if args is None:
        args = parse_arguments()
    else:
        args = parse_arguments(args)

    print("Arguments for parallel crace runs:")
    for k, v in vars(args).items():
        print(f"   {k}: {v}")
    print(f"\nExtra arguments for crace itself: {extra}")


    check_crace()

    execdir_prefix = Path(args.execdir)
    processes = []
    width = max(2, len(str(args.repetitions)))
    for i in range(1, int(args.repetitions) + 1):

        execdir = execdir_prefix / f"{args.expname}-{i:0{width}d}"
        if execdir.exists():
            shutil.rmtree(execdir)

        execdir.mkdir(parents=True)

        runseed = args.seeds + i
        proc = run_crace(execdir, runseed, extra)
        processes.append(proc)

        print(f"\npid: {proc.pid} - submitting crace to {execdir}")

        time.sleep(1)

    # # wait all
    # for p in processes:
    #     p.wait()

    return 0


if __name__ == "__main__":
    sys.exit(crace_parallel())




class CustomFormatter(argparse.HelpFormatter):

    def _format_action_invocation(self, action):
        if not action.option_strings:
            return super()._format_action_invocation(action)

        if action.nargs == 0:
            return ", ".join(action.option_strings)

        metavar = self._format_args(
            action,
            self._get_default_metavar_for_optional(action)
        )

        if len(action.option_strings) == 1:
            return f"{action.option_strings[0]} {metavar}"

        short = action.option_strings[:-1]
        long = action.option_strings[-1]

        return f"{', '.join(short)}, {long} {metavar}"
