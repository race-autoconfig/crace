# scripts/crace-parallel.py

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def error(msg):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def check_crace():
    """check crace command exists"""
    from shutil import which

    if which("crace") is None:
        print("# ERROR: 'crace' command not found, is 'crace' installed in current environment?\n")
        print("# Python location:")
        print("  python :", shutil.which("python"))
        print("  python3:", shutil.which("python3"))
        print("\n# PATH:")
        for p in os.environ["PATH"].split(os.pathsep):
            print(" ", p)
        sys.exit(127)


def run_crace(execdir, seed, params):
    print(f"#   submitting crace to {execdir}")

    pid = os.getpid()

    stdout = execdir / f"crace-{pid}.stdout"
    stderr = execdir / f"crace-{pid}.stderr"

    with open(stdout, "w") as out, open(stderr, "w") as err:
        proc = subprocess.Popen(
            ["crace", "--exec-dir", str(execdir), "--seed", str(seed)] + params,
            stdout=out,
            stderr=err,
        )

    return proc


def parse_arguments(argv=None):
    """
    Parse the command line arguments and return them.
    """
    parser = argparse.ArgumentParser(description="Run crace in parallel")

    parser.add_argument("-rp", "--repetitions", default=1, help="The number of parallel crace", dest="rp")
    parser.add_argument("-ep", "--exec-parent", default=os.getcwd(), help="The folder to save the parallel crace results", dest="ep")
    parser.add_argument("-en", "--expname", default="exp", help="The prefix of the folder name to save the parallel crace results", dest="en")
    parser.add_argument("--seeds", nargs='+', default=1234567, help="A list of seeds used for parallel crace", dest="seed")

    args, extra = parser.parse_known_args(argv)

    return args, extra


def crace_parallel(args=None):

    check_crace()

    if args is None:
        args, extra = parse_arguments()
    else:
        args, extra = parse_arguments(args)

    print("# NOTE: calling python using the global environment")

    execdir_prefix = Path(args.ep)

    processes = []

    width = max(2, len(str(args.rp)))
    for i in range(1, int(args.rp) + 1):

        execdir = execdir_prefix / f"{args.en}-{i:0{width}d}"

        if execdir.exists():
            shutil.rmtree(execdir)

        execdir.mkdir(parents=True)

        runseed = args.seed + i

        proc = run_crace(execdir, runseed, extra)

        processes.append(proc)

        time.sleep(1)

    # wait all
    for p in processes:
        p.wait()

    return 0


if __name__ == "__main__":
    sys.exit(crace_parallel())