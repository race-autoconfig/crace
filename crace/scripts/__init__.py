# scripts/__init__.py

from crace.scripts.main import crace_cmdline, crace_main, run
from crace.scripts.mpi import start_mpi as crace_mpi
from crace.scripts.parallel import crace_parallel

def crace_guide():
    import os, sys
    import pathlib
    import webbrowser
    try:
        import crace
        dir_crace = pathlib.Path(crace.__file__).resolve().parent
    except:
        dir_crace = pathlib.Path(__file__).resolve().parent.parent

    guide = pathlib.Path(os.path.join(dir_crace, 'vignettes/crace-package.pdf'))

    # not support system without GUI
    if guide.exists():
        webbrowser.open(guide.as_uri())
    else:
        sys.exit(1)

def crace_examples():
    import os, sys
    import pathlib
    import platform
    import subprocess
    try:
        import crace
        dir_crace = pathlib.Path(crace.__file__).resolve().parent
    except:
        dir_crace = pathlib.Path(__file__).resolve().parent.parent

    examples = pathlib.Path(os.path.join(dir_crace, 'inst/examples'))

    # not support system without GUI
    system = platform.system()
    if examples.exists():
        if system == "Darwin":  # macOS
            subprocess.run(["open", examples])
        elif system == "Windows":
            subprocess.run(["explorer", examples])
        else:  # Linux
            subprocess.run(["xdg-open", examples])
    else:
        sys.exit(1)

def crace_run():
    import sys

    if len(sys.argv) > 1:

        if sys.argv[1] == "doc":
            crace_guide()
            return

        if sys.argv[1] == "examples":
            crace_examples()
            return

        # if sys.argv[1] == "mpi":
        #     crace_mpi(args=sys.argv[2:], cli=True)
        #     return

        if sys.argv[1] == "parallel":
            crace_parallel(sys.argv[2:])
            return

    crace_cmdline(sys.argv[1:])