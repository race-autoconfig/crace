#!/usr/bin/env python3
# This is a version of crace/scripts/main.py for executing
# crace with cProfile.Profile().

import io
import os
import sys
import pstats
import inspect
import logging
import cProfile

CRACE_HOME = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if CRACE_HOME not in map(os.path.abspath, sys.path):
    sys.path.insert(0, os.path.abspath(CRACE_HOME))

import crace.errors as CE
from crace.containers.scenario import Scenario
from crace.configurator.crace import Crace, Check
from crace.configurator.tester import Tester


asyncio_logger = logging.getLogger("asyncio")
logDir = None


def start_cmdline(arguments=None):
    """
    Function that executes the configuration procedure with arguments from the command line
    """
    data = None
    try:
        scenario = Scenario.from_input(arguments)

        # change to exech directory if provided
        if scenario.options.debugLevel.value >=1:
            print("# Setting working directory: ", scenario.options.execDir.value)
        current_dir = os.getcwd()
        os.chdir(scenario.options.execDir.value)

        if scenario.options.readlogs.value:
            os.chdir(current_dir)
            configurator = Crace(scenario)
            data = configurator.read_log_files()
        elif scenario.options.check.value:
            configurator = Check(scenario)
            _ = configurator.run()
        elif scenario.options.onlytest.value:
            tester = Tester(scenario)
            tester.do_test()
        else:
            # Here configuration
            # TODO: select configurator
            configurator = Crace(scenario)
            _ = configurator.run()
            if scenario.instances.has_test():
                tester = Tester(scenario)
                conf = configurator.get_elite_configurations()
                tester.do_test(conf)

        # return to initial directory
        os.chdir(current_dir)

        global logDir
        logDir = scenario.options.logDir.value

    except Exception as e:
        if any(isinstance(e, cls) for cls in [x[1] for x in inspect.getmembers(CE, inspect.isclass)]):
            pass
        else:
            print("There was an error while executing crace:")
            print(e)
            asyncio_logger.error(e, exc_info=True)
            raise e

    finally:
        if data: return data
        else: sys.exit(1)

def start_profile(arguments=None):
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_cmdline(arguments)

    profiler.disable()
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats()

    profile_log = os.path.join(logDir, 'profile.log')
    with open(profile_log, 'w') as f:
        f.write(s.getvalue())

    sys.exit(1)

if __name__ == '__main__':
    arguments = sys.argv # command line arguments as a list
    start_profile(arguments)
