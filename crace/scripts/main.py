import re
import os
import sys
import inspect
import logging
import traceback

current_file = os.path.normpath(__file__)
CRACE_HOME = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if CRACE_HOME not in map(os.path.abspath, sys.path):
    sys.path.insert(0, os.path.abspath(CRACE_HOME))

import crace.errors as CE
from crace.configurator.tester import Tester
from crace.containers.scenario import Scenario
from crace.configurator.crace import Crace, Check
from crace.scripts.utils import _check_scenario_parameters

asyncio_logger = logging.getLogger("asyncio")


def start_crace(scenario, console: bool=False):
    """
    Main procedure to start crace
    """
    data = None
    # change to exech directory if provided
    if scenario.options.debugLevel.value >=1:
        print("# Setting working directory: ", scenario.options.execDir.value)
    current_dir = os.getcwd()
    if os.path.exists(scenario.options.execDir.value): os.chdir(scenario.options.execDir.value)

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
    if data: return data

def crace_main(scenario: Scenario=None, scenario_file=None, console=False):
    """
    Allowed to be called only in python console
    Scenario(options: CraceOptions,
             parameters: Parameters,
             instances: Instances
             initial_configurations: Configurations/None)

    :param scenario: objective Scenario including options, parameters, instances and initial_configurations
    :param scenario_file: the path of scenario file
    """
    if scenario is None and scenario_file is None:
        scenario_file = os.path.join(os.path.abspath('./'), 'scenario.txt')

    try:
        if scenario is None:
            scenario = Scenario.from_input(scenario_file=scenario_file, console=console)
        else:
            _check_scenario_parameters(scenario)

        start_crace(scenario)

    except Exception as e:
        if any(isinstance(e, cls) for cls in [x[1] for x in inspect.getmembers(CE, inspect.isclass)]):
            pass
        else:
            print("\nERROR: There was an error while executing crace:")
            asyncio_logger.error(e, exc_info=True)
            err_info = traceback.format_exc()
            print(err_info)

def start_cmdline(arguments=None, console: bool=True):
    """
    Function that executes the configuration procedure with arguments from the command line
    """

    scenario=None
    data=None

    try:
        scenario = Scenario.from_input(arguments=arguments, console=console)

        data = start_crace(scenario=scenario, console=console)

        if console: return data

    except KeyboardInterrupt:
        print()

    except SystemExit:
        pass

    except Exception as e:
        if any(isinstance(e, cls) for cls in [x[1] for x in inspect.getmembers(CE, inspect.isclass)]):
            pass
        elif isinstance(e, AssertionError):
            pass
        else:
            print("\nERROR: There was an error while executing crace:")
            asyncio_logger.error(e, exc_info=True)
            traceback.print_exc()
            print(e)

    if not console: sys.exit(1)
    else: return data

def run(*inputs):
    if len(inputs) == 1 and isinstance(inputs[0], list):
        arguments = inputs[0]
    else:
        arguments = list(inputs)
    arguments = [str(x) for x in arguments]
    return crace_cmdline(arguments=arguments)

def crace_cmdline(arguments: list=None, console: bool=True, cli: bool=False):
    """
    Allowed to be called: 1. by entry point, 
                          2. by crace_mpi 
                          3. in python console
    """
    if not isinstance(arguments, list):
        if isinstance(arguments, str):
            arguments = re.split(r"[ ,]+", arguments)

    if cli:
        arguments = arguments

    elif not arguments:
        # arguments may be [] (not None)
        arguments = sys.argv[1:] # command line arguments as a list

    data = start_cmdline(arguments, console)

    if data: return data

if __name__ == '__main__':
    """
    Allowed to be called only by this file
    """
    crace_cmdline(console=False)
