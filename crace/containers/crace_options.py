import os
import re
import csv
import sys
import copy
import json
import shutil
import random
import textwrap
import importlib
import pandas as pd

import crace.settings.description as csd
from crace.utils.const import bold, underline, reset
from crace.utils import Reader, ConditionalReturn, get_logger, format_string
from crace.errors import OptionError, FileError, ExitError
from crace.containers.crace_option import IntegerOption, RealOption, StringOption, BooleanOption, FileOption, \
    ExeOption, EnablerOption, ListOption

def option_decoder(obj):
    """
    instantiates the correct class for an option

    :param obj: option (dictionary) obtained from the json settings definition
    :return: an instance of the Option class
    :raises OptionError: if type attribute is not found in obj or if type was not recognized
    """
    o = None
    if not ('type' in obj):
        raise OptionError("Option {} has no type.".format(obj['name']))
    if obj['type'] == 'i':
        o = IntegerOption(obj)
    if obj['type'] == 'r':
        o = RealOption(obj)
    if obj['type'] == 'b':
        o = BooleanOption(obj)
    if obj['type'] == 's':
        o = StringOption(obj)
    if obj['type'] == 'p':
        o = FileOption(obj)
    if obj['type'] == 'x':
        o = ExeOption(obj)
    if obj['type'] == 'e':
        o = EnablerOption(obj)
    if obj['type'] == 'l':
        o = ListOption(obj)

    if o is None:
        raise OptionError("Option cannot be read: " + obj['name'])
    return o


class CraceOptions(Reader):
    """
    Class contains the set of options that crace defined for the configurator.
    The options are defined in a setting file provided by the package.
    """
    def __init__(self, scenario_file=None, arguments=None, silent: bool = False, console: bool=False):
        """
        Initialization of an CraceOptions object. Only scenario_file or arguments must be provided.

        :param scenario_file: crace scenario file
        :param arguments: command line arguments provided by the user (if it is None, default options are loaded)
        """
        if scenario_file is not None and arguments is not None:
            OptionError("Either scenario_file or arguments, but not both must be provided to initialize CraceOptions")

        # TODO: option definition file should be set in the module/package configuration..
        # load options definition from json file
        package_dir, file = os.path.split(__file__)
        filename = package_dir + "/../settings/crace_options.json"
        with open(filename, "r") as read_file:
            # if not silent:
            #     print("# Loading crace options...")
            all_options = json.loads(read_file.read(), object_hook=option_decoder)

        # create option variables (default values are put in place)
        self.options = []
        for o in all_options:
            self.options.append(o.name)
            setattr(self, o.name, o)

        # log arguments in case they are needed later
        self.arguments = arguments

        """
        if not os.path.exists(self.recoveryDir.value):
            os.makedirs(self.recoveryDir.value, exist_ok=True)

        #path.parent

        for entry in os.scandir(src):
            if != entry.is_dir():
                shutil.move(entry.path, self.recoveryDir.value)
        """
        # load option values
        try:
            self._load_options(scenario_file, arguments, silent=silent)

            # load log files to continue the crace run
            if self.recoveryDir.value is not None:
                self.recoveryDir.set_value(get_logger(self.recoveryDir.value))
                self._load_from_log(option='recoveryDir', recovery_folder=self.recoveryDir.value,
                                    silent=silent, load=False)
                self.copy_log(self.recoveryDir, self.execDir, self.logDir)

            # load log files to check the procedure
            elif self.readlogs.value is not None:
                self.readlogs.set_value(get_logger(self.readlogs.value))
                self.check_readable(self.readlogs.value)
                self._load_from_log(option='readlogs', recovery_folder=self.readlogs.value,
                                    silent=silent, load=True)

            else: self._set_scenario_dependent_options()

            if self.readlogs.value is None: self.check_options(silent=silent)

        except OptionError as err:
            if not silent:
                print("\n! ERROR: There was an error while reading crace options:")
                print(textwrap.indent(str(err), "!   "))
                sys.tracebacklimit = 0
                raise
            sys.exit(1)
        except FileError as err:
            if not silent:
                print("\n! ERROR: There was an error while reading crace options file:")
                print(textwrap.indent(str(err), "!   "))
                sys.tracebacklimit = 0
                raise
            sys.exit(1)
        except ExitError as err:
            if console:
                pass
            elif not silent:
                sys.tracebacklimit = 0
                raise
            sys.exit(0)
        except Exception as err:
            if not silent:
                print("\n! ERROR: There was an error: ")
                print(textwrap.indent(str(err), "!   "))
                sys.tracebacklimit = 0
                raise
            sys.exit(1)

    def _load_options(self, scenario_file=None, arguments=None, silent: bool = False):
        """
        load crace options from arguments or from scenario file to
        class attributes which are instances of the CraceOption class

        :param scenario_file: crace scenario file path
        :param arguments: command line arguments if provided
        :return: none
        """
        recovery_flag = False
        read_log_flag = False

        # initialize or clean arguments if provided
        if arguments is None:
            arguments = []

        # check for help command
        if self.help.long in arguments:
            self.print_help_long()
            raise ExitError("")

        # check for help command
        if self.help.short in arguments:
            self.print_help_short()
            raise ExitError("")

        # check for version command
        if self.version.short in arguments:
            self.print_version_short()
            raise ExitError("")

        # check for version command
        if self.version.long in arguments:
            self.print_version_long()
            raise ExitError("")

        # check for selected option
        if self.man.long in arguments or self.man.short in arguments:
            if self.man.long in arguments:
                index = arguments.index(self.man.long)
            else:
                index = arguments.index(self.man.short)

            if len(arguments) > index+1:
                if arguments[index+1] not in self.options:
                    raise OptionError(f"{arguments[index+1]} is an incorrect crace option name.")
                else:
                    self.print_man(self.get_option(arguments[index+1]))

            else:
                raise OptionError(f"One option shoud be provided for man.")
            raise ExitError("")

        # copy templates from crace
        if self.initialize.long in arguments:
            print("# Arguments: {}".format(self.arguments))
            self.copy_templates()
            raise ExitError("")

        # specific option called in craceplot
        if not silent and not self.readLogsInCplot.long in arguments:
            print('#------------------------------------------------------------------------------')
            print("# Arguments: {}".format(self.arguments))
            print("# Reading crace options..")

        # check if there is debug level argument
        debug_level = None
        if self.debugLevel.long in arguments:
            i = arguments.index(self.debugLevel.long)
            debug_level = eval(arguments[i+1])

        # check if there is a scenario file in the arguments
        if self.scenarioFile.long in arguments:
            i = arguments.index(self.scenarioFile.long)
            self.scenarioFile.set_value(arguments[i+1])

        elif self.scenarioFile.short in arguments:
            i = arguments.index(self.scenarioFile.short)
            self.scenarioFile.set_value(arguments[i+1])

        # check if there is a recovery file in the arguments
        elif self.recoveryDir.long in arguments or self.recoveryDir.short in arguments:
            recovery_flag = True
            if not silent: print("# Note: Ignoring scenario file as recovery option is set")

        # check if read log files
        elif self.readlogs.long in arguments:
            read_log_flag = True

        # check if the default scenario file exists
        elif self.scenarioFile.exists_file():
            if not silent: print("# Default scenario file was found in current directory")

        # the default scenario file does not exist
        elif scenario_file is not None:
            self.scenarioFile.set_value(scenario_file)
            if not self.scenarioFile.exists_file():
                raise OptionError(f"The provided scenario file {scenario_file} is not readable or does not exist.")

        else:
            # in this case, we remove the scenario file and assume parameters will be handled manually
            if not silent:
                print(f"# No scenario file was provided, setting option scenarioFile to None\n  Loading other options from arguments or default values..")
            self.scenarioFile.set_value(None)

        # If file exists load options from scenario file
        if self.scenarioFile.value and self.scenarioFile.exists_file() and not recovery_flag and not read_log_flag:
            # read options
            try:
                opts = self._read_options_file(path=self.scenarioFile.value, origin=None, silent=silent)
            except Exception:
                raise

            # update debugLevel if needed
            if ("debugLevel" in opts.keys()) and (debug_level is None):
                debug_level = eval(opts["debugLevel"])
            
            # set option variables
            for o in opts.keys():
                self.set_option(o, opts[o])

        # load options from arguments (arguments override options in file)
        options_long = [self.get_option(x).long for x in self.options if self.get_option(x).long]
        options_short = [self.get_option(x).short for x in self.options if self.get_option(x).short]

        self.arguments = copy.deepcopy(arguments)

        for o in self.options:
            if len(arguments) < 1:
                break
            if self.get_option(o).short and self.get_option(o).short in arguments:
                index = arguments.index(self.get_option(o).short)
            elif self.get_option(o).long and self.get_option(o).long in arguments:
                index = arguments.index(self.get_option(o).long)
            else:
                continue

            # no value
            if isinstance(self.get_option(o), EnablerOption): 
                self.set_option(o, True)
                del arguments[index]

            # >= 1 value(s)
            elif isinstance(self.get_option(o), ListOption):
                s = []
                start = index
                end = start + 1
                while (end < len(arguments) and 
                       arguments[end] not in options_long and
                       arguments[end] not in options_short):
                    s.append(arguments[end])
                    end += 1
                for i in reversed(range(start, end)):
                    del arguments[i]
                self.set_option(o, s)

            # 1 value
            else:
                try:
                    self.set_option(o, arguments[index + 1])
                    del arguments[index:(index + 2)]
                except Exception as e:
                    if isinstance(e, OptionError) or isinstance(e, ValueError):
                        raise OptionError(e)
                    else:
                        raise OptionError(f"{self.get_option(o).name} has no value.")

        if len(arguments) > 0:
            raise OptionError("Argument not recognized: " + arguments[0])

    def _read_options_file(self, path, origin=None, silent: bool=False):
        """
        Loads the attributes of the class from a route. If a parameter given
        in the file does not exists it raises AttributeError.

        :param path:  path of the scenario file
        :param origin: paths of the scenario files from which this file was included (None if its not included from file)
        :return: dictionary of options set in the files {"option_name": option_value}
        :raises OptionError: if there is repeated nested scenario files
        :raises FileError: if scenario file cannot be read
        """
        if self.check_readable(path):
            if not silent: print("# Reading scenario file " + os.path.abspath(path))
            lines = self.get_readable_lines(path)
            opts = {} 

            for line in lines:
                # split over the first = or ->
                values = [x.strip().strip("'").strip('"') for x in re.split("=|->", line, 1)]
                # Validate option exists
                if not(values[0] in self.options):
                    raise OptionError("Not recognized crace option " + str(values[0]) + " in scenario file")
                # Validate option is not repeated
                if values[0] in opts.keys():
                    raise OptionError("Duplicated option " + values[0] + " in scenario file " + path)
                opts[values[0]] = values[1]

            if "scenarioFile" in opts.keys():
                # nested scenario files are allow and deepest one will overwrite options in previous
                if origin and opts["scenarioFile"] in origin:
                    raise OptionError("Nested scenario file is repeated " + opts["scenarioFile"])

                else:
                    if not silent: print("# Nested scenario file detected: " + opts["scenarioFile"] + 
                                         " repeated parameters will be replaced")
                    if origin is None:
                        origin = []
                    try:
                        n_opts = self._read_options_file(opts["scenarioFile"], origin=origin.append(path))
                    except RecursionError as e:
                        raise OptionError("Nested scenario file is recursive.")
                    # join options from different files
                    opts = {**opts, **n_opts}

        else:
            raise FileError("Scenario file cannot be read" + str(path))
        return opts

    def _load_from_log(self, option, recovery_folder, silent: bool = False, load: bool = False):
        """
        Load variable values from a logged scenario file

        :param recovery_folder: Folder path to where logs are saved
        """
        log_file = recovery_folder + "/scenario.log"

        if not os.path.exists(log_file):
            raise OptionError("Option " + option + ": provided file " + recovery_folder + " does have scenario.log.")

        # exclude enabler options when recover for continue running
        do_not_recover = ["readlogs"]
        # exclude debuglevel when load results
        if load: do_not_recover += ["debugLevel"]
        # 
        can_modify = ['debugLevel', 'execDir', 'logDir', 'testNbElites']

        rename = {'recoveryFile': 'recoveryDir',
                  'logFile': 'logDir',}

        info = []
        with open(log_file, newline='') as f:
            reader = csv.reader(f)
            data = [(rename.get(row[0], row[0]), *row[1:]) for row in reader]

            for o in data:
                op = self.get_option(o[0])  # obtain class Option
                in_arg = op.short in self.arguments or op.long in self.arguments

                # ignore enabler options, developing options and do_not_recover options
                if op.type == 'e' or op.critical == 7 or o[0] in do_not_recover:
                    continue

                if o[0] in can_modify and in_arg:
                    continue
                elif o[0] not in can_modify and in_arg:
                    info.append(o[0])

                if load:
                    # do not check value when loading results
                    if op.type == 'p':
                        op.set_value(o[1], check_file=False, check_parent_dir=False)
                        continue
                    elif op.type == 'x':
                        op.set_value(o[1], check=False)
                        continue

                op.set_value(o[1])

        if len(info) > 0:
            if not silent: print(f"# Option(s) {', '.join(info)} can not be re-assigned")

    def check_options(self, check_log_files: bool = False, silent: bool=False):
        """
        Checks the crace options are correct
        :return: True if the options are correct
        :raises OptionError: if there is any error in the option values
        """
        # checking parameter file
        self.parameterFile.check_value_file(False)

        # checking execution directory
        self.execDir.check_value_file(True)

        # check if logDir is a relative path
        base_log = os.path.basename(self.logDir.value)
        new_value = os.path.join(self.execDir.value, base_log)
        self.logDir.set_value(new_value, check_parent_dir=False)

        # checking log directory
        if not os.path.exists(self.logDir.value):
            os.makedirs(self.logDir.value, exist_ok=True)

        # check recovery file
        # FIXME: here we should adapt the check
        if self.recoveryDir.is_set():
            self.recoveryDir.check_value_file(False)
            if self.recoveryDir.value == self.logDir.value:
                raise OptionError("Recovery directory must be different from log directory")

        # check training instances
        if self.trainInstancesDir.is_set():
            self.trainInstancesDir.check_value_file(False)
        if self.trainInstancesFile.is_set():
            self.trainInstancesFile.check_value_file(False)
        if not self.trainInstancesDir.is_set() and not self.trainInstancesFile.is_set():
            raise OptionError(f"No instances provided, please provide {bold}trainInstancesDir{reset} "
                              f"or {bold}trainInstancesFile{reset} options")

        # check test instances
        if self.testInstancesDir.is_set():
            self.testInstancesDir.check_value_file(False)
        if self.testInstancesFile.is_set():
            self.testInstancesFile.check_value_file(False)
        if self.onlytest.value and not self.testInstancesDir.is_set() and not self.testInstancesFile.is_set():
            raise OptionError(f"No instances provided, please provide {bold}testInstancesDir{reset} "
                              f"or {bold}testInstancesFile{reset} options when {bold}onlytest{reset} "
                              f"is enabled.")

        # check configurations file
        if self.configurationsFile.is_set():
            self.configurationsFile.check_value_file(True)

        # check forbidden file
        if self.forbiddenFile.is_set():
            self.forbiddenFile.check_value_file(False)

        # check targetrunner file
        # set target runner launcher and file according to the option value (if provided correctly)
        if self.targetRunnerLauncher.is_set():
            try:
                cmd, file = self.targetRunnerLauncher.value
            except Exception:
                raise OptionError(f"Option {underline}targetRunnerLauncher{reset} must include {bold}2{reset} elements: "
                                  f"provided {bold}{', '.join(self.targetRunnerLauncher.value)}{reset}.")
            self.targetRunnerLauncher.set_value(cmd)
            self.targetRunner.set_value(file)
            self.check_executable_cmdline(cmd, file)
        elif self.targetRunner.is_set():
            self.targetRunner.check_executable(True)
            self.targetRunnerLauncher.set_value(None, False)

        # check budgets
        if not self.maxExperiments.is_set() or self.maxExperiments.value == 0:
            if not self.maxTime.is_set() or self.maxTime.value == 0:
                raise OptionError(f"Configuration budget was not provided. Provide the budget "
                                  f"via option {bold}maxExperiments{reset} or {bold}maxTime{reset}")

        # check necessary options for capping version
        if self.capping.is_set() and self.capping.value:
            # check execution budget for capping version
            if not self.maxTime.is_set() or self.maxTime.value == 0:
                raise OptionError(f"Configuration budget should be provided as {bold}maxTime{reset} "
                                  f"when {bold}capping{reset} is active.")
            if not self.boundMax.is_set() or self.boundMax.value <= 0:
                raise OptionError(f"You must provide the maximum execution budget via "
                                  f"{bold}boundMax{reset} option.")

            # check testType and domType
            if not self.domType.is_set() or self.domType.value in (None, "none"):
                self.domType.set_value("adaptive-dom")

            # check elitist
            if not self.elitist.value:
                self.elitist.set_value(1)
                if self.debugLevel.value > 0:
                    if not silent: print(f"# Option {bold}elitist{reset} must be True when "
                                         f"{bold}capping{reset} is enabled.")
    
        # check necessary options for quality problems
        else:
            # testType and domType must not be None at the same time
            if not self.testType.is_set() or self.testType.value in (None, "none"):
                if self.domType.is_set() or self.domType.value in (None, "none"):
                    self.testType.set_value("F-test")

            # print info when providing capping options
            if self.boundMax.is_set() and not self.boundMax.is_default() > 0:
                print(f"Option {bold}boundMax{reset} must not be specified when "
                      f"{bold}capping{reset} is disabled.")
            if self.boundDigits.is_set() and not self.boundDigits.is_default():
                print(f"Option {bold}boundDigits{reset} must not be specified when "
                      f"{bold}capping{reset} is disabled.")
            # if self.boundAsTimeout.is_set() and not self.boundAsTimeout.is_default():
            #     raise OptionError(f"Option boundAsTimeout must not be specified when capping is disabled.")

        # check the value of firstTest
        if not self.firstTest.is_set() or self.firstTest.is_default():
            if self.capping.is_set() and self.capping.value:
                self.firstTest.set_value(1)
            else:
                self.firstTest.set_value(5)

        if self.eachTest.is_set() and self.eachTest.value > self.firstTest.value:
            raise OptionError(f"Option {bold}eachTest{reset} must not greater than "
                              f"{bold}firstTest{reset}.")

        if self.mpi.value and self.parallel.value < 2:
            raise OptionError(f"Option {bold}parallel{reset} must be > 1 if {bold}mpi{reset} is enabled.")
        
        if self.globalModel.value and not self.elitist.value:
            raise OptionError(f"Option {bold}elitist{reset} must be activated when option "
                              f"{bold}globalModel{reset} is enabled.")

        return True

    def _set_scenario_dependent_options(self):
        if self.maxNbElites.is_default():
            self.maxNbElites.set_value(5)
        if self.seed.is_default():
            #FIXME: check the added seed
            self.seed.set_value(random.randint(1, 9999999))
        self.deterministic.set_value(0)

    def set_option(self, option_name, option_value):
        """
        set the value of an option in the CraceOptions instance object

        :param option_name: name of the option
        :param option_value: value to be set
        :return: none
        :raises OptionError: if the option name is unknown
        """
        if option_name not in self.options:
            raise OptionError("Attempt to set unknown option " + option_name + " with value " + option_value)
        getattr(self, option_name).set_value(option_value)

    def get_option(self, option_name):
        """
        get a option in the CraceOption instance object

        :param option_name: name of the option to be returned
        :return: value of the option
        :raises OptionError: if the option name is unknown
        """
        if option_name not in self.options:
            raise OptionError("Attempt to get unknown option " + option_name)
        return getattr(self, option_name)

    def print_help_long(self):
        self.print_crace_header()

        package_dir= os.path.dirname(os.path.dirname(__file__))
        filename = package_dir + "/settings/crace_options.json"
        with open(filename, "r") as read_file:
            all_options = json.loads(read_file.read(), object_hook=option_decoder)

        options = []
        for o in all_options:
            if o.critical > 1:
                continue
            options.append(o.name)

            print(f"{bold}# Name:{reset} {o.name}")
            print(f"  {bold}Long:{reset} {o.long if o.long else 'None'}")
            print(f"  {bold}Short:{reset} {o.short if o.short else 'None'}")
            if o.type in ['i', 's']:
                print(f"  {bold}Domain:{reset} {o.domain if o.domain else 'None'}")
            if o.default is not None and o.default != "" and o.default != []:
                print(f"  {bold}Default:{reset} {o.default}")
            else:
                print(f"  {bold}Default:{reset} None")
            print(f"""{format_string(f"{bold}Description:{reset} {o.description}")}""")
            print(f"""{format_string(f"{bold}Vignettes:{reset} {o.vignettes}")}""")
            print()

        print(f'\n# call with {bold}--man [option_name]{reset} to check the details of specified option')
        print('#------------------------------------------------------------------------------')

    def print_help_short(self):
        self.print_crace_header()

        package_dir= os.path.dirname(os.path.dirname(__file__))
        filename = package_dir + "/settings/crace_options.json"
        with open(filename, "r") as read_file:
            all_options = json.loads(read_file.read(), object_hook=option_decoder)

        options = []
        for o in all_options:
            if o.critical > 1:
                continue
            options.append(o.name)
            print("# {:<25}{:<5}{}".format(o.name, o.short,o.long))
        print(f'\n# call with {bold}--man [option_name]{reset} to check the details of specified option')
        print('#------------------------------------------------------------------------------')

    def print_man(self, o):
        """
        print selected options: critical <= 1
        """
        if o.critical > 1 and o.critical <= 4:
            print(f"Option {bold}{o.name}{reset} is not available in crace currently.")
            raise ExitError("")
        elif o.critical > 4 and o.critical <7:
            print(f"{bold}{o.name}{reset} is not an available crace option.")
            raise ExitError("")

        print(f"{bold}Name:{reset} {o.name}")
        print(f"{bold}Long:{reset} {o.long if o.long else 'None'}")
        print(f"{bold}Short:{reset} {o.short if o.short else 'None'}")
        if o.type in ['i', 's']:
            print(f"{bold}Domain:{reset} {o.domain if o.domain else 'None'}")
        if o.default is not None and o.default != "" and o.default != []:
            print(f"{bold}Default:{reset} {o.default}")
        else:
            print(f"{bold}Default:{reset} None")
        print(f"""\n{format_string(f"{bold}Description:{reset} {o.description}", hanging=0)}\n""")
        print(f"""{format_string(f"{bold}Vignettes:{reset} {o.vignettes}", hanging=0)}""")

    def print_version_long(self):
        self.print_crace_header()
        self.print_cite_info()

    def print_version_short(self):
        print(f"{csd.package_name} {csd.version}")

    def print_all(self):
        """
        Prints all options: critical <= 1
        """
        print("# Scenario options:")
        scenario = ("# Scenario options:\n")
        for o in self.options:
            if self.get_option(o).critical > 1:
                continue
            if self.get_option(o).name != "testType":
                if self.get_option(o).type == "e":
                    continue
                if self.get_option(o).value is None:
                    continue
                if self.get_option(o).type == "l" and len(self.get_option(o).value) < 1:
                    continue
            print("#   " + self.get_option(o).name + ": " + str(self.get_option(o).value))
            scenario += f"#   {self.get_option(o).name}: {str(self.get_option(o).value)}\n"
        
        return ConditionalReturn(scenario=scenario)

    def print_selects(self):
        """
        print selected options: critical == 1
        """
        print("# Scenario options:")
        for o in self.options:
            if self.get_option(o).critical > 1:
                continue
            if self.get_option(o).critical == 0:
                if self.get_option(o).type == "e":
                    continue
                if self.get_option(o).value is None:
                    continue
                print("#   " + self.get_option(o).name + ": " + str(self.get_option(o).value))

    def print_to_log(self, log_file):
        file = open(log_file, "w")
        file.close()
        for o in self.options:
            if self.get_option(o).critical > 1:
                continue
            # create a dataframe
            instance_df = pd.DataFrame([{"name": self.get_option(o).name , "value": self.get_option(o).value}])
            instance_df.to_csv(log_file, mode='a', index = False, header = False)

    def move_log(self, path_src, path_dst):
        if not os.path.exists(path_dst):
            os.makedirs(path_dst, exist_ok=True)

        for entry in os.scandir(path_src):
            if entry.is_dir() != True:
                shutil.move(entry.path, path_dst)

    def copy_log(self, option_rec, option_exec, option_log):
        try:
            self.check_readable(option_rec.value)
        except FileError:
            raise OptionError(f"Option recoveryDir provided file {option_rec.value} is not readable.")

        path_src = option_rec.value
        path_base = os.path.basename(option_log.value)
        path_dst = os.path.join(option_exec.value, path_base)

        if not os.path.exists(path_dst):
            os.makedirs(path_dst, exist_ok=True)

        for entry in os.scandir(path_src):
            if entry.is_dir() != True:
                shutil.copy(entry.path, path_dst)

    def copy_templates(self):
        """
        Initialize the tuning directory with template config files. 
        Two scenarios and some script files are copied to current folder.
        """
        dest = os.getcwd()

        spec = importlib.util.find_spec('crace')
        if spec is None or spec.origin is None:
            raise FileNotFoundError(f"! Error: Package {'crace'} is not installed.")
        package_path = os.path.dirname(spec.origin)

        source = os.path.join(package_path, 'inst/templates')

        if not os.path.exists(source):
            raise FileNotFoundError(f"! Error: The file {source} does not exist in the package {'crace'}.")

        destination = os.path.join(dest, 'templates')

        # check if destination already exists
        if os.path.exists(destination):
            ans = input(
                f"! WARNING: destination directory already exists:\n"
                f"!   {destination}\n"
                f"! Overwrite existing files? [Y/n]: "
            ).strip().lower()

            # default
            if ans not in ("", "y", "yes"):
                print("! Copy aborted.")
                return

        shutil.copytree(source, destination, dirs_exist_ok=True)
        print(f"# Copied templates from package {bold}crace{reset} to {destination}")

    def print_crace_header(self):
        print('#------------------------------------------------------------------------------')
        print(f'# {csd.description}')
        print(f'# Version: {csd.version}')
        print(f'# {csd.copyright}')
        print('#')
        print('# Authors: ')
        for x in csd.authors: print(f'#   {x}')
        print('#')
        print('# Contributors: ')
        for x in csd.contributers: print(f'#   {x}')
        print('#')
        print(f'# Contact:\n#   {csd.contact} ({csd.contact_email})')
        print('#')
        print(f'# Check more details at {csd.url} ')
        line = format_string(csd.license, hanging=0, space=True)
        print("\n".join(f"# {line}" for line in line.splitlines()))
        print('#')
        current_file = os.path.normpath(__file__)
        CRACE_HOME = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        print(f'# installed at: {CRACE_HOME}')
        if self.arguments: print(f'# called with: {" ".join(self.arguments)}')
        print('#------------------------------------------------------------------------------')

    def print_cite_info(self):
        cit = format_string(csd.citiation, width=100, hanging=0, space=True)
        print("\n".join(f"# {line}" for line in cit.splitlines()))
        print('#------------------------------------------------------------------------------')
