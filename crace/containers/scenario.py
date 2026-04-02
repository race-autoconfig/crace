import os
import sys
import math
import random
import pickle

import pandas as pd

from crace.errors import CraceError
from crace.containers.instances import Instances
from crace.containers.parameters import Parameters
from crace.containers.crace_options import CraceOptions
from crace.containers.configurations import Configurations

class Scenario:
    def __init__(self, options=None, parameters=None, instances=None, initial_configurations=None):
        self.options = options
        self.parameters = parameters
        self.instances = instances
        self.initial_configurations = initial_configurations

        # If call Scenario in python console:
        parameters_to_check = {
            "CraceOptions": ('options', options),
            "Parameters": ('parameters', parameters),
            "Instances": ('instances', instances),
        }

        # Error check while executing in python console
        for obj, param in parameters_to_check.items():
            if param[1] is None:
                print("\nERROR: There was an error while loading scenario:")
                sys.tracebacklimit = 0
                raise ValueError(f"The parameter '{param[0]}' should be provided by {obj} object.")

        # by default nbConfigurations is 0, we should calculate it if it is still default
        if options.nbConfigurations.value == options.nbConfigurations.default:
            #TODO: we should check if we still consider this a good number
            #      should this be variable during the race?
            options.nbConfigurations.set_value(math.floor(2 + math.log(self.parameters.nb_parameters, 2)))

        # check nbConfigurations for capping version
        if (options.capping.is_set() and options.capping.value) or options.testExperimentSize.value > 1:
            options.nbConfigurations.set_value(max(options.nbConfigurations.value, options.parallel.value))

        # check modelUpdateBy..
        # set default value of modelUpdateByStep as twice of that in irace
        if options.modelUpdateByStep.value == options.modelUpdateByStep.default and not options.modelUpdateBySampling.value:
            options.modelUpdateByStep.set_value(2 * math.floor(2 + math.log(self.parameters.nb_parameters, 2)))

    @staticmethod
    def from_input(arguments=None, scenario_file: str = None, silent=False, console=False):
        """
        load scenario settings from input

        :ivar: arguments: call crace via providing options in command line
        :ivar: scenario_file: call crace via providing a scenario file
        :ivar: silent: boolean to check from master node or not
        :ivar: console: boolear to check from python console or not
        """
        # neither arguments or scenario file is provided
        if arguments is None and scenario_file is None:
            print("\nERROR: There was an error while loading scenario:")
            print("Either a scenario file or arguments must be provided to load crace scenario")
            raise CraceError("Either a scenario file or arguments must be provided to load crace scenario")

        # options are provided via command line
        elif arguments is not None:
            options = CraceOptions(arguments=arguments, silent=silent, console=console)

        # options are provided via a scenario file
        else:
            options = CraceOptions(scenario_file=scenario_file, silent=silent, console=console)

        if not silent:
            random.seed(options.seed.value)

        # not options.readlogs.is_set(): to execute a crace run
        # not os.path.exists(options.execDir.value): exec dir does not exist
        if not os.path.exists(options.execDir.value) and not options.readlogs.is_set():
            os.mkdir(options.execDir.value)

        # not from an existing crace run
        if options.recoveryDir.value is None and not options.readlogs.is_set():
            # load parameters
            if not silent and options.debugLevel.value >= 1:
                print("# Reading parameter file: " + options.parameterFile.value)
            parameters = Parameters(parameters_file=options.parameterFile.value,
                                    exec_dir=options.execDir.value,
                                    forbidden_file=options.forbiddenFile.value,
                                    forbidden_text=options.forbiddenExps.value,
                                    debug_level=options.debugLevel.value,
                                    silent=silent,
                                    log_base_name=os.path.basename(options.logDir.value))
            if not silent and options.debugLevel.value >= 2:
                print("#   total parameters found: ", parameters.nb_parameters)
            if not silent and options.debugLevel.value >= 3:
                parameters.show_parameters()

            # load instances
            if not silent and options.debugLevel.value >= 1:
                print("# Reading instances ...")
            instances = Instances(instances_list=options.instances.value,
                                  instances_dir=options.trainInstancesDir.value,
                                  instances_file=options.trainInstancesFile.value,
                                  tinstances_dir=options.testInstancesDir.value,
                                  tinstances_file=options.testInstancesFile.value,
                                  exec_dir=options.execDir.value,
                                  deterministic=options.deterministic.value,
                                  debug_level=options.debugLevel.value,
                                  seed=options.seed.value,
                                  shuffle_ins=options.shuffleInstances.value,
                                  sample_ins=options.sampleInstances.value,
                                  silent=silent,
                                  log_base_name=os.path.basename(options.logDir.value))

            # load initial configurations
            if options.configurationsFile.value is not None:
                initial_configurations = None
                if not silent:
                    if options.debugLevel.value >= 1:
                        print("# Reading initial configurations file: " + options.configurationsFile.value)
                    initial_configurations = Configurations(parameters, exec_dir=options.execDir.value,
                                                            recovery_folder=options.recoveryDir.value,
                                                            conf_repair=options.repairConfiguration.value,
                                                            debug_level=options.debugLevel.value,
                                                            log_base_name=os.path.basename(options.logDir.value))

                    try:
                        initial_configurations.add_from_file(configurations_file=options.configurationsFile.value)
                    except Exception as e:
                        print("\nERROR: There was an error while initializing the configuration(s)")
                        print(e)
                        sys.exit(1)

                    if options.debugLevel.value >= 2:
                        print("#   total configurations read: ", initial_configurations.n_config)
                    if options.debugLevel.value >= 3:
                        initial_configurations.print_all()
                    if initial_configurations.n_config == 0:
                        initial_configurations = None
            else:
                initial_configurations = None

        # continue an existing crace run
        elif options.recoveryDir.value is not None:
            if not silent: print("# Recovering execution from folder: " + options.recoveryDir.value)

            # recover parameters
            if not silent and options.debugLevel.value >= 1:
                print("#  recovering parameters...")
            parameters = Parameters(exec_dir=options.execDir.value,
                                    recovery_folder=options.recoveryDir.value,
                                    debug_level=options.debugLevel.value,
                                    silent=silent,
                                    log_base_name=os.path.basename(options.logDir.value))

            # recover instances
            if not silent and options.debugLevel.value >= 1:
                print("#  recovering instances...")
            instances = Instances(exec_dir=options.execDir.value,
                                  recovery_folder=options.recoveryDir.value,
                                  deterministic=options.deterministic.value,
                                  debug_level=options.debugLevel.value,
                                  seed=options.seed.value,
                                  shuffle_ins=options.shuffleInstances.value,
                                  sample_ins=options.sampleInstances.value,
                                  silent=silent,
                                  log_base_name=os.path.basename(options.logDir.value))

            # recover configurations
            if not silent and options.debugLevel.value >= 1:
                print("#  recovering configurations...")
            initial_configurations = Configurations(parameters=parameters, exec_dir=options.execDir.value,
                                                    recovery_folder=options.recoveryDir.value,
                                                    conf_repair=options.repairConfiguration.value,
                                                    debug_level=options.debugLevel.value,
                                                    log_base_name=os.path.basename(options.logDir.value))
            if not silent and options.debugLevel.value >= 1:
                print(f"#   {len(initial_configurations.all_configurations)} configurations recovered")

        # options.readlogs is not None
        # load crace results
        else:
            # load from craceplot: fewer print information
            if options.readLogsInCplot.value < 2:
                print("# Reading log files from folder: " + os.path.dirname(options.readlogs.value))
            
            # load parameters
            if not silent and options.debugLevel.value >= 1:
                print("#  recovering parameters...")
            parameters = Parameters(exec_dir=options.execDir.value,
                                    load_folder=options.readlogs.value,
                                    debug_level=options.debugLevel.value,
                                    silent=silent,
                                    log_base_name=os.path.basename(options.logDir.value))

            # load instances
            if not silent and options.debugLevel.value >= 1:
                print("#  recovering instances...")
            instances = Instances(exec_dir=options.execDir.value,
                                  load_folder=options.readlogs.value,
                                  deterministic=options.deterministic.value,
                                  debug_level=options.debugLevel.value,
                                  seed=options.seed.value,
                                  shuffle_ins=options.shuffleInstances.value,
                                  sample_ins=options.sampleInstances.value,
                                  silent=silent,
                                  log_base_name=os.path.basename(options.logDir.value))

            # load configurations
            if not silent and options.debugLevel.value >= 1:
                print("#  recovering configurations...")
            initial_configurations = Configurations(parameters=parameters,
                                                    load_folder=options.readlogs.value,
                                                    conf_repair=options.repairConfiguration.value,
                                                    debug_level=options.debugLevel.value,
                                                    log_base_name=os.path.basename(options.logDir.value))
            if not silent and options.debugLevel.value >= 1:
                print(f"#   {len(initial_configurations.all_configurations)} configurations recovered")

        # print scenario log for current crace run
        if not silent and not options.readlogs.value:
            options.print_to_log(options.logDir.value + "/scenario.log")

        # finally return Scenario object
        return Scenario(options=options,
                        parameters=parameters,
                        instances=instances,
                        initial_configurations=initial_configurations)

    @staticmethod
    def print_to_log(path, options):
        """
        print options to provided file
        """
        #generic function to add to csv
        if(Scenario.csv_exist(path) == True): # check if csv exist
            pass

        for i in options.__dict__['options']:
            instance_df = pd.DataFrame([{"name": i , "value":options.__dict__[i].value}]) # create a dataframe per dictionary
            instance_df.to_csv(path,mode='a',index = False,header = False)

    @staticmethod
    def csv_exist(path):
        """
        check if provided csv file existing
        """
        #says if the csv exists, if not create it
        if(os.path.isfile(path) == False):
            open(path, 'w')
            return False
        return True

def save_as_object(path, object_):
    file = open(path, "wb")
    pickle.dump(object_, file)
    file.close()

def load_object(path):
    file = open(path, "rb")
    response = pickle.load(file)
    file.close()
    return response

