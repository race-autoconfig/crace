import sys
import crace.errors as CE

def check_scenario_parameters(scenario):
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
