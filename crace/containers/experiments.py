
import os
import ast
import csv
import json
import math
import time
import itertools
import numpy as np
import pandas as pd

from typing import Dict, List
from collections import Counter, defaultdict

from crace.utils.const import DELTA
from crace.errors import CraceError, FileError
from crace.containers.instances import InstanceEntry
from crace.containers.configurations import ConfigurationEntry, Configurations


# Define experiments as a list since append is much cheaper
# https://stackoverflow.com/questions/13784192/creating-an-empty-pandas-dataframe-then-filling-it/56746204#56746204
class ExperimentEntry:
    """
    Class ExperimentEntry
    Contains information about a experiment and its current state

    :ivar experiment_id: Unique ID for the experiment
    :ivar configuration_id: ID of configuration
    :ivar instance_id: ID of the instance
    :ivar budget: Budget assigned for the experiment
    :ivar bound_max: Default budget assigned for experiments on new instance.
    :ivar cmd_line: Full execution line of the experiment
    :ivar param_eval: Parameters command line
    :ivar quality: Result of the experiment (configuration objetive)
    :ivar time: Time of the experiment reported by the user
    :ivar state: State of the experiment (posible values: pending/waiting/testing/finished/discarded)
    :ivar creation_time: Time stamp for when the experiment was created (does not imply execution)
    :ivar start_time: Time stamp for when the experiment started execution
    :ivar end_time: Time stamp for when the experiment finished execution
    """
    def __init__(self, experiment_id, configuration_id, instance_id, budget, budget_format, bound_max, bound_max_format, cmd_line,
                 param_line=None, quality=None, exp_time=None, state="pending", creation_time=None,
                 start_time=None, end_time=None):
        """
        Creates an instance of ExperimentEntry

        :param experiment_id: Unique ID for the experiment
        :param configuration_id: ID of configuration
        :param instance_id: ID of the instance
        :param budget: Budget assigned for the experiment
        :param cmd_line: Execution line of the experiment
        :param param_line: Parameters command line
        :param quality: Result of the experiment (configuration objective)
        :param exp_time: Time of the experiment reported by the user
        :param state: State of the experiment ( possible values: pending/waiting/testing/finished/discarded)
        :param creation_time: Time stamp for when the experiment was created (does not imply execution)
        :param start_time: Time stamp for when the experiment started execution
        :param end_time: Time stamp for when the experiment finished execution
        """
        self.experiment_id = experiment_id
        self.configuration_id = configuration_id
        self.instance_id = instance_id
        self.budget = budget
        self.budget_format = budget_format
        self.bound_max = bound_max
        self.bound_max_format = bound_max_format
        self.cmd_line = cmd_line
        self.param_line = param_line
        self.quality = quality
        self.time = exp_time
        assert state in ["pending", "waiting", "testing", "finished", "discarded"], "State provided to experiments " + \
                                                                         state + "not recognized"
        self.state = state
        if creation_time is None:
            self.creation_time = time.localtime()
        else:
            self.creation_time = creation_time
        self.start_time = None
        self.end_time = None
        if start_time is not None:
            self.start_time = start_time
        if end_time is not None:
            self.end_time = end_time

    def __eq__(self, other):
        if not isinstance(other, ExperimentEntry):
            raise CraceError("Attempt to compare experiment to non experiment")
        if self.experiment_id == other.experiment_id:
            return True
        return False

    def as_dict(self):
        """
        Gets a dictionary of class variables
        """
        # FIXME: add time variables if needed
        return {"experiment_id": self.experiment_id, "configuration_id": self.configuration_id,
                "instance_id": self.instance_id, "budget": self.budget, "bound_max": self.bound_max,
                "cmd_line": self.cmd_line, "quality": self.quality, "time": self.time, "state": self.state,
                "creation_time": self.creation_time, "start_time": self.start_time, "end_time": self.end_time}

    def set_start_time(self, start_time):
        self.start_time = start_time

    def set_end_time(self, end_time):
        self.end_time = end_time

    def get_exec_line(self):
        if self.budget is not None and self.budget != 0:
            command_line = self.cmd_line + " " + self.budget_format + " " + self.bound_max_format + " " + self.param_line
        else:
            command_line = self.cmd_line + " " + self.param_line

        return command_line


class Experiments:
    """
    Class Experiments, manages a set of experiments and their states

    :ivar all_experiments: Dictionary of all ExperimentEntry objects by experiment ID (key)
    :ivar experiments_by_configuration: Dictionary of lists of experiments IDs by configuration ID (key)
    :ivan experiments_by_instance: Dictionary of lists of experiments IDs by instance ID (key)
    :ivar pending_ids: List of experiment IDs that have pending state
    :ivar pending_by_instance: Dictionary of lists of pending experiments IDs by instance ID (key)
    :ivar n_waiting: Number of experiments waiting for evaluation (target evaluator setup)
    :ivar waiting_ids: List of experiments ids that are waiting for evaluation (target evaluator setup)
    :ivar n_discarded: Number of discarded experiments
    :ivar discarded_ids: List of experiments that were discarded
    :ivar n_complete: Number of finished experiments (finished state)
    :ivar n_complete_by_config: Dictionary of the number of finished experiments by configuration ID (key)
    :ivar n_experiments: Number of experiments in the class (all states included)
    :ivar n_experiments_by_instance: Dictionary of the number of experiments (all states) by instance ID (key)
    :ivar total_time: Sum of reported times in the experiments (ExperimentEntry.time)
    :ivar submitted_log_file: Filename to log submitted experiments
    :ivar finished_log_file: Filename to log finished experiments
    :ivar canceled_log_file: Filename to log canceled experiments
    """
    def __init__(self, log_folder, budget_digits:int=0, recovery_folder=None, read_folder=None, capping=False, log_level=0, configurations:Configurations=None, test_folder=None, onlytest=False):
        """
        Creates a Experiments object instance

        :param log_folder: File path for log experiments
        :param budget_digits: Number that indicates how many decimal numbers are allowed for the budget
        :param recovery_folder: Folder that contains log files
        :param test_folder: Folder that contains test log files, used to load test results
        """
        self.exps_view = None

        self.all_experiments = {}
        # most of the recovering of this data will be given a configuration
        self.experiments_by_configuration = {}
        self.experiments_by_instance = {}

        # pending experiments (submitted)
        self.pending_ids = []
        self.pending_by_instance = {}

        # waiting experiments (to be evaluated, only when target eval is used)
        self.n_waiting = 0
        self.waiting_ids = []

        # testing experiments (to be evaluated, only when testExperimentSize is greater than 1)
        self.n_testing = 0
        self.testing_ids = []

        # discarded experiments
        self.n_discarded = 0
        self.discarded_ids = []

        # completed experiments
        self.n_complete = 0
        self.n_complete_by_config = {}
        self.n_complete_by_instance = {}

        # experiment counters
        self.n_experiments = 0
        self.n_experiments_instance = {}

        # aux vars
        self.total_time = 0
        self.budget_digits = budget_digits

        # capping or not
        self.capping = capping
        if self.capping:
            self.old_elites = {}
            self.priori_bounds = {}

        self.onlytest = onlytest

        self.log_level = log_level
        # log files
        self.submitted_log_file = log_folder + "/exps_sub.log"
        self.finished_log_file = log_folder + "/exps_fin.log"
        self.canceled_log_file = log_folder + "/exps_disc.log"

        if recovery_folder is not None:
            self.configurations = configurations
            self.load_from_log(recovery_folder)
            print("# Recovering ", self.n_experiments, " experiments from log files")
            print("#   Finished: ", self.n_complete, " Pending: ", len(self.pending_ids))
        elif read_folder is not None:
            self.configurations = configurations
            self.load_from_log(read_folder)
        elif test_folder is not None:
            self.configurations = configurations
        else:
            file = open(self.submitted_log_file, 'w')
            file.close()
            file = open(self.finished_log_file, 'w')
            file.close()
            file = open(self.canceled_log_file, 'w')
            file.close()


    #######################################################
    ## Functions for creating and modifiying experiments ##
    #######################################################

    def from_dict(self, data):
        """
        Creates a new ExperimentEntry from a dict
        """
        #FIXME: this might be better if listing all variables and recovering
        _int = int
        _float = float
        _parse_time = self._parse_time

        t_creat = _parse_time(data['creation_time'])
        t_start = _parse_time(data['start_time'])
        t_end = _parse_time(data['end_time'])

        budget = _float(data['budget'])
        budget_format = f"{budget:.{self.budget_digits}f}"
        bound_max = _float(data.get("bound_max") if "bound_max" in data else data.get("boundMax"))
        bound_max_format = f"{bound_max:.{self.budget_digits}f}"

        param_line = self.configurations.get_cmd_from_configuration_id(data["configuration_id"])

        return ExperimentEntry(experiment_id=_int(data["experiment_id"]),
                               configuration_id=_int(data["configuration_id"]),
                               instance_id=_int(data["instance_id"]),
                               budget=budget,
                               budget_format=budget_format,
                               bound_max=bound_max,
                               bound_max_format=bound_max_format,
                               cmd_line=data["cmd_line"],
                               param_line=param_line,
                               quality=data["quality"],
                               exp_time=data["time"],
                               state=data["state"],
                               creation_time=t_creat,
                               start_time=t_start,
                               end_time=t_end)

    @staticmethod
    def _parse_time(v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            # unix -> float
            return float(v)
        if isinstance(v, time.struct_time):
            # struct_time -> unix -> float
            return float(time.mktime(v))
        if isinstance(v, (list, tuple)):
            # list / tuple -> struct_time -> unix -> float
            return float(time.mktime(time.struct_time(v)))

        # string fallback
        try:
            return float(v)
        except (ValueError, TypeError):
            # list / tuple -> struct_time -> unix -> float
            return float(time.mktime(time.struct_time(ast.literal_eval(v))))

    def from_csv(self, exp):
        """
        Creates a new ExperimentEntry from a csv line
        """
        _int = int
        _float = float
        _parse_time = self._parse_time

        exp_id = _int(exp.experiment_id)
        config_id = _int(exp.configuration_id)
        ins_id = _int(exp.instance_id)

        budget = _float(exp.budget)
        bound_max = _float(exp.bound_max)
        budget_format = f"{budget:.{self.budget_digits}f}"
        bound_max_format = f"{bound_max:.{self.budget_digits}f}"

        t_creat = _parse_time(exp.creation_time)
        t_start = _parse_time(exp.start_time)
        t_end = _parse_time(exp.end_time)

        param_line = self.configurations.get_cmd_from_configuration_id(_int(exp.configuration_id))

        return ExperimentEntry(experiment_id=exp_id,
                               configuration_id=config_id,
                               instance_id=ins_id,
                               budget=budget,
                               budget_format=budget_format,
                               bound_max=bound_max,
                               bound_max_format=bound_max_format,
                               cmd_line=exp.cmd_line,
                               param_line=param_line,
                               quality=exp.quality,
                               exp_time=exp.time,
                               state=exp.state,
                               creation_time=t_creat,
                               start_time=t_start,
                               end_time=t_end)

    @staticmethod
    def get_cmdline_arguments(configuration: ConfigurationEntry, instance: InstanceEntry):
        """
        Generates the full command line string of a experiment for target runner execution

        :param configuration: Configuration object to be executed
        :param instance: Instance object to be executed
        :param budget: budget assigned to the experiment

        :return: Command line string, param_line
        """
        configuration_id = configuration.get_id()
        instance_id = instance.instance_id
        instance_name = instance.path
        instance_seed = instance.seed

        param_line = configuration.get_command()

        command_line = str(configuration_id) + " " + str(instance_id) + " " + \
                        str(instance_seed) + " " + instance_name

        return command_line, param_line

    def new_experiment(self, configuration: ConfigurationEntry, instance: InstanceEntry, budget=None, bound_max=None):
        """
        Created a new ExperimentEntry and calls the function to add it as a pending experiment

        :param configuration: configuration object that must be executed in the experiment
        :param instance: instance object that must be executed in the experiment
        :param budget: budget asigned for the experiment

        :return: experiment.id and ExperimentEntry of the created experiment
        """
        #FIXME: add command line to data and budget
        experiment_id = self.n_experiments + 1
        configuration_id = configuration.get_id()
        instance_id = instance.instance_id
        budget_format = '{0:.{prec}f}'.format(budget, prec=self.budget_digits)
        bound_max_format = '{0:.{prec}f}'.format(bound_max, prec=self.budget_digits)
        cmd_line, param_line = self.get_cmdline_arguments(configuration, instance)
        experiment = ExperimentEntry(experiment_id=experiment_id,
                                     configuration_id=configuration_id,
                                     instance_id=instance_id,
                                     budget=budget,
                                     budget_format=budget_format,
                                     bound_max=bound_max,
                                     bound_max_format=bound_max_format,
                                     cmd_line=cmd_line,
                                     param_line=param_line,
                                     quality=None,
                                     exp_time=None,
                                     state="pending")
        self._log_experiment(self.submitted_log_file, experiment)
        return self._add_experiment(experiment)

    def _add_experiment(self, experiment, _from_parent=False):
        """
        Adds an experiment to the object. Note: to add a new experiment use new_experiment

        :param experiment_id: Experiment ID
        :param configuration_id: Configuration ID
        :param instance_id: Instance ID
        :param experiment: ExperimentEntry object

        :return: experiment.id and ExperimentEntry of the experiment
        """
        experiment_id = experiment.experiment_id
        configuration_id = experiment.configuration_id
        instance_id = experiment.instance_id

        # TODO: Check if an experiment with the same ID existed already?
        self.all_experiments[experiment.experiment_id] = experiment
        self.n_experiments = self.n_experiments + 1

        if configuration_id not in self.experiments_by_configuration.keys():
            self.experiments_by_configuration[configuration_id] = []
        self.experiments_by_configuration[configuration_id].append(experiment_id)
        if configuration_id not in self.n_complete_by_config.keys():
            self.n_complete_by_config[configuration_id] = 0

        if instance_id not in self.n_complete_by_instance.keys():
            self.n_complete_by_instance[instance_id] = 0

        if instance_id not in self.experiments_by_instance.keys():
            self.experiments_by_instance[instance_id] = []
        self.experiments_by_instance[instance_id].append(experiment_id)
        if instance_id not in self.n_experiments_instance.keys():
            self.n_experiments_instance[instance_id] = 0
        self.n_experiments_instance[instance_id] = self.n_experiments_instance[instance_id] + 1

        if experiment.state == "pending":
            self.pending_ids.append(experiment_id)
            if instance_id not in self.pending_by_instance.keys():
                self.pending_by_instance[instance_id] = []
            self.pending_by_instance[instance_id].append(experiment_id)

        elif experiment.state == "finished":
            # loading an already finished experiment (recovery)
            self.n_complete += 1
            self.n_complete_by_config[configuration_id] += 1
            self.n_complete_by_instance[instance_id] += 1
            self.total_time = self.total_time + experiment.time

        elif experiment.state == "discarded":
            self.n_discarded += 1
            self.discarded_ids.append(experiment_id)

        elif experiment.state == "waiting":
            self.n_waiting += 1
            self.waiting_ids.append(experiment_id)

        elif experiment.state == "testing":
            self.n_testing += 1
            self.testing_ids.append(experiment_id)

        else:
            raise CraceError("Experiment state " + experiment.state + " not recognized")

        # sync only when parent call
        if (not _from_parent) and hasattr(self, "exps_view") and self.exps_view:
            self.exps_view._add_experiment_from_parent(experiment)

        return experiment_id, self.get_experiment_by_id(experiment_id, as_dataframe=False)

    def new_experiments(self, configurations: List, instances: list, budgets: dict):
        """
        Creates a set of new pending experiments

        :param configurations: List of configurations, configuration IDs as key
        :param instances: List of Instance objects
        :param budget: List of execution budgets for the experiments (based on instances)

        :return: List of new experiment IDs and list of new ExperimentEntry objects
        """
        new_exps = []
        new_ids = []
        if type(configurations) is dict:
            configurations = configurations.values()

        for instance in instances:
            for configuration in configurations:
                exp_id, exp = self.new_experiment(configuration, instance, budgets[0], budgets[0]) if not self.capping else \
                                self.new_experiment(configuration, instance, budgets[instance.instance_id], budgets[0])
                new_exps.append(exp)
                new_ids.append(exp_id)

        return new_ids, new_exps

    def complete_experiment(self, experiment_id, solution_quality, budget=0, budget_format=0, exp_time=None,
                            start_time=None, end_time=None, _from_parent=False):
        """
        Sets an experiment as finished and adds the ExperimentEntry.end_time time stamp

        :param experiment_id: Experiment ID
        :param solution_quality: Result reported by the execution
        :param exp_time: Time reported by the execution

        :return: ExperimentEntry of the finished experiment
        """
        assert experiment_id in self.all_experiments, "Experiment " + str(experiment_id) + " is not in the index."
        assert self.all_experiments[experiment_id].experiment_id == experiment_id, \
            f"Stored experiment ID does not match index. The index was {experiment_id} and the stored ID was " \
            f"{self.all_experiments[experiment_id].experiment_id}"

        prev_state = self.all_experiments[experiment_id].state

        assert prev_state != "finished", "Attempt to complete already finished experiment id " + str(experiment_id)

        self.all_experiments[experiment_id].budget = budget
        self.all_experiments[experiment_id].budget_format = budget_format
        self.all_experiments[experiment_id].quality = solution_quality
        self.all_experiments[experiment_id].time = exp_time
        self.all_experiments[experiment_id].state = "finished"
        self.all_experiments[experiment_id].start_time = start_time
        self.all_experiments[experiment_id].end_time = end_time
        #self.all_experiments[experiment_id].end_time = time.localtime()
        self.total_time = self.total_time + exp_time

        if not math.isinf(solution_quality) or math.isnan(solution_quality):
            # log the experiment termination
            self._log_experiment(self.finished_log_file, self.all_experiments[experiment_id])
        else:
            self._log_experiment(self.canceled_log_file, self.all_experiments[experiment_id])

        if experiment_id in self.pending_ids:
            pen_id = self.pending_ids.index(experiment_id)
            del self.pending_ids[pen_id]

        instance_id = self.all_experiments[experiment_id].instance_id
        if experiment_id in self.pending_by_instance[instance_id]:
            pen_id = self.pending_by_instance[instance_id].index(experiment_id)
            del self.pending_by_instance[instance_id][pen_id]

        if instance_id not in self.n_complete_by_instance.keys():
            self.n_complete_by_instance[instance_id] = 0
        self.n_complete_by_instance[instance_id] += 1

        id_config = self.all_experiments[experiment_id].configuration_id
        if id_config not in self.n_complete_by_config.keys():
            self.n_complete_by_config[id_config] = 0
        self.n_complete_by_config[id_config] += 1

        self.n_complete += 1

        # discarded experiments can be finished before been terminated when
        # discarded, in that case, we remove it from the discarded experiments
        if prev_state == "discarded":
            self.n_discarded -= 1
            dis_id = self.discarded_ids.index(experiment_id)
            del self.discarded_ids[dis_id]

        if prev_state == "waiting":
            self.n_waiting -= 1
            wait_id = self.waiting_ids.index(experiment_id)
            del self.waiting_ids[wait_id]

        if prev_state == "testing":
            self.n_testing -= 1
            test_id = self.testing_ids.index(experiment_id)
            del self.testing_ids[test_id]

        # sync only when parent call
        if (not _from_parent) and hasattr(self, "exps_view") and self.exps_view:
            self.exps_view._complete_experiment_from_parent(self.all_experiments[experiment_id])

        return self.all_experiments[experiment_id]

    def complete_experiment_to_test(self, experiment_id, solution_quality, budget=0, budget_format=0, exp_time=None,
                            start_time=None, end_time=None, _from_parent=False):
        """
        called when option testExperimentSize is activated
        set state as "testing"
        """
        assert experiment_id in self.all_experiments, "Experiment " + str(experiment_id) + " is not in the index."
        assert self.all_experiments[experiment_id].experiment_id == experiment_id, \
            f"Stored experiment ID does not match index. The index was {experiment_id} and the stored ID was " \
            f"{self.all_experiments[experiment_id].experiment_id}"
        prev_state = self.all_experiments[experiment_id].state

        assert prev_state != "finished", "Attempt to complete already finished experiment id" + str(experiment_id)

        self.all_experiments[experiment_id].budget = budget
        self.all_experiments[experiment_id].budget_format = budget_format
        self.all_experiments[experiment_id].quality = solution_quality
        self.all_experiments[experiment_id].time = exp_time
        self.all_experiments[experiment_id].state = "testing"
        self.all_experiments[experiment_id].start_time = start_time
        self.all_experiments[experiment_id].end_time = end_time
        #self.all_experiments[experiment_id].end_time = time.localtime()
        self.total_time = self.total_time + exp_time

        if math.isinf(solution_quality) or math.isnan(solution_quality):
            self._log_experiment(self.canceled_log_file, self.all_experiments[experiment_id])

        if experiment_id in self.pending_ids:
            pen_id = self.pending_ids.index(experiment_id)
            del self.pending_ids[pen_id]

        instance_id = self.all_experiments[experiment_id].instance_id
        if experiment_id in self.pending_by_instance[instance_id]:
            pen_id = self.pending_by_instance[instance_id].index(experiment_id)
            del self.pending_by_instance[instance_id][pen_id]

        if instance_id not in self.n_complete_by_instance.keys():
            self.n_complete_by_instance[instance_id] = 0
        self.n_complete_by_instance[instance_id] += 1

        if prev_state == "discarded":
            self.n_discarded -= 1
            dis_id = self.discarded_ids.index(experiment_id)
            del self.discarded_ids[dis_id]

        self.n_complete += 1
        self.n_testing += 1
        self.testing_ids.append(experiment_id)

        id_config = self.all_experiments[experiment_id].configuration_id
        if id_config not in self.n_complete_by_config.keys():
            self.n_complete_by_config[id_config] = 0
        self.n_complete_by_config[id_config] += 1

        # sync only when parent call
        if (not _from_parent) and hasattr(self, "exps_view") and self.exps_view:
            self.exps_view._complete_experiment_from_parent(self.all_experiments[experiment_id])

        return self.all_experiments[experiment_id]

    def complete_experiments(self, experiments_ids):
        """
        Update counters for experiments which are watiting to be tested
        """
        for experiment_id in experiments_ids:
            if experiment_id in self.testing_ids:
                # update complete experiments
                self.all_experiments[experiment_id].state = "finished"
                self._log_experiment(self.finished_log_file, self.all_experiments[experiment_id])

                # update testing experiments
                self.n_testing -= 1
                test_id = self.testing_ids.index(experiment_id)
                del self.testing_ids[test_id]

    def complete_experiment_time(self, experiment_id, exp_time, start_time=None, end_time=None):
        """
        Sets execution time of an experiment and adds the ExperimentEntry.end_time time stamp.
        To complete an experiment later the complete_experiment_quality must be used.

        :param experiment_id: Experiment ID
        :param exp_time: Time reported by the execution

        :return: ExperimentEntry of the finished experiment
        """
        assert experiment_id in self.all_experiments, "Experiment " + str(experiment_id) + " is not in the index."
        assert self.all_experiments[experiment_id].experiment_id == experiment_id, \
            f"Stored experiment ID does not match index. The index was {experiment_id} and the stored ID was " \
            f"{self.all_experiments[experiment_id].experiment_id}"

        prev_state = self.all_experiments[experiment_id].state
        assert prev_state != "finished", "Attempt to complete time of an already finished experiment id" + \
                                         str(experiment_id)

        self.all_experiments[experiment_id].time = exp_time
        self.all_experiments[experiment_id].state = "waiting"
        self.all_experiments[experiment_id].start_time = start_time
        self.all_experiments[experiment_id].end_time = end_time
        #self.all_experiments[experiment_id].end_time = time.localtime()
        self.total_time = self.total_time + exp_time

        if experiment_id in self.pending_ids:
            pen_id = self.pending_ids.index(experiment_id)
            del self.pending_ids[pen_id]

        instance_id = self.all_experiments[experiment_id].instance_id
        if experiment_id in self.pending_by_instance[instance_id]:
            pen_id = self.pending_by_instance[instance_id].index(experiment_id)
            del self.pending_by_instance[instance_id][pen_id]

        self.n_waiting += 1
        self.waiting_ids.append(experiment_id)
        return self.all_experiments[experiment_id]

    def complete_experiment_quality(self, experiment_id, solution_quality, _from_parent=False):
        """
        Sets an experiment as finished and adds the solution quality to it.
        The experiment time and end timestamp should be previously added with
        self.set_experiment_time

        :param experiment_id: Experiment ID
        :param solution_quality: Result reported by the execution

        :return: ExperimentEntry of the finished experiment
        """

        assert experiment_id in self.all_experiments, "Experiment " + str(experiment_id) + " is not in the index."
        assert self.all_experiments[experiment_id].experiment_id == experiment_id, \
            f"Stored experiment ID does not match index. The index was {experiment_id} and the stored ID was " \
            f"{self.all_experiments[experiment_id].experiment_id}"

        prev_state = self.all_experiments[experiment_id].state
        assert prev_state == "waiting", "Attempt to complete a non waiting experiment id" + str(experiment_id)

        self.all_experiments[experiment_id].quality = solution_quality
        self.all_experiments[experiment_id].state = "finished"

        self._log_experiment(self.finished_log_file, self.all_experiments[experiment_id])

        self.n_waiting -= 1
        wait_id = self.waiting_ids.index(experiment_id)
        del self.waiting_ids[wait_id]

        self.n_complete += 1
        id_config = self.all_experiments[experiment_id].configuration_id
        if id_config not in self.n_complete_by_config.keys():
            self.n_complete_by_config[id_config] = 0
        self.n_complete_by_config[id_config] += 1

        instance_id = self.all_experiments[experiment_id].instance_id
        if instance_id not in self.n_complete_by_instance.keys():
            self.n_complete_by_instance[instance_id] = 0
        self.n_complete_by_instance[instance_id] += 1

        # sync only when parent call
        if (not _from_parent) and hasattr(self, "exps_view") and self.exps_view:
            self.exps_view._complete_experiment_from_parent(self.all_experiments[experiment_id])

        return self.all_experiments[experiment_id]

    def discard_experiment(self, experiment_id):
        """
        Sets an experiment as discarded. It adds the ExperimentEntry.end_time time stamp
        to log the time it was discarded

        :param experiment_id: Experiment ID
        """
        assert experiment_id in self.all_experiments, "Experiment " + str(experiment_id) + " is not in the index."
        assert self.all_experiments[experiment_id].experiment_id == experiment_id, \
            f"Stored experiment ID does not match index. The index was {experiment_id} and the stored ID was " \
            f"{self.all_experiments[experiment_id].experiment_id}"

        prev_state = self.all_experiments[experiment_id].state

        if prev_state == "finished":
            # no need to discard an already finished experiment
            return

        self.all_experiments[experiment_id].state = "discarded"
        self.all_experiments[experiment_id].end_time = time.localtime()

        # log as discarded
        self._log_experiment(self.canceled_log_file, self.all_experiments[experiment_id])

        if prev_state == "waiting":
            wait_id = self.waiting_ids.index(experiment_id)
            del self.waiting_ids[wait_id]

        else:
            pen_id = self.pending_ids.index(experiment_id)
            del self.pending_ids[pen_id]
            instance_id = self.all_experiments[experiment_id].instance_id
            if experiment_id in self.pending_by_instance[instance_id]:
                pen_id = self.pending_by_instance[instance_id].index(experiment_id)
                del self.pending_by_instance[instance_id][pen_id]

        self.discarded_ids.append(experiment_id)
        self.n_discarded = self.n_discarded + 1

    ####################################################
    ## Functions for getting experiments and results  ##
    ####################################################

    def get_all_experiments(self):
        """
        Gets all the experiments as data frame

        :return: pandas data frame
        """
        df = pd.DataFrame([d.as_dict() for x, d in self.all_experiments.items()])
        return df

    def get_all_instances(self):
        """
        Gets all the instances in all experiments

        :return: list of instance ids
        """
        return list(self.experiments_by_instance.keys()) 

    def get_experiment_by_id(self, experiment_id, as_dataframe=True):
        """
        Get a set the experiments performed for a configuration

        :param experiment_id: Experiment ID
        :param as_dataframe: Boolean that indicates if the experiment should be returned as a data frame

        :return: The selected experiments either in a pandas data frame or ExperimentEntry
        """
        # we assume index is experiment_id-1
        # This was changed, in order to make the recovery
        experiment_found = self.all_experiments[experiment_id]

        if as_dataframe:
            df = pd.DataFrame([experiment_found.as_dict()])
            return df
        else:
            return experiment_found

    def get_experiments_by_configuration(self, configuration_id, as_dataframe=True):
        """
        Get a set the experiments performed for a configuration

        :param configuration_id: Configuration ID
        :param as_dataframe: Boolean that indicates if the results should be returned as a data frame

        :return: The selected experiments either in a data frame or in a dictionary of ExperimentEntry
                 with configuration IDs as keys
        """
        if configuration_id not in self.experiments_by_configuration.keys():
            return None
        if as_dataframe:
            exp_index = self.experiments_by_configuration[configuration_id]
            selected = [self.all_experiments[x] for x in exp_index]
            df = pd.DataFrame([x.as_dict() for x in selected])
            return df
        else:
            selected = {configuration_id: self.all_experiments[self.experiments_by_configuration[configuration_id]]}
            return selected

    def get_experiments_by_configurations(self, configuration_ids, as_dataframe=True):
        """
        Get a set the experiments performed for a set of configurations

        :param configuration_ids: Configuration IDs
        :param as_dataframe: Boolean that indicates if the experiments should be returned as a data frame

        :return: The selected experiments either in a pandas data frame or in a dictionary of ExperimentEntry
                 with configuration IDs as keys
        """
        assert isinstance(configuration_ids, list), "configurations_ids must a be a list of integers"
        assert all([x in self.experiments_by_configuration.keys() for x in configuration_ids]), \
            "Cant find configuration in experiments"

        if as_dataframe:
            index = list(itertools.chain.from_iterable([self.experiments_by_configuration[x] for x in configuration_ids]))
            selected = [self.all_experiments[x] for x in index]
            df = pd.DataFrame([x.as_dict() for x in selected])
            return df
        else:
            selected = {x: [self.all_experiments[y] for y in self.experiments_by_configuration[x]] for x in configuration_ids}
            return selected
    
    def get_result_by_configuration(self, configuration_id: int, drop_na_instances=False):
        """
        Get a set the experiments results for a configuration

        :param configuration_id: Configuration ID
        :param drop_na_instances: Boolean that indicates if pending experiments results should be excluded.
                                  If pending experiments are included the results will included None values.

        :return: Dictionary of the selected experiments results with instance IDs as keys
        """
        assert isinstance(configuration_id, int), "configuration_id must be an integer"
        assert configuration_id in self.experiments_by_configuration.keys(), \
            "Requesting results of an unknown configuration " + str(configuration_id)

        indexes = self.experiments_by_configuration[configuration_id]
        if drop_na_instances:
            results = {}
            for x in indexes:
                quality = self.all_experiments[x].quality
                if (quality not in [None, 'None']
                    and not np.isnan(quality) 
                    and self.all_experiments[x].state not in ("pending", "discarded")):
                    results[self.all_experiments[x].instance_id] = quality

        else:
            results = {self.all_experiments[x].instance_id: self.all_experiments[x].quality
                       for x in indexes if self.all_experiments[x].state != "discarded"}

        return results

    def get_results_by_configurations(self, configuration_ids, as_dataframe=True, drop_na_instances=True):
        """
        Get a set the experiments results for a configuration

        :param configuration_ids: List of configuration IDs
        :param as_dataframe: Boolean that indicates if the results should be returned as a data frame
        :param drop_na_instances: Boolean that indicates if pending experiments results should be excluded.
                                  If pending experiments are included the results will included None values.

        :return: Dictionary of the selected experiments results with instance IDs as keys
        """
        assert isinstance(configuration_ids, list), "configurations_ids must a be a list of integers"
        assert all([x in self.experiments_by_configuration.keys() for x in configuration_ids]), \
            "Cant find configuration {} in experiments".format(configuration_ids)

        if as_dataframe:
            data = {x: self.get_result_by_configuration(x) for x in configuration_ids}
            df = pd.DataFrame(data)
            if drop_na_instances:
                return df.dropna()
            else:
                return df

        else:
            results_dict = {x: self.get_result_by_configuration(x, drop_na_instances=drop_na_instances) for x in configuration_ids}
            return results_dict

    def get_result_list_by_configuration(self, id_conf):
        """
        Gets a list of results (quality) of a configuration

        :param id_conf: Configuration ID

        :return: List of results
        """
        if id_conf not in self.experiments_by_configuration.keys():
            return None
        ids_exp = self.experiments_by_configuration[id_conf]
        results = [self.all_experiments[x].quality for x in ids_exp if self.all_experiments[x].quality is not None]
        return results

    def get_pending_experiments_by_conf_ids(self, configuration_ids):
        """
        Get ExperimentEntry objects which are pending for a set of configurations

        :param configuration_ids: Configuration IDs

        :return: List of ExperimentEntry objects
        """
        selected = []
        for exp_id in self.pending_ids:
            experiment = self.all_experiments[exp_id]
            if isinstance(configuration_ids, int):
                if experiment.configuration_id == configuration_ids:
                    selected.append(experiment)
            elif experiment.configuration_id in configuration_ids:
                selected.append(experiment)
        return selected

    def get_waiting_experiments_by_conf_ids(self, configuration_ids):
        """
        Get ExperimentEntry objects which are waiting for evalution for a set of configurations

        :param configuration_ids: Configuration IDs

        :return: List of ExperimentEntry objects
        """
        selected = []
        for exp_id in self.waiting_ids:
            experiment = self.all_experiments[exp_id]
            if isinstance(configuration_ids, int):
                if experiment.configuration_id == configuration_ids:
                    selected.append(experiment)
            elif experiment.configuration_id in configuration_ids:
                selected.append(experiment)
        return selected

    def get_testing_experiments(self):
        """
        Get ExperimentEntry objects which are waiting to be test

        :return: List of ExperimentEntry objects
        """
        selected = []
        for exp_id in self.testing_ids:
            experiment = self.all_experiments[exp_id]
            selected.append(experiment)
        return selected

    def get_n_testing_experiments(self):
        """
        Get number of experiments which are waiting to be test

        : return: Int number
        """
        return self.n_testing

    def get_ncompleted_by_configuration(self, id_config):
        """
        Get number of finished experiments by configuration

        :param id_config: Configuration ID

        :return: Number of finished experiments
        """
        return self.n_complete_by_config[id_config]

    def get_ncompleted_by_configurations(self, ids_config) -> Dict[int, int]:
        """
        Get the number of finished experiments of a set of configurations

        :param ids_config: Configuration IDs

        :return: Dictionary of number of experiments with configuration IDs as key
        """
        assert isinstance(ids_config, list), "ids_config must a be a list of integers"
        n_complete = {x: self.n_complete_by_config[x] for x in ids_config if x in self.n_complete_by_config}
        return n_complete

    def get_nincompleted_by_configurations(self, ids_config):
        """
        Get the number of finished experiments of a set of configurations

        :param ids_config: Configuration IDs

        :return: Dictionary of number of experiments with configuration IDs as key
        """
        assert isinstance(ids_config, list), "ids_config must a be a list of integers"
        n_complete = {x: len(self.n_experiments_instance) - self.n_complete_by_config[x] for x in ids_config}
        return n_complete

    def get_max_ncompleted_by_configurations(self, ids_config):
        """
        Get the number of experiments of the configuration (of a set) that was executed the most.
        This can be seen as the maximum number of instances executed.

        :param ids_config: Configuration IDs

        :return: Number of experiments of the most executed configuration in the set
        """
        assert isinstance(ids_config, list), "ids_config must a be a list of integers"
        n_complete = [self.n_complete_by_config[x] for x in ids_config]
        return max(n_complete)

    def get_n_instances(self):
        """
        Get number of different instances in the experiments
        """
        return len(self.n_experiments_instance)

    def get_experiments_by_instance(self, instance_id, as_dataframe=True):
        """
         Get a set the experiments performed for an instance

         :param instance_id: Instance ID
         :param as_dataframe: Boolean that indicates if the results should be returned as a data frame

         :return: The selected experiments either in a data frame or in a dictionary of ExperimentEntry
                  with instance IDs as keys in experiments_by_instance
         """
        if instance_id not in self.experiments_by_instance.keys():
            return None

        if as_dataframe:
            exp_index = self.experiments_by_instance[instance_id]
            selected = [self.all_experiments[x] for x in exp_index]
            df = pd.DataFrame([x.as_dict() for x in selected])
            return df
        else:
            selected = [{instance_id: self.all_experiments[x]} for x in self.experiments_by_instance[instance_id]]
            return selected

    def get_ncomplete_by_instance(self, instance_id):
        if instance_id not in self.n_complete_by_instance.keys():
            return 0
        return self.n_complete_by_instance[instance_id]

    def get_pending_experiments_by_instance_id(self, instance_id):
        """
        Get ExperimentEntry objects which are pending for an instance

        :param instance_id: Instance id

        :return: List of Experiment Entry objects
        """
        selected = []
        for exp_id in self.pending_ids:
            experiment = self.all_experiments[exp_id]
            if experiment.instance_id == instance_id:
                selected.append(experiment)
        return selected

    def get_waiting_experiments_by_instance_id(self, instance_id):
        """
        Get ExperimentEntry objects which are waiting for evaluation for an instance

        :param instance_id: Instance id

        :return: List of Experiment Entry objects
        """
        selected = []
        for exp_id in self.waiting_ids:
            experiment = self.all_experiments[exp_id]
            if experiment.instance_id == instance_id:
                selected.append(experiment)
        return selected

    def get_nexperiments_by_instance(self, instance_id):
        """
        Get number of experiments (all states) for an instance
        
        :param instance_id: Instance ID

        :return: number of experiments
        """
        if instance_id not in self.n_experiments_instance.keys():
            return 0
        return self.n_experiments_instance[instance_id]

    def print_pending(self):
        """
        Print all pending experiments
        """
        for exp_id in self.pending_ids:
            experiment = self.all_experiments[exp_id]
            print(experiment.configuration_id, ":", experiment.instance_id)

    ####################################################
    ##        Functions for log and recovering        ##
    ####################################################

    def _log_experiment(self, log_file, experiment: ExperimentEntry):
        """
        Adds a line to the experiment log file indicated

        :param log_file: File path to write the log
        :param experiment: ExperimentEntry object to write in the log
        """
        # check if csv exist
        if not os.path.isfile(log_file):
            raise FileError("Log file " + log_file + " was not found")

        # create a row from experiment
        exp_row = {"experiment_id": experiment.experiment_id,
                   "configuration_id": experiment.configuration_id,
                   "instance_id": experiment.instance_id,
                   "budget": experiment.budget,
                   "bound_max": experiment.bound_max,
                   "cmd_line": experiment.cmd_line,
                   "quality": experiment.quality,
                   "time": experiment.time,
                   "state": experiment.state,
                   "creation_time": list(experiment.creation_time),
                   "start_time": experiment.start_time,
                   "end_time": experiment.end_time if not isinstance(
                       experiment.end_time, time.struct_time) else list(experiment.end_time)}

        # write log of the experiment
        with open(log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=exp_row.keys())
            if os.path.getsize(log_file) == 0:
                writer.writeheader()
            writer.writerow(exp_row)

    def _read_log_data(self, log_file):
        """
        Read and returns experiment data en in a log File

        :param log_file: File path of the log file

        :return: Dictionary of ExperimentEntry objects, experiment_id are the keys.
                 Returns None if the file does not exist or cannot be opened
        """
        dtype_mapping = {
            "experiment_id": int,
            "configuration_id": int,
            "instance_id": int,
            "budget": float,
            "bound_max": float,
            "cmd_line": str,
            "quality": float,
            "time": float,
            "state": str,
            "creation_time": str,
            "start_time": float,
            "end_time": str
        }

        # if the file exists it gets the rows
        all_data = {}
        if os.path.exists(log_file):
            with open(log_file, newline='') as f:
                try:
                    # load the header from the provided file
                    # NOTE: rewind the pointer
                    header = pd.read_csv(f, nrows=0).columns.values.tolist()
                    f.seek(0)

                    if header[0] != 'experiment_id':
                        # the provided file is not in csv format
                        # goto except
                        raise FileError('Provided file is not in csv format.')

                    # load experiments from csv file
                    # load log file and fill nan with None
                    # NOTE: csv format of configurations
                    #   experiment_id,configuration_id,instance_id,budget,bound_max,cmd_line,quality,time,state,creation_time,start_time,end_time
                    for reader in pd.read_csv(f, dtype=dtype_mapping,
                                             na_values=["", "NULL", "none", "NA", "missing", None],
                                             chunksize=10000):
                        for exp in reader.itertuples(index=False):
                            experiment = self.from_csv(exp)
                            all_data[experiment.experiment_id] = experiment
                    return all_data

                except FileError:
                    _json_loads = json.loads
                    # load experiments from the dict file
                    for line in f:
                        d = _json_loads(line)
                        all_data[d["experiment_id"]] = self.from_dict(d)
                    return all_data

                except pd.errors.EmptyDataError:
                    # empty file
                    return all_data

        else:
            # raise FileError(f"Attempt to recover from a non-existent file: {log_file}")
            print(f"# Note: attempt to recover experiments "
                  f"from a non-existent file: {log_file.split('/')[-1]}.")

        return None

    def load_from_log(self, log_folder):
        """
        Loads experiment data from experiments log files

        :param log_folder: Log folder where are the files:
            - exps_sub.log: submitted experiments
            - exps_fin.log: finished experiments
            - exps_dis.log: cancelled experiments
        """
        if self.onlytest: return

        submitted_log_file = log_folder + "/exps_sub.log"
        finished_log_file = log_folder + "/exps_fin.log"
        canceled_log_file = log_folder + "/exps_disc.log"

        s_exp = self._read_log_data(submitted_log_file)
        f_exp = self._read_log_data(finished_log_file)
        c_exp = self._read_log_data(canceled_log_file)

        if s_exp is not None:
            for i in s_exp.keys():
                if (f_exp is not None) and (i in f_exp.keys()):
                    exp = f_exp[i]
                elif (c_exp is not None) and (i in c_exp.keys()):
                    exp = c_exp[i]
                else:
                    exp = s_exp[i]

                self._add_experiment(exp)
        else:
            for exp in f_exp.values(): self._add_experiment(exp)
            if c_exp is not None:
                for exp in c_exp.values(): self._add_experiment(exp)

    #####################################################
    ## Functions for calculating bounds (capping only) ##
    #####################################################
    def update_priori_bounds(self, instance_ids: list, bound_max=None, domType="adaptive-dom", new_ins: list=[], 
                             view_entry:bool=True, readlogs:bool=False):
        """
        This method is called by calculate_priori_bounds to update the priori bounds of the elite configurations
        It includes two methods: capping and adaptive capping

        :param instance_ids: a list of instances used by the new experiments
        :param config_ids: a list of configurations from the new experiments
        """
        if view_entry:
            src = self._resolve_source()
        else:
            src = self

        self.priori_bounds = {}
        if len(new_ins) > 0: assert instance_ids[-len(new_ins):] == new_ins, \
            "new sampled instance is not at the end of instance stream"

        ins_to_update = [x for x in instance_ids if x not in new_ins]

        for selected_elite in self.old_elites:

            c_bounds = self.priori_bounds[selected_elite] = {x: bound_max for x in [0] + instance_ids}

            if len(ins_to_update) == 0:
                return

            bounds = {x: bound_max for x in instance_ids}
            selected_ins = []

            elite_exps = src.get_experiments_by_configuration(selected_elite)

            finished_exps = elite_exps.loc[elite_exps["state"] == "finished"]
            exps = finished_exps.loc[finished_exps["instance_id"].isin(ins_to_update)]
            total_sum = sum(exps["time"])

            for x in ins_to_update:
                selected_ins.append(x)
                selected_exps = finished_exps.loc[finished_exps["instance_id"].isin(selected_ins)]
                c_exp = selected_exps.loc[selected_exps["instance_id"] == x]

                if (len(selected_exps) > 0
                    and len(c_exp) > 0
                    and not c_exp["time"].isna().any()):
                    if domType in ("adaptive-dom", "adaptive-dominance", "adaptive-Dominance"):
                        ref_sum = min(sum(selected_exps["time"]) + DELTA, bounds[x])
                        bounds[x] = ref_sum
                    else:
                        bounds[x] = total_sum

                if not readlogs: assert bounds[x] > 0, "Error: Calculated negative priori bound!!"
                else: assert bounds[x] >= 0, "Error: Calculated negative priori bound!!"
                bounds[x] = max(round(bounds[x], self.budget_digits), 1.0/pow(10, self.budget_digits))

            for k, v in bounds.items():
                c_bounds[k] = v

    # FIXME TEST_improvement: priori
    def calculate_priori_bounds(self, instance_ids: list, elitist_ids: list, bound_max=None,
                                domType="adaptive-dom", recovery: bool=False, readlogs: bool=False, new_ins: list=[], view_entry:bool=True):
        """
        This method is called by race to calculate the priori bound without the information of
        finished experiments for the new exeriments
        It includes two methods: capping and adaptive capping

        :param instance_ids: a list of instances used by the new experiments
        :param config_ids: a list of configurations from the new experiments
        """
        if len(new_ins) > 0: self.get_stream_experiments(instance_ids)

        if view_entry:
            src = self._resolve_source()
        else:
            src = self

        # check if elites are updated (elite configurations and the selected instances)
        update_flag = False
        if not self.old_elites:
            update_flag = True
            for x in elitist_ids: self.old_elites[x] = src.get_ncompleted_by_configuration(x)

        else:
            if Counter(elitist_ids) != Counter(self.old_elites.keys()):
                update_flag = True
                self.old_elites = {}
                for x in elitist_ids: self.old_elites[x] = src.get_ncompleted_by_configuration(x)

            else:
                for x in elitist_ids:
                    if src.get_ncompleted_by_configuration(x) != self.old_elites[x]:
                        update_flag = True
                        self.old_elites[x] = src.get_ncompleted_by_configuration(x)

        if update_flag or recovery or len(new_ins) > 0:
            self.update_priori_bounds(
                instance_ids=instance_ids, bound_max=bound_max, domType=domType, new_ins=new_ins, readlogs=readlogs)

        selected_elite = 1
        if len(elitist_ids) > 0:
            # selected_elite = random.choice(elitist_ids)
            selected_elite = elitist_ids[0]

        return self.priori_bounds[selected_elite]

    def calculate_post_bounds(self, exp: ExperimentEntry, view_entry: bool=True):
        """
        This method is called by the executioner to update the bound (post bound) when submit experiment
        from waiting queue to the free worker using the information from finished experiments
        This method is only valid for capping problems

        :param experiment: the next experiment
        """
        if view_entry:
            src = self._resolve_source()
        else:
            src = self

        finished_ins = list(src.get_result_by_configuration(exp.configuration_id, drop_na_instances=True).keys())
        exps = src.get_experiments_by_configuration(exp.configuration_id)
        exps = exps.loc[exps["instance_id"].isin(finished_ins)]
        exps = exps.loc[exps["state"] == "finished"]
        exps = exps.loc[exps["budget"] != exps["bound_max"]]

        # update bounds
        # exp.budget: priori bound of current experiment
        #   capping: total sum of all finished executions (elite) or bound_max
        #   adaptive capping: sum of finished executions on selected ins (elite) or bound_max
        if len(exps) > 0:
            used_bound = sum(exps["time"])
            new_bound = exp.budget - used_bound + DELTA
            full_finished_ins = list(self.get_result_by_configuration(exp.configuration_id, drop_na_instances=True).keys())
            full_all = self.get_experiments_by_configuration(exp.configuration_id)
            full_exps = full_all.loc[full_all["instance_id"].isin(full_finished_ins)]

            assert new_bound > 0, (
                f"Error: Calculated negative post bound!! {exp.experiment_id}, {exp.configuration_id}, {exp.instance_id}, {src.get_all_instances()}\n"
                f"  View:\n"
                f"      finished_ins = {finished_ins}, exps: \n{exps}\n"
                f"  Full:\n"
                f"      finished_ins = {full_finished_ins}, exps: \n{full_exps}\n"
            )

            new_bound_format = '{0:.{prec}f}'.format(new_bound, prec=self.budget_digits)
            experiment = ExperimentEntry(experiment_id=exp.experiment_id,
                                        configuration_id=exp.configuration_id,
                                        instance_id=exp.instance_id,
                                        budget=new_bound,
                                        budget_format=new_bound_format,
                                        bound_max=exp.bound_max,
                                        bound_max_format=exp.bound_max_format,
                                        cmd_line=exp.cmd_line,
                                        param_line=exp.param_line,
                                        quality=exp.quality,
                                        exp_time=exp.time,
                                        state=exp.state)
            return experiment
        return exp

    def get_scheduled_experiments(self):
        """
        Get ExperimentEntry objects which are pending for a set of configurations

        :param configuration_ids: Configuration IDs

        :return: List of ExperimentEntry objects
        """
        configs = []
        exps = []
        for exp_id, exp in self.all_experiments.items():
            if exp.state in ("pending", "waiting"):
                configs.append(exp.configuration_id)
                exps.append(exp_id)
        return configs, exps

    def get_scheduled_experiments_by_instance_id(self, instance_id):
        """
        Get ExperimentEntry objects which are pending for an instance

        :param instance_id: Instance id

        :return: List of Experiment Entry objects
        """
        selected = []
        for exp_id in self.pending_ids + self.waiting_ids:
            experiment = self.all_experiments[exp_id]
            if experiment.instance_id == instance_id:
                selected.append(experiment)
        return selected

    def update_status_for_recovering(self, elites=None, alive=None):
        """
        update parameter values for recovering
        """
        if alive is not None:
            for x in alive:
                if x not in self.n_complete_by_config.keys():
                    self.n_complete_by_config[x] = 0
        if self.capping and elites is not None:
            for x in elites: self.old_elites[x] = self.get_ncompleted_by_configuration(x)

    def get_stream_experiments(self, stream: list=None):
        if set(self.get_all_instances()) == set(stream):
            return self

        if self.exps_view is None:
            self.exps_view = ExperimentsView(parent=self, stream=stream)

        elif self.exps_view.stream != set(stream):
            self.exps_view.update_view(stream)

        return self.exps_view

    def _resolve_source(self):
        """
        obtain active experiments view

        :return: current available view
        """
        ref = getattr(self, "exps_view", None)
        if ref is not None:
            return ref
        return self

class ExperimentsView(Experiments):
    """
    A filtered view of Experiments restricted to a subset of instance IDs.
    All parent methods work unchanged since the same attribute names are used.
    """
    def __init__(self, parent: Experiments, stream: list):
        self.exps_view = None

        self.parent = parent
        self.stream = set(stream or [])

        self._reset_counters()
        self._rebuild_from_parent()

    def _reset_counters(self):
        # subset experiment ids
        self.all_experiments = {}

        # necessary counters
        self.experiments_by_configuration = defaultdict(list)
        self.experiments_by_instance = defaultdict(list)

        self.n_experiments = 0
        self.n_experiments_instance = Counter()

        self.n_complete = 0
        self.n_complete_by_config = Counter()
        self.n_complete_by_instance = Counter()

    def update_view(self, new_stream: list):
        new_stream = set(new_stream or [])

        if new_stream == self.stream:
            return

        self.stream = new_stream
        self._rebuild_from_parent()

    def _rebuild_from_parent(self):
        self._reset_counters()

        for _, e in self.parent.all_experiments.items():
            if getattr(e, "instance_id", None) not in self.stream:
                continue

            self._add_experiment(e)

    def _add_experiment_from_parent(self, experiment):

        if experiment.instance_id not in self.stream:
            return

        self._add_experiment(experiment)

    def _add_experiment(self, experiment):
        eid = experiment.experiment_id
        cfg = experiment.configuration_id
        ins = experiment.instance_id
        st  = experiment.state

        # ---- minimal state ----
        self.all_experiments[eid] = experiment
        self.n_experiments += 1

        self.experiments_by_configuration[cfg].append(eid)
        self.experiments_by_instance[ins].append(eid)
        self.n_experiments_instance[ins] += 1

        if st == "finished":
            self.n_complete += 1
            self.n_complete_by_config[cfg] += 1
            self.n_complete_by_instance[ins] += 1

    def _complete_experiment_from_parent(self, experiment):
        if experiment.instance_id not in self.stream:
            return
        self.complete_experiment(experiment)

    def complete_experiment(self, experiment):
        cfg = experiment.configuration_id
        ins = experiment.instance_id

        self.n_complete += 1
        self.n_complete_by_config[cfg] += 1
        self.n_complete_by_instance[ins] += 1
