import os
import enum
import logging
import collections
import pandas as pd

from cmath import isnan
from datetime import datetime
from tzlocal import get_localzone

from crace.containers.experiments import ExperimentEntry
from crace.containers.configurations import ConfigurationEntry


TaskLog = collections.namedtuple(
    "TaskLog",
    [
        "n_task",
        "action",
        "instances",
        "total_instances",
        "total_configurations",
        "best_id",
        "best_configuration",
        "tbest_id",
        "mean_best",
        "ranks",
        "t_alive_ids",
        "alive_ids",
        "discarded_ids",
        "elites",
        "elite_list",
        "exp_so_far",
        "pending_exp",
        "pending_config",
        "t_time",
        "w_time",
        "used_budget",
        "remaining_budget",
        "nb_new_configs",
        "n_pre_budget",
        "idx_updating",
        "idx_updating_restart",
        "n_all_updating",
    ],
)

CTypeLog = collections.namedtuple(
    "CTypeLog",
    [
        "type",
    ],
)


class TaskType(enum.Enum):
    """
    Class that defines the type of task to be reported in the stdout
    """
    start = 0
    test = 1
    completion = 2
    configuration = 3
    instance_challenge = 4
    instance_expect = 5
    instance_stream = 6
    shuffle = 7
    elite_instance = 8
    model_update = 9
    model_restart = 10
    end = 11


class SamplePolicy(enum.IntFlag):
    configuration = enum.auto()
    instance_challenge = enum.auto()
    instance_expect = enum.auto()
    elites = enum.auto()
    none = enum.auto()

    def __contains__(self, item):
        return (self.value & item.value) == item.value


class RaceType(enum.Enum):
    quality = 1
    runtime = 2
    capping = 3


class RaceLogger:
    """
    The RaceLogger class manages the stdout log of the race

    :ivar tasks: list of tasks of tasks logged so far
    :ivar n_tasks: number of tasks logged so far
    :ivar logging_variables:
    :ivar printed_lines:
    :ivar race_type: defines the type of the log that should be produced (checl RaceType class)
    """

    def __init__(self, options, logging_variables, budget, n_pre_budget, n_configs, n_updating):
        self.ctype = None
        self.n_task = 0
        self.logging_variables = logging_variables

        # counter for lines of one block
        self.printed_lines = 0

        # counter for each slice
        self.n_sampled_ins = 0
        self.n_discarded = 0

        # counter for the racing produre
        self.sum_discarded = 0
        self.nb_restart = 0

        # old elite configuration ids
        self.old_elites = []

        # necessary options
        self.debug_level = options.debugLevel.value
        self.log_level = options.logLevel.value

        self.current_task = TaskLog(
                                    n_task=0,
                                    action=TaskType.start,
                                    instances=None,
                                    total_instances=0,
                                    total_configurations=0,
                                    best_id=None,
                                    best_configuration=None,
                                    tbest_id=None,
                                    mean_best=None,
                                    ranks=None,
                                    t_alive_ids=None,
                                    alive_ids=None,
                                    discarded_ids=None,
                                    elites=None,
                                    elite_list=None,
                                    exp_so_far=0,
                                    pending_exp=None,
                                    pending_config=None,
                                    t_time=0,
                                    w_time=0,
                                    used_budget=0,
                                    remaining_budget=budget,
                                    nb_new_configs=n_configs,
                                    n_pre_budget=n_pre_budget,
                                    idx_updating=0,
                                    idx_updating_restart=0,
                                    n_all_updating=n_updating,
        )

    def log_ctype(self, type):
        self.ctype = CTypeLog(type=type)

    def log_task(
        self,
        action,
        instances,
        total_instances,
        total_configurations,
        best_id,
        best_configuration,
        tbest_id,
        mean_best,
        ranks,
        t_alive_ids,
        alive_ids,
        discarded_ids,
        exp_so_far,
        pending_exp,
        pending_config,
        t_time,
        w_time,
        n_all_updating,
        elites=[],
        elite_list={},
        used_budget=0,
        remaining_budget=0,
        nb_new_configs=0,
        n_pre_budget=0,
        idx_updating=0,
        idx_updating_restart=0,
    ):
        self.current_task = TaskLog(
                                    n_task=self.n_task + 1,
                                    action=action,
                                    instances=instances,
                                    total_instances=total_instances,
                                    total_configurations=total_configurations,
                                    best_id=best_id,
                                    best_configuration=best_configuration,
                                    tbest_id=tbest_id,
                                    mean_best=mean_best,
                                    ranks=ranks,
                                    t_alive_ids=t_alive_ids,
                                    alive_ids=alive_ids,
                                    discarded_ids=discarded_ids,
                                    elites=elites,
                                    elite_list=elite_list,
                                    exp_so_far=exp_so_far,
                                    pending_exp=pending_exp,
                                    pending_config=pending_config,
                                    t_time=t_time,
                                    w_time=w_time,
                                    used_budget=used_budget,
                                    remaining_budget=remaining_budget,
                                    nb_new_configs=nb_new_configs,
                                    n_pre_budget=n_pre_budget,
                                    idx_updating=idx_updating,
                                    idx_updating_restart=idx_updating_restart,
                                    n_all_updating=n_all_updating,
                                )
        self.n_task += 1
        self.print_current_task()


    def _print_table_header(self):
        if self.ctype.type is RaceType.capping:
            print(
                "+-+------------+-----------+---------------+---------------+------------+---------+---------+---------+---------+---------"
            )
            print(
                "| |  Instances |      Alive|           Best|      Mean best| Exp so far |  W time |  T time | E queue | C queue | Elites"
            )
            print(
                "+-+------------+-----------+---------------+---------------+------------+---------+---------+---------+---------+---------"
            )

        else:
            print(
                "+-+------------+-----------+---------------+---------------+------------+---------+---------+---------+---------"
            )
            print(
                "| |  Instances |      Alive|           Best|      Mean best| Exp so far |  W time | E queue | C queue | Elites"
            )
            print(
                "+-+------------+-----------+---------------+---------------+------------+---------+---------+---------+---------"
            )


    def print_race_header(self):
        c_task = self.current_task
        print("\n\n# {}: Slice {} of {}".format(datetime.now(get_localzone()).strftime('%Y-%m-%d %H:%M:%S %Z'),
                                                c_task.idx_updating+1,
                                                c_task.n_all_updating+1))
        print("# expectSliceBugdet: %d" % (c_task.n_pre_budget))

        if c_task.used_budget == 0:
            print("# nbInitialConfigurations: ", c_task.nb_new_configs)

        print("# Markers:")
        print("    x No test is performed.")
        print("    - The test is performed and some configurations are discarded")
        # print('    ! The test is performed and configurations could be discarded but elite configurations are preserved')
        print("    . All alive configurations are elite and nothing is discarded")
        print("    c Configurations added to the race by sampling")
        print("    i Instances added to the race based on challenge from elimination test")
        print("    e Instances added to the race based on option expecetInstance")
        print("    p Instances stream updated based on option instancePatience")

        if c_task.used_budget == 0:
            print("# Each task:")
            print("    Instances: the number of selected instances for current task (sampled instances)")
            print("    Alive: alive configurations in test (in crace)")
            print("    Best: best configuration in test (in crace)")
            # print("    Elites: (the number of used instances) configurations having the most instances")

        self._print_table_header()


    def print_current_task(self):
        """
        print current task
        the format is related to the TaskType

        all task information are seperated based on model updating
        """
        c_task = self.current_task

        # check if update the models
        # if the model is updated: end for summary (and maybe a new header)
        if c_task.action in [TaskType.model_update, TaskType.model_restart]:
            self.sum_discarded += self.n_discarded
            restart = False
            if c_task.action is TaskType.model_restart:
                restart = True
                self.nb_restart += 1

            self.print_race_footer(
                best_id=c_task.best_id,
                mean_best=c_task.mean_best,
                configuration=c_task.best_configuration,
                elite_ids=c_task.elites,
                elite_list=c_task.elite_list,
                end=False, restart=restart,
                terminated=False)

            # reset counters for each slice
            self.n_sampled_ins = 0
            self.n_discarded = 0
            self.printed_lines = 0

            # if there is budget remaining: new header
            if c_task.remaining_budget > 0:
                self.print_race_header()
                self.old_elites = []

        # print task based on the return value of _able_to_print()
        if self._able_to_print():
            if self.printed_lines >= 15:
                self._print_table_header()
                self.printed_lines = 0
            self._print_task()
            self.printed_lines = self.printed_lines + 1
            self.header = False


    def _able_to_print(self):
        """
        check if it's necessary to print the current task
        """
        c_task = self.current_task

        # FIXME: improve the rules for printing

        if c_task.action is TaskType.test:
            if len(c_task.discarded_ids) > 0:
                self.n_discarded += len(c_task.discarded_ids)
                if self.old_elites != c_task.elites: return True
                elif self.debug_level >= 1: return True
                elif self.printed_lines == 0: return True

        elif c_task.action is TaskType.completion:
            if self.debug_level >= 2:
                return True

        elif c_task.action is TaskType.configuration:
            if self.debug_level >= 2:
                return True

        elif c_task.action is TaskType.instance_challenge:
            self.n_sampled_ins += 1
            if self.old_elites != c_task.elites: return True
            elif self.debug_level >= 1: return True

        elif c_task.action is TaskType.instance_expect:
            self.n_sampled_ins += 1
            if self.old_elites != c_task.elites: return True
            elif self.debug_level >= 1: return True

        elif c_task.action is TaskType.instance_stream:
            self.n_sampled_ins += 1
            if self.debug_level >= 1: return True

        elif c_task.action is TaskType.shuffle:
            return True

        elif c_task.action is TaskType.elite_instance:
            return True

        return False

    def _print_task(self):
        """
        Prints current task
        """
        c_task = self.current_task

        # Print task type / marker
        if c_task.action is TaskType.test:
            if len(c_task.discarded_ids) > 0:
                print("|-|", end="")
            else:
                print("|.|", end="")
        elif c_task.action is TaskType.completion:
            print("|x|", end="")
        elif c_task.action is TaskType.configuration:
            print("|c|", end="")
        elif c_task.action is TaskType.instance_challenge:
            print("|i|", end="")
        elif c_task.action is TaskType.instance_expect:
            print("|e|", end="")
        elif c_task.action is TaskType.instance_stream:
            print("|p|", end="")
        elif c_task.action is TaskType.shuffle:
            print("|s|", end="")
        # elif c_task.action is TaskType.elite_instance:
        #     print("|e|", end="")

        # Print number of instances
        if c_task.action in [TaskType.test, TaskType.instance_stream]:
            print(
                "{0:>12}".format(
                    "{}({})".format(len(c_task.instances), c_task.total_instances)
                ),
                end="",
            )
        elif c_task.action in [TaskType.instance_challenge, TaskType.instance_expect, TaskType.elite_instance, TaskType.completion]:
            print("{0:>12}".format("({})".format(c_task.total_instances)), end="")
        else:
            print("            ", end="")
        print("|", end="")

        # print number of alive configurations
        if c_task.action is TaskType.test:
            print(
                "{0:>11}".format("{}({})".format(len(c_task.t_alive_ids) if
                                                 len(c_task.t_alive_ids)>0 else "-", len(c_task.alive_ids))),
                end="",
            )
        elif c_task.action in [TaskType.instance_challenge, TaskType.instance_expect,
                               TaskType.shuffle, TaskType.elite_instance]:
            print("           ", end="")
        else:
            print(
                "{0:>11}".format("{}({})".format(' ', len(c_task.alive_ids))),
                end="",
            )
        print("|", end="")

        # print current best configuration
        if c_task.action is TaskType.test:
            print(
                "{0:>15}".format("{}({})".format(c_task.tbest_id, c_task.best_id)),
                end="",
            )
        else:
            print("               ", end="")
        print("|", end="")

        # print current best mean
        if c_task.action is TaskType.test:
            if self.ctype.type is RaceType.capping:
                if (c_task.mean_best is None) or isnan(c_task.mean_best):
                    print("%15s" % "-", end="")
                else:
                    print("%15.5f" % c_task.mean_best, end="")
            else:
                if (c_task.mean_best is None) or isnan(c_task.mean_best):
                    print("%15s" % "-", end="")
                else:
                    print("%15d" % c_task.mean_best, end="")
        else:
            print("               ", end="")
        print("|", end="")

        # print number of experiments so far
        if c_task.action is TaskType.test:
            print("%12d" % c_task.exp_so_far, end="")
        else:
            print("            ", end="")
        print("|", end="")

        # print w time
        print("%9d" % c_task.w_time, end="")
        print("|", end="")

        # print t time for capping scenarios
        if self.ctype.type is RaceType.capping:
            print("%9d" % c_task.t_time, end="")
            print("|", end="")

        # print number of pending experiments
        print("%9d" % c_task.pending_exp, end="")
        print("|", end="")

        # print number of pending configurations
        print("%9d" % c_task.pending_config, end="")
        print("|", end="")

        # print elite configurations
        if c_task.elites is not None and len(c_task.elites) > 0:
            if c_task.elites != self.old_elites: self.old_elites = c_task.elites
            format_elites = ",".join([str(x) for x in c_task.elites])
            print(" {}".format(format_elites), end="")
        print("")

        # print current task to log file
        if self.log_level >= 5:
            self._print_to_log(
                self.logging_variables,
                [
                    c_task.alive_ids,
                    c_task.best_id,
                    c_task.mean_best,
                    c_task.exp_so_far,
                    c_task.w_time,
                ],
            )


    def _print_elites(self, elite_ids, elite_list):
        """
        Print the elite configurations
        :param elite_ids: list of ids of elite configurations
        :param elite_list: dict of elite configuration id and its number of used instances
        """
        n_ins = elite_list.get(elite_ids[0])
        format_elites = ",".join([str(x) for x in elite_ids])
        print(" {}{}".format(n_ins, format_elites), end="")
        print("|e|", end="")
        print(" %-84s" % format_elites, end="")
        print("|")
        self.printed_lines = self.printed_lines + 1


    def _print_to_log(self, path, variables):
        # generic function to add to csv
        if self._csv_exist(path) == True:  # check if csv exist
            pass

        configuration_df = pd.DataFrame(
            [
                {
                    "alive": variables[0],
                    "best": variables[1],
                    "mean_best": variables[2],
                    "exp_so_far": variables[3],
                    "w_time": variables[4],
                }
            ]
        )  # create a dataframe per dictionary
        configuration_df.to_csv(path, mode="a", index=False, header=False)


    def _csv_exist(self, path):
        # says if the csv exists, if not create it
        if not os.path.isfile(path):
            with open(path, "w"): pass
            return False
        return True


    def log_experiment_completion(
        self, experiment: ExperimentEntry, quality, time, debug_level
    ):
        """
        print returned experiment to log file

        :param experiment: current returned experiment
        :param quality: returned quality
        :param time: returned time
        :param debug_level: option debug_level
        """
        if self.debug_level > 0:
            logger_1 = logging.getLogger("race_log")
            logger_1.debug(
                "# Experiment {}({},{},{})=({},{}) id(conf, inst, bud)=(res,time)".format(
                    experiment.experiment_id,
                    experiment.configuration_id,
                    experiment.instance_id,
                    experiment.budget,
                    quality,
                    time,
                )
            )

    def print_race_footer(self, best_id, mean_best, configuration: ConfigurationEntry, elite_ids, elite_list,
                          end: bool=True, restart: bool=False, terminated=False):
        """
        print footer information when update or restart models
        
        :param best_id: current elitist configuration id
        :param mean_best: mean values (quality/runtime) on used instances of the best configuration
        :param configuration: the configuration object of current task
        :param elite_ids: a list of current elite configuration ids
        :param elite_list: a dict of current elite configuration id and its number of used instances
        :param end: boolean for the end of whole racing procedure
        :param restart: boolean for updating/restarting models
        """
        c_task = self.current_task

        elites = ",".join([str(x) for x in elite_ids])

        if self.ctype.type is RaceType.capping:
            print(
                "+-+------------+-----------+---------------+---------------+------------+---------+---------+---------+---------+---------"
            )
        else:
            print(
                "+-+------------+-----------+---------------+---------------+------------+---------+---------+---------+---------"
            )

        if restart:
            print("{}:  Restart models {} of {}".format(datetime.now(get_localzone()).strftime('%Y-%m-%d %H:%M:%S %Z'),
                                                    c_task.idx_updating_restart,
                                                    c_task.n_all_updating))   
        else:         
            print("{}:  Update models {} of {}".format(datetime.now(get_localzone()).strftime('%Y-%m-%d %H:%M:%S %Z'),
                                                    c_task.idx_updating,
                                                    c_task.n_all_updating))

        print("experimentsTestedSoFar:     %d" % (c_task.exp_so_far))
        print("budgetUsedSoFar:            %d" % (c_task.used_budget))
        print("remainingBudget:            %d" % (c_task.remaining_budget))
        print("nbNewConfigurations:        %d " % (c_task.nb_new_configs))
        print("nbNewInstances:             %d " % (self.n_sampled_ins))
        print("nbEliminatedConfigurations: %d " % (self.n_discarded))

        if best_id is None or mean_best is None:
            print("No elimination test is performed here!")
        else:
            print("Elite-so-far configuration: %-13s    mean value: %.5f" % (best_id, mean_best))
            print("\nDescription of the elite-so-far configuration %d:" % (best_id))
            print(configuration.cmd)

        if end: self.print_race_summary(best_id, mean_best, elites, configuration, terminated)


    def print_race_summary(self, best_id, mean_best, elites, configuration: ConfigurationEntry, terminated):
        c_task = self.current_task
        self.sum_discarded += self.n_discarded
        self.n_discarded = 0

        if terminated in ['time', 'exp', 'budget']:
            print("\n\n# {} BUDGET IS EXHAUSTED!".format(datetime.now(get_localzone()).strftime('%Y-%m-%d %H:%M:%S %Z')))
            if terminated in ['time', 'budget']:
                print("# timeUsedSoFar:            %d " % (c_task.t_time))
            if terminated in ['exp', 'budget']:
                print("# experimentsUsedSoFar:            %d " % (c_task.exp_so_far))
            if terminated == 'budget':
                print("# budgetUsedSoFar:            %d " % (c_task.used_budget))

        elif terminated == 'restart':
            print("\n\n# {} RESTART IS RESRICTED!".format(datetime.now(get_localzone()).strftime('%Y-%m-%d %H:%M:%S %Z')))
            print("# nbRestart:                  %d " % (self.nb_restart))

        print("# nbUsedInstances:            %d " % (c_task.total_instances))
        print("# nbUsedConfigurations:       %d " % (c_task.total_configurations))
        print("# nbEliminatedConfigurations (elimination test): %d " % (self.sum_discarded))
        print("# final returned elite configurations: %s " % (elites))
        print("# mean value of the final best configuration %d on %d instances: %.5f" % (best_id, c_task.total_instances, mean_best)
        )
        print("\n# Description of the final best configuration %d:" % (best_id))
        print(configuration.cmd)
        print()

    # @staticmethod
    def log_elimination_test(self, test_result, debug_level):
        if self.debug_level > 0:
            logger_1 = logging.getLogger("race_log")
            logger_1.debug(
                "# Elimination test best: {} alive: {} discarded {}".format(
                    test_result.best, test_result.alive, test_result.discarded
                )
            )

    # @staticmethod
    def log_alive_configurations(
        self, final_best, alive_ids, discarded_ids, debug_level
    ):
            logger_1 = logging.getLogger("race_log")
            logger_1.debug(
                "# After correction best: {} alive: {} discarded {}".format(
                    final_best, alive_ids, discarded_ids
                )
            )
