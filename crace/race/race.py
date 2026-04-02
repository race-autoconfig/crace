import os
import csv
import copy
import time
import math
import json
import random
import asyncio
import logging

import numpy as np
from typing import List, Tuple

from crace.errors import CraceError
from crace.containers.instances import Instances
from crace.containers.parameters import Parameters
from crace.containers.crace_options import CraceOptions
from crace.containers.crace_results import CraceResults
from crace.containers.configurations import Configurations
from crace.models.model import ModelInfo, ProbabilisticModel
from crace.containers.experiments import ExperimentEntry, Experiments
from crace.elimination_tests.eliminator import EliminationTestResults
from crace.race.race_log import RaceLogger, SamplePolicy, TaskType, RaceType
from crace.utils import get_eliminator, get_loop, setup_race_loggers, NpEncoder


logger_1 = logging.getLogger("race_log")
logger_2 = logging.getLogger("slice")


class Race:
    """
    Class that implements the racing procedure

    :ivar options: craceOptions object that contains all crace options
    :ivar log_level: Log level value to generate the log files
    :ivar debug_level: Debug level value to generate the output
    :ivar capping: Boolean that indicates if race should enable capping
    :ivar adaptive: Dominance method
    :ivar strict_pool: Boolean that indicates if race decreases the frequency of evaluation
    :ivar num_cores: Number of slave nodes
    :ivar budget: The maximal of experiments / time for the race
    :ivar expect_ins_num: Number of expected instances in the race
    :ivar target_eval: Boolean that indicates whether target evaluator should be used
    :ivar race_logger: RaceLogger object that handles the logs of the race in the stdout
    :ivar parameters: Parameters object that describes the parameter space
    :ivar instances: Instances object that describes a set of training
                     instances
    :ivar configurations: Configurations object that contains all configurations
                          generated during the race
    :ivar experiments: Experiments object that manages the experiments data
    :ivar model: Model (ProbabilisticModel class) object
    :ivar eliminator: Eliminator object
    :ivar n_alive: number of alive configurations
    :ivar first_test_done: Boolean that indicates if the first elimination test was already applied
    :ivar ranks: Dictionary of the ranks of alive elite configurations [id: nb_completed]
    :ivar best_id: Current best configuration ID
    :ivar mean_best: Mean value of current best configration
    :ivar elite_ids: Elite configurations IDs
    :ivar elite_list: Dictionary of selected elite configurations [id: nb_completed]
    :ivar old_elites: List of current elite configuration ids, old_elites[0] is the number of instance
                      used by the best configuration
    :ivar elitist_ids: List of selected elites having the most instances
    :ivar last_config_id: The id of last configuration
    :ivar n_experiments: Counter of experiments completed the target algorithm
    :ivar n_instances: Counter of instances included in the race
    :ivar n_time: Total time elapsed in the race
    :ivar n_pre_budget: Counter of predict budget for current slice
    :ivar used_budget: Counter of total used budget
    :ivar c_used_budget: Counter of used budget in current slice
    :ivar c_new_configurations: Counter of sampled configurations in current slice
    :ivar idx_updating: Index of model updating since last configuration sampling was performed
    :ivar idx_updating_restart: Index of model updating with restart since last configuration sampling was performed
    :ivar n_all_updating: Counter of model updating in the race
    :ivar _scheduled_jobs: List of experiment IDs scheduled for execution
    :ivar _scheduled_configs: List of unique configuration IDs of scheduled experiments
    :ivar _to_eval_jobs: List of experiment IDs waiting to be evaluated (when target evaluator is active)
    :ivar terminated: Boolean that indicated that the race has finished
    :ivar priori_bounds: List of predict time bound for each instance based on the elite configuration(s)
                         The index in this list is correspoding to the instance id
                         priori_bounds[0] is the default maximal timi bound
    :ivar _pool: ExecutionPool object which handles the job submission
    :ivar _result_future: Future to handle asyncronous jobs
    :ivar clock: Time state
    :ivar do_shuffle_instances: Boolean that indicates if instances should be shuffled
    :ivar returned_exps: List of complete experiments waiting to be evaluation
    :ivar n_returned: Counter of complete experiments waiting to be evaluation
    :ivar log_race: File path of the text file where the race should be logged
    :ivar log_elite: File path of the text file where elite configurations should logged

    """

    def __init__(
        self,
        initial_configurations: Configurations,
        instances: Instances,
        parameters: Parameters,
        options: CraceOptions
    ):
        """
            Creates the structures used in the race. This class should be instantiated form a
            the Crace class object.

            :param initial_configurations: Configurations object that describes a set of initial configurations
            :param instances: Instances object that describes a set of training instances
            :param parameters: Parameters object that describes the parameter space
            :param options: craceOptions object that contains all crace options
        """

        if not options.readlogs.is_set():
            setup_race_loggers(options)

        # ###################################################################################################################
        #                                                                                                                   #
        #                                          SCENARIO PARAMETERS / OPTIONS                                            #
        #                                                                                                                   #
        # ###################################################################################################################
        self.options = options
        self.log_level = self.options.logLevel.value
        self.debug_level = self.options.debugLevel.value
        self.capping = self.options.capping.value
        self.adaptive = self.options.domType.value in ("adaptive", "adaptive-dom", "adaptive-dominance", "adaptive-Dominance")
        self.strict_pool = True if self.adaptive or options.testExperimentSize.value > 1 else False

        # Minumum number of experiments to be scheduled at any time
        self.num_cores = max(1, self.options.parallel.value)
        self.budget = self.options.maxExperiments.value
        if self.options.maxTime.value > 0:
            self.budget = self.options.maxTime.value

        # compute the number of total expect instances
        self.expect_ins_num = 0
        if self.options.expectInstances.is_set() and self.options.expectInstances.value > 0:
            self.expect_ins_num = int(self.options.expectInstances.value * self.instances.fixsize) \
            if int(self.options.expectInstances.value * self.instances.fixsize) >= self.options.firstTest.value \
            else self.options.firstTest.value
            print("# Expecting instance size is {} (provided: {}) ".format(self.expect_ins_num, self.instances.fixsize))

        # Check if the race includes a target evaluator
        self.target_eval = False
        # if self.options.targetEvaluator.value is not None:
        #     self.target_eval = True


        # ###################################################################################################################
        #                                                                                                                   #
        #                                                   CRACE OBJCTS                                                    #
        #                                                                                                                   #
        # ###################################################################################################################
        self.parameters = parameters
        self.instances = instances

        # Initialize configurations
        if initial_configurations is not None:
            # Recovered configurations should be obtained when reading the scenario
            self.configurations = copy.deepcopy(initial_configurations)
        else:
            self.configurations = Configurations(
                parameters=self.parameters,
                exec_dir=self.options.execDir.value,
                recovery_folder=self.options.recoveryDir.value,
                conf_repair=self.options.repairConfiguration.value,
                debug_level=self.debug_level,
                log_base_name=os.path.basename(options.logDir.value)
            )

        # Initialize experiments
        if self.debug_level >= 3:
            print("# Initializing experiment structures")
        self.experiments = Experiments(
            log_folder=self.options.logDir.value,
            recovery_folder=self.options.recoveryDir.value,
            read_folder=self.options.readlogs.value,
            budget_digits=self.options.boundDigits.value,
            capping=self.capping,
            log_level=self.log_level,
            configurations=self.configurations,
            onlytest=self.options.onlytest.value,
        )
        self.exps_view = self.experiments   # experiments window for selected instances


        # ###################################################################################################################
        #                                                                                                                   #
        #                                                   RECOVER CRACE OR NOT                                            #
        #                                                                                                                   #
        # ###################################################################################################################

        recovery_folder=self.options.recoveryDir.value     # recover crace
        if self.options.readlogs.value is not None:
            recovery_folder=self.options.readlogs.value     # load results

        # Load data for recovering or displaying
        if recovery_folder is not None:
            # Recover model
            if self.options.recoveryDir.value is not None:
                if self.debug_level >= 1:
                    print("# Recovering sampling models")

            # Recover race
            self.model = ProbabilisticModel.from_log(
                recovery_folder=self.options.recoveryDir.value,
                read_folder=self.options.readlogs.value,
                configurations=self.configurations,
                parameters=self.parameters,
                log_folder=self.options.logDir.value,
                log_level=self.options.logLevel.value,
                global_model=self.options.globalModel.value,
                onlytest=self.options.onlytest.value,
            )

            # Get eliminator
            self.eliminator = get_eliminator(
                test_type=self.options.testType.value,
                dom_type=self.options.domType.value,
                confidence_level=self.options.confidence.value,
                first_test=self.options.firstTest.value,
                capping=self.capping,
            )

        # Initialize parameters / options
        else:
            if self.debug_level >= 2:
                print("# Initializing sampling models")

            # Initialize race model
            self.model = ProbabilisticModel(
                parameters=parameters,
                log_folder=self.options.logDir.value,
                log_level=self.options.logLevel.value,
                global_model=self.options.globalModel.value,
                )

            # In the case there is already configurations in place
            self.n_alive = len(self.configurations.get_alive_ids())

            # Sample configurations
            self.model.add_models(self.model.size)

            # FIXME: add a method for this
            to_sample = max(0, self.options.nbConfigurations.value - self.n_alive)
            if self.debug_level >= 3:
                print("# Sampling {} new configurations...".format(to_sample))
            logger_1.debug("# Sampling {} new configurations...".format(to_sample))

            # sample configurations based randomly
            if to_sample > 0:
                if initial_configurations is None:
                    configurations = self.model.sample_random_parameters(to_sample)
                    self.configurations.add_from_model(configurations)
                else:
                    configurations = self.model.sample_from_random_parents(to_sample, initial_configurations)
                    self.configurations.add_from_model(configurations)

            # Update number of alive elites
            self.n_alive = len(self.configurations.get_alive_ids())

            if self.debug_level >= 2:
                print("# Initial configurations: ", self.configurations.n_config)

            if self.debug_level >= 3:
                self.configurations.print_all()

            # Add initial models for all alive configurations
            if self.debug_level >= 3:
                print("# Adding initial sampling models")

            # FIXME: check what variables are actually used/needed!
            # Initialize eliminator
            if self.debug_level >= 3:
                print("# Initializing race eliminator")
            self.eliminator = get_eliminator(
                test_type=self.options.testType.value,
                dom_type=self.options.domType.value,
                confidence_level=self.options.confidence.value,
                first_test=self.options.firstTest.value,
                capping=self.capping,
            )


        # ###################################################################################################################
        #                                                                                                                   #
        #                                       OTHER PUBLIC PARAMETERS IN CLASS RACE                                       #
        #                                                                                                                   #
        # ###################################################################################################################

        self.first_test_done = False
        self.ranks = {}
        self.best_id = None
        self.mean_best = None
        self.elite_ids = []
        self.elite_list = {}
        self.elitist_ids = []
        self.old_elites = [self.options.firstTest.value]    # List of last elite configuration ids,
                                                            # old_elites[0] is the number of instance
                                                            # used by the best configuration
        self.last_config_id = self.configurations.get_alive_ids()[-1]

        # Counter variables
        self.RESTART_TIMES = 0      # Counter of model restart

        self.n_experiments = 0      # Counter of returned experiments
        self.n_time = 0             # Counter of runtime returned by target execution
        self.used_budget = 0        # Counter of totally used budget

        self.n_best_instances = 0   # Counter of selected instance of the best configuration FIXME: same to old_elites[0]
        self.n_elites_config = {}   # Counter of configurations that an instance can undergo without yielding any improvement

        self.n_pre_budget = 0       # Counter of predict budget for current slice
        self.c_used_budget = 0      # Counter of used budget in current slice
        self.c_new_configurations = 0   # Counter of sampled configurations in current slice

        self.idx_updating = 0           # Index/Counter of model updated totally
        self.idx_updating_restart = 0   # Index/Counter of model updated with restart
        self.n_all_updating = self.options.modelUpdateByStep.value  # Total <predict or real> number of model updating

        # call function to initialize budgets
        self._check_budget()

        # Execution handling variables
        self._scheduled_jobs = []
        self._scheduled_configs = []
        self._to_eval_jobs = []

        # boolean to check if terminated
        # values in: [False, 'restart', 'exp', 'time', 'budget']
        self.terminated = False

        self.priori_bounds = {} # dict of priori bounds on selected instances for elitist configuration
        self.priori_bounds[0] = self.options.boundMax.value # List of priori bounds based on instance id
                                                            # bounds[0] is the provided boundMax
        if self.options.boundMaxDEV.is_set():
            self.priori_bounds[0] = self.options.boundMaxDEV.value

        # Asyncio related
        self._pool = None

        # self._result_future = asyncio.Future()    # old version in python 3.6
        # since python 3.7, suggested to use:
        #   loop = asyncio.get_running_loop()
        #   future = loop.create_future()
        loop = get_loop()
        self._result_future = loop.create_future()

        self.clock = time.time()

        # Other things to be initialized
        self.do_shuffle_instances = False

        self.returned_exps = []
        self.n_returned = 0

        # Initialize log files
        self.race_logger = RaceLogger(  
                options=self.options,
                logging_variables=self.options.logDir.value + '/logging_variables.log',
                budget=self.budget,
                n_pre_budget=self.n_pre_budget,
                n_configs=self.configurations.n_config,
                n_updating=self.n_all_updating,
            )

        if not self.options.readlogs.is_set():
            self.race_logger.log_ctype(type=RaceType.capping if self.capping else RaceType.quality)
            if self.log_level >= 5:
                self.log_race = options.logDir.value + "/race.log"
                if self.options.recoveryDir.value is None:
                    file = open(self.log_race, "w")
                    file.close()
            self.log_elite = options.logDir.value + "/elite.log"


    def log_state(self):
        """
        Logs the race state variables to the log file set in the object
        """
        # Variables to include the state log
        variables = [
            "first_test_done",
            "n_alive",
            "ranks",
            "best_id",
            "n_experiments",
            "n_instances",
            "n_time",
            "_scheduled_jobs",
            "terminated"
        ]
        self.ranks = {int(x): int(self.ranks[x]) for x in self.ranks.keys()}

        # Get variable values
        state = {x: self.__dict__[x] for x in variables}

        # Create to log file
        if os.path.isfile(self.log_race):
            with open(self.log_race, "w") as f:
                json.dump(state, f, cls=NpEncoder)
        else:
            raise CraceError(
                "Attempt to save race from a non-existent file: ", self.log_race
            )


    def load_state_log(self, log_file):
        """
        Load variables from race log
        :param log_file: File path where the race log can be found
        """
        if os.path.isfile(log_file):
            variables = json.load(open(log_file, "r"))
            # Set variable values
            for v, value in variables.items():
                setattr(self, v, value)
        else:
            raise CraceError(
                "Attempt to recover race from a non-existent file: ", log_file
            )


    def set_pool(self, pool):
        """
        Set pool variable of the race
        :param pool: Pool object
        """
        self._pool = pool

    def syncronous_race(self):
        # TODO: do this in the future
        pass

    async def asynchronous_race(self):
        """
        Starts an asynchronous race and returns a coroutine that needs to be awaited.
        :return: Configuration object containing the set of alive configurations
        """
        # Recover if indicates
        if self.options.recoveryDir.value is not None:
            self.update_status_for_recovering()
            if self.terminated is not False:
                self._terminate_racing()
                return
            # Submit experiments in _scheduled_jobs
            if self.debug_level >= 2:
                print("# Recovering experiments")
            # FIXME: check log handling!
            self.race_logger.print_race_header()

        else:
            # FIXME: check log handling!
            self.race_logger.print_race_header()
            # FIXME: check what happens with deterministic + few instances probably suggest to modify first test
            # Add initial instances
            if self.debug_level >= 3:
                print(
                    "# Adding {} instances to the race".format(
                        self.options.firstTest.value
                    )
                )

            while self.instances.size < self.options.firstTest.value:
                ins_id = self.instances.new_instance()
                if self.capping:
                    self.priori_bounds[ins_id] = self.priori_bounds[0]  # bound of new ins is the default value
            self.n_elites_config = {k: 0 for k in self.instances.instances_ids}

            # Add experiments
            if self.debug_level >= 3:
                print("# Adding initial experiments to the race")
            self._add_experiments(
                self.configurations.get_alive(),
                self.instances.instances,
                self.priori_bounds
            )

        # Wait for race to finish
        await self._result_future

        # Print footer
        self.race_logger.log_task(
            action=TaskType.end,
            instances=self.instances.instances_ids,
            total_instances=self.experiments.get_n_instances(),
            total_configurations=self.configurations.n_config,
            best_id=self.best_id,
            best_configuration=self.configurations.get_configuration(self.best_id),
            tbest_id=self.best_id,
            mean_best=self.mean_best,
            ranks=None,
            t_alive_ids=[],
            alive_ids=self.configurations.get_alive_ids(),
            discarded_ids=[],
            elites=self.elite_ids,
            elite_list=self.elite_list,
            exp_so_far=self.experiments.n_complete,
            pending_exp=len(self._scheduled_jobs),
            pending_config=len(set(self._scheduled_configs)),
            t_time=self.experiments.total_time,
            w_time=time.time() - self.clock,
            used_budget=self.used_budget,
            remaining_budget=self.budget-self.used_budget,
            nb_new_configs=self.c_new_configurations,
            n_pre_budget=self.n_pre_budget,
            idx_updating=self.idx_updating,
            idx_updating_restart=self.idx_updating_restart,
            n_all_updating=self.n_all_updating,
        )

        self.race_logger.print_race_footer(best_id=self.best_id,
                                           mean_best=self.mean_best,
                                           configuration=self.configurations.get_configuration(self.best_id),
                                           elite_ids=self.elite_ids,
                                           elite_list=self.elite_list,
                                           end=True,
                                           terminated=self.terminated
                                           )

        return self.configurations.get_alive()

    
    def read_log_files(self):
        data = self.update_status_for_recovering()
        return data

    def _add_experiments(self, configurations: list, instances: list, bounds: dict, challenges: list=[], stream: list=[]):
        """
        Generates and adds new experiments to the race

        :param configurations: List of configurations which should be executed
        :param instances: List of instances that should be executed
        :param bound: List of bound for executing the experiments, based on instances
        """
        if isinstance(configurations, dict):
            configurations = list(configurations.values())

        # Create experiments
        exp_ids, exps = self.experiments.new_experiments(configurations, instances, bounds)
        if self.capping:
            logger_1.debug(
                "# Adding experiments config: {} instances: {} bound: {} exp ids: {}".format(
                    [x.id for x in configurations],
                    [x.instance_id for x in instances],
                    [bounds[x.instance_id] for x in instances],
                    exp_ids
                )
            )

        else:
            logger_1.debug(
                "# Adding experiments config: {} instances: {} exp ids: {} ".format(
                    [x.id for x in configurations],
                    [x.instance_id for x in instances],
                    exp_ids,
                )
            )

        # Submit experiments
        self._pool.submit_experiments(exps, challenges)
        # Register them in the queue
        self._scheduled_jobs.extend(exp_ids)
        # Register the config_ids of them in the list
        self._scheduled_configs.extend([x.configuration_id for x in exps])

        # update experiments view
        if self.options.instancePatience.value !=0:
            self.exps_view = self.experiments.get_stream_experiments(stream=self.instances.instances_ids)


    def _complete_experiment(self, experiment, result, result_time):
        """
        Mark as completed and save experiment

        :param experiment: Experiment that should be completed
        :param result: Solution quality obtained by the experiment
        :param result_time: Time that the experiment reported as execution time

        :return: Updated experiment (completed)
        """
        assert (
            experiment.experiment_id in self._scheduled_jobs
        ), "Experiment {} is not scheduled".format(experiment.experiment_id)

        # Report completed depending on target_eval
        if not self.target_eval:
            if self.options.testExperimentSize.value <= 1:
                if self.capping:
                    result_experiment = self.experiments.complete_experiment(
                        experiment_id=experiment.experiment_id,
                        budget=experiment.budget,
                        budget_format=experiment.budget_format,
                        solution_quality=result,
                        exp_time=result_time,
                        start_time=experiment.start_time,
                        end_time=experiment.end_time
                    )
                else:
                    result_experiment = self.experiments.complete_experiment(
                        experiment_id=experiment.experiment_id,
                        solution_quality=result,
                        exp_time=result_time,
                        start_time=experiment.start_time,
                        end_time=experiment.end_time
                    )
            else:
                if self.capping:
                    result_experiment = self.experiments.complete_experiment_to_test(
                        experiment_id=experiment.experiment_id,
                        budget=experiment.budget,
                        budget_format=experiment.budget_format,
                        solution_quality=result,
                        exp_time=result_time,
                        start_time=experiment.start_time,
                        end_time=experiment.end_time
                    )
                else:
                    result_experiment = self.experiments.complete_experiment_to_test(
                        experiment_id=experiment.experiment_id,
                        solution_quality=result,
                        exp_time=result_time,
                        start_time=experiment.start_time,
                        end_time=experiment.end_time
                    )
                if self.debug_level >= 4:
                    print(
                        "# Waiting to test experiment {}".format(
                            experiment.experiment_id
                        )
                    )
        else:
            result_experiment = self.experiments.complete_experiment_time(
                experiment_id=experiment.experiment_id,
                exp_time=result_time,
                start_time=experiment.start_time,
                end_time=experiment.end_time
            )
            if self.debug_level >= 4:
                print(
                    "# Waiting to evaluate experiment {}".format(
                        experiment.experiment_id
                    )
                )
            # Add experiments to the jobs to be evaluated
            self._to_eval_jobs.append(experiment.experiment_id)

        # Remove the experiment from the scheduled works
        # and remove the correspoding config_id from the scheduled configs
        if experiment.experiment_id in self._scheduled_jobs:
            index = self._scheduled_jobs.index(experiment.experiment_id)
            del self._scheduled_jobs[index]
            del self._scheduled_configs[index]
        return result_experiment

    def _evaluate_experiment(self, experiment):
        """
        Call to target evaluator when an a set of experiments is done.

        :param experiment: experiment object
        :return: Updated experiment object (completed)
        """
        if self.target_eval:
            assert (
                experiment.experiment_id in self._to_eval_jobs
            ), "Experiment is not waiting for evaluation"
            if self.debug_level >= 2:
                print(
                    "# Triggering evaluation by experiment {}".format(
                        experiment.experiment_id
                    )
                )

            exps = self.experiments.get_waiting_experiments_by_instance_id(
                experiment.instance_id
            )
            alive_ids = self.configurations.get_alive_ids()
            evaluations = self._pool.evaluate_experiments(exps, alive_ids)

            for e, solution_quality in evaluations:
                if self.debug_level >= 3:
                    print("#   evaluating experiment {}".format(e.experiment_id))

                result_experiment = self.experiments.complete_experiment_quality(
                    e.experiment_id, solution_quality
                )
                index = self._to_eval_jobs.index(e.experiment_id)
                del self._to_eval_jobs[index]

            return self.experiments.get_experiment_by_id(experiment.experiment_id)

        else:
            return None


    def _evaluate_experiments(self, experiments):
        """
        Call to target evaluator when an a set of experiments is done.
        :param experiment: experiment object
        :return: Updated experiment object (completed)
        """
        return [self._evaluate_experiment(x) for x in experiments]


    def race_callback(self, experiment: ExperimentEntry, solution_quality, solution_time):
        """
        Called when one experiment is finished. At the moment only called in the async_race.

        New race_callback includes the basic components for storing the information of the finished experiment
        then call test or tests based on option testExperimentSize.

        :param experiment: Experiment object which triggers the callback
        :param solution_quality: Result obtained by the experiment
        :param solution_time: Execution time reported by the experiment
        """
        # check illegel output from the target runner
        illegal_out_q = True if (self.capping
                                 and float(solution_quality) > float(experiment.budget_format)
                                 and float(solution_quality) != float(experiment.bound_max_format)) else False
        illegal_out_t = True if (self.capping
                                 and float(solution_time) > float(experiment.budget_format)) else False
        inf_out_q = True if math.isinf(solution_quality) or math.isnan(solution_quality) else False

        # parse illegal output returned by target algorithm
        # ignore infinite value
        if illegal_out_t or illegal_out_q:
            if illegal_out_t and not inf_out_q:
                if self.debug_level >= 2:
                    print(f"Amend illegal solution time from {solution_time} to budget {experiment.budget_format}")
                solution_time = float(experiment.budget_format)

            if illegal_out_q and not inf_out_q:
                if self.debug_level >= 2:
                    print(f"Amend illegal solution quality from {solution_quality} to budget_max {experiment.bound_max_format}")
                solution_quality = float(experiment.bound_max_format)

        # add experiment information to log file race_log
        self.race_logger.log_experiment_completion(
            experiment,
            solution_quality,
            solution_time,
            self.debug_level
        )

        if self.debug_level >= 3:
            print(
                "# Callback of experiment {} in race".format(experiment.experiment_id)
            )
            aux = [int(x) for x in self._scheduled_jobs]
            print("# Race is waiting for experiments {}".format(aux))

        # parse infinite value
        #   discard all experiments of current configuration since its infinite value
        #   ignore infinite experiments: do not update used budget
        if inf_out_q:
            self._discard(experiment.configuration_id, forbid=True)
            self._extend_race(experiment) # extend after discarding

        else:
            # only for valid experiment/task: add to class Experiments
            # log experiment: add to exp_fin.log or exp_dis.log
            #   update counter in Experiments Object:
            #   n_complete_by_config, n_complete_by_instance, n_complete
            complete_experiment = self._complete_experiment(
                experiment, solution_quality, solution_time
            )

            # update counters in Race object
            # Add new experiment as finished
            self.n_experiments = self.n_experiments + 1
            # Add used time
            self.n_time = self.n_time + solution_time

            # get used budget (either time or experiments)
            # update used budget in whole race
            self.used_budget = self.n_time if self.options.maxTime.value > 0 else self.n_experiments
            # update used budget in current slice
            self.c_used_budget = self.c_used_budget + solution_time if self.options.maxTime.value > 0 else \
                                self.c_used_budget + 1

            # update the counter of evaluate configurations on each instance
            try:
                self.instances.update_config_num(experiment.instance_id)
                self.n_elites_config[experiment.instance_id] += 1
            except:
                if self.debug_level >= 2:
                    print(
                        f"# This finished experiment {experiment.experiment_id} is not in current steam {self.instances.instances_ids}"
                    )
                return

            # obtain view of experiments based on instance stream
            # especially a sub-view when instancePatience != 0
            if self.options.instancePatience.value !=0:
                self.exps_view = self.experiments.get_stream_experiments(stream=self.instances.instances_ids)

            # obtain alive experiment_results
            self.alive_experiment_results = self.experiments.get_results_by_configurations(
                self.configurations.get_alive_ids(), drop_na_instances=False
                )


            # continue test(s) when the result is legal
            if self.options.testExperimentSize.value > 1:  # enable returned list
                # update the counters for returned list
                self.returned_exps.append(complete_experiment)
                self.n_returned = self.n_returned + 1

                # check if call tests
                if self.n_returned >= self.options.testExperimentSize.value:
                    self._call_tests(self.returned_exps, experiment)
                else:
                    self._update_elites(experiment)
                    self._extend_race(experiment)  # extend after finishing an experiment

            else:
                self._call_test(experiment)


        # check if update the model
        self._update_models(experiment)

        # check if update the instance
        self._update_instances()

        self._check_race_termination(experiment)
        if self.log_level >= 5: self.log_state()

        # If the race was terminated, we stop here
        if self.terminated is not False:
            # check if some experiments are waiting to be tested
            if self.options.testExperimentSize.value > 1 and len(self.returned_exps) > 0:
                self._call_tests(self.returned_exps, experiment)

            self._terminate_racing()

            if self.debug_level >= 2:
                print("# Race terminated (exp: {}, conf: {})".format(experiment.experiment_id, experiment.configuration_id))
            return


    def _call_test(self, experiment: ExperimentEntry):
        """
        Call elimination test when one experiment is finished. At the moment only called in the race_callback.

        :param experiment: Experiment object which triggers the callback

        """
        # TODO: we might want to perform a test with a minimum number of configurations completed instead of waiting
        #       to finish all the executions of an instance. Nevertheless, waiting for an instance to finish simplifies
        #       the test
        if self._able_to_test(experiment.configuration_id, experiment.instance_id, self.options.aggressiveTest.value):
            # We evaluate in case there is a target evaluator
            # self._evaluate_experiment(experiment)
            # apply statistical test once
            test_result, alive_ids = self._do_test(experiment.configuration_id)
            self._solve_test_results(test_result, alive_ids)
            # extend race after test: sample ins if there is a challenge
            self._extend_race(experiment, experiment.configuration_id, True)

        else:
            # current experiment is not able to be tested
            self._update_elites(experiment)
            # extend race without challenge
            self._extend_race(experiment)


    def _call_tests(self, experiments, experiment):
        """
        Called when a set of experiments are returned. At the moment only called in the race_callback.

        :param experiments: a list of experiments

        """
        # able_configs: a list of configuration ids that can be tested
        able_configs, able_exps = self._able_to_tests(experiments, self.options.aggressiveTest.value)

        if len(able_exps) > 0:
            self.experiments.complete_experiments(able_exps)
            for x in able_exps:
                index = self.returned_exps.index(self.experiments.all_experiments[x])
                del self.returned_exps[index]

        if len(able_configs) > 0:
            # We evaluate in case there is a target evaluator
            id_experiments = self.exps_view.get_experiments_by_configurations(able_configs,as_dataframe=False)
            experiments = [y for x in id_experiments.values() for y in x]
            # self._evaluate_experiments(experiments)
            # apply statistical test
            test_result, alive_ids = self._do_tests(able_configs)
            self._solve_test_results(test_result, alive_ids)
            self._extend_race(experiment, able_configs, True)

        else:
            self._update_elites(experiments)
            self._extend_race(experiment)

        self.n_returned = 0


    def _solve_test_results(self, test_result, alive_ids):
        """
        solve results from do_test(s)
        """
        if not self.first_test_done: self.first_test_done = True
        # Remove from the race the discarded configurations
        # one more chance for the old best configuration:
        #   if the old best is in the discarded list, refuse to discard
        if len(test_result.discarded) > 0:
            if self.debug_level >= 2:
                print("# Discarded configurations:", test_result.discarded)
            self._discard(test_result.discarded, forbid=False)

        self.race_logger.log_alive_configurations(
            test_result.overall_best, test_result.alive, test_result.discarded, self.debug_level
        )

        ####### from elimination test: select elites
        self.ranks = {k: v for k, v in test_result.overall_ranks.items()
                    if np.isfinite(v)
                    and self.exps_view.get_ncompleted_by_configuration(k) >= self.options.firstTest.value}
        self.best_id = int(test_result.overall_best)
        self.elite_ids = sorted(self.ranks, key=self.ranks.get)[
                        0: min(self.options.maxNbElites.value, len(self.ranks))
                        ]

        ####### from the number of evaluations: select elitist(s)
        completed_exps = self.exps_view.get_ncompleted_by_configurations(self.elite_ids)
        # elitist configurations: configuration(s) with the most evaluations
        self.elitist_ids = [int(k) for k, v in completed_exps.items() if v == max(completed_exps.values())]

        ####### replace elites with elitist or not: elites is used for logs
        if self.options.elitist.value:
            self.elite_ids = self.elitist_ids
        self.elite_list = {x: self.exps_view.get_ncompleted_by_configuration(x) for x in self.elite_ids}

        ####### update old elites
        if self.old_elites[1:] != self.elite_ids and len(self.elite_ids) > 0:
            # update the counter of configurations that an instance can undergo without yielding any improvement
            if self.old_elites[1:] != self.elite_ids[:len(self.old_elites[1:])]:
                self.n_elites_config = {k: 0 for k in self.instances.instances_ids}

            self.configurations.print_to_file(self.elite_ids, self.log_elite)
            self.old_elites[1:] = self.elite_ids

        # FIXME: FIX THIS!
        # decide if we need to shuffle the instances because of the test
        if self.options.shuffleInstancesOnTest.value > 0 and len(test_result.discarded) <= 0:
            self.do_shuffle_instances = True

        # check best and overal best from the test result
        if self.best_id != int(test_result.best):
            self.mean_best = float(np.array(list(self.experiments.get_result_by_configuration(self.best_id, True).values())).mean())
        else:
            self.mean_best = float(test_result.means[test_result.best])

        # log task
        self.race_logger.log_task(
            action=TaskType.test,
            instances=test_result.instances,
            total_instances=self.experiments.get_n_instances(),
            total_configurations=self.configurations.n_config,
            best_id=self.best_id, # test_result.overall_best
            best_configuration=self.configurations.get_configuration(self.best_id),
            tbest_id=test_result.best,
            mean_best=self.mean_best,
            ranks=test_result.ranks,
            t_alive_ids=alive_ids,
            alive_ids=self.configurations.get_alive_ids(),
            discarded_ids=test_result.discarded,
            elites=self.elite_ids,
            elite_list=self.elite_list,
            exp_so_far=self.experiments.n_complete,
            pending_exp=len(self._scheduled_jobs),
            pending_config=len(set(self._scheduled_configs)),   # the number of configs in pending jobs
            t_time=self.experiments.total_time,
            w_time=time.time() - self.clock,
            used_budget=self.used_budget,
            remaining_budget=self.budget-self.used_budget,
            nb_new_configs=self.c_new_configurations,
            n_pre_budget=self.n_pre_budget,
            idx_updating=self.idx_updating,
            idx_updating_restart=self.idx_updating_restart,
            n_all_updating=self.n_all_updating,
        )


    def _get_elite_candidates(self, current_ins):
        """
        Called to get the elite candidates

        :param current_ins: the number of used instances of current experiment
        :param best_ins: the most used instances of the best configuration
        """
        # get the number of elite instances based on the current used instance size
        elite_candidates_n_ins = max(math.ceil(self.n_best_instances * self.options.eliteInstances.value), current_ins)
        assert(self.n_best_instances <= self.exps_view.get_n_instances())

        # get the configurations ids that can be elites based on the number of executions
        alive_ids = self.configurations.get_alive_ids()

        # Generate a list of lists [id, n_exp, mean_exp]
        elite_candidates_means = []
        for conf_id in alive_ids:
            n_exp = self.exps_view.get_ncompleted_by_configuration(conf_id)
            # check experiments are enough to be elite
            if n_exp >= elite_candidates_n_ins:
                all_exp = self.exps_view.get_result_list_by_configuration(conf_id)
                mean_exp = sum(all_exp)/len(all_exp)
                elite_candidates_means.append([conf_id, n_exp, mean_exp])

        # sort the alive configurations first by n_exp and then by mean_exp
        if len(elite_candidates_means) > 1:
            sorted(elite_candidates_means, key=lambda e: (e[1], e[2]))

        # If there are more elite than needed keep the maxNbElites first
        if len(elite_candidates_means) > self.options.maxNbElites.value:
            elite_candidates_means = elite_candidates_means[0:self.options.maxNbElites.value]

        # gather the ids of the elite configurations
        elite_candidates_ids = [x[0] for x in elite_candidates_means]

        return elite_candidates_ids


    def _able_to_test(self, configuration_id: int, instance_id: int, aggressive_test: bool):
        """
        If the configuration has been tested at least `firstTest` times, and if it's been tested a multiple
        of `eachTest` times, then we can trigger the test

        :param configuration_id: the id of the configuration to test
        :type configuration_id: int
        :param instance_id: the id of the instance to test on
        :type instance_id: int
        :param aggressive_test:
        :type aggressive_test: bool
        :return: A boolean value.
        """
        if (
            self.exps_view.get_ncompleted_by_configuration(configuration_id)
            < self.options.firstTest.value
        ):
            return False

        if (
            (self.exps_view.get_ncompleted_by_configuration(configuration_id) - self.options.firstTest.value)
            % self.options.eachTest.value
            != 0
        ):
            return False

        finished_ins = list(self.exps_view.get_result_by_configuration(
                        configuration_id, drop_na_instances=True).keys())
        stream_ins = self.instances.instances_ids

        # check if finished experiments are on continous instances
        if self.options.testExperimentSize.value > 1 and not set(stream_ins).issubset(set(finished_ins)):
            return False

        # check if current configuration exeuctes new instance faster than the elite configurations
        if (self.options.domType.value in ("adaptive", "adaptive-dom", "adaptive-dominance", "adaptive-Dominance")
            and len(self.elitist_ids)) > 0:
            trigger_instances = list(self.exps_view.get_result_by_configuration(
                            int(self.elitist_ids[0]), drop_na_instances=True).keys())
            if trigger_instances != stream_ins[:len(trigger_instances)]:
                return False

        if len(self.old_elites) > 2 and configuration_id in self.old_elites[1:]:
            if instance_id < self.exps_view.get_n_instances():
                return False

        if aggressive_test:
            trigger_instances = list(
                self.exps_view.get_result_by_configuration(
                    configuration_id, drop_na_instances=True
                ).keys()
            )
            instances = []
            for x in self.instances.instances_ids:
                if x in trigger_instances and x not in instances:
                    instances.append(x)
            exps = self.alive_experiment_results.loc[instances, :]
            exps = exps.dropna(axis=1, how="any")

            if exps.shape[1] > 1:
                return True

            return False

        else:
            exps = self.experiments.get_pending_experiments_by_instance_id(instance_id)
            if len(exps) > 0:
                return False

            for experiment_id in self._scheduled_jobs:
                experiment = self.experiments.get_experiment_by_id(
                    experiment_id, as_dataframe=False
                )
                if instance_id == experiment.instance_id:
                    return False

            return True


    def _able_to_tests(self, experiments, aggressive_test: bool):
        """
        check if any returned configruations can be tested by calling able_to_test

        :param experiments: a list of returned experiments
        :type  aggressive_test: bool

        :return: a list of configuration ids that can be tested
        :return: a list of experiment ids that can be tested
        """

        returned_configs = []
        returned_exps = []

        for x in experiments:
            if self._able_to_test(x.configuration_id, x.instance_id, aggressive_test):
                if x.configuration_id not in returned_configs:
                    returned_configs.append(x.configuration_id)
                if x.experiment_id not in returned_exps:
                    returned_exps.append(x.experiment_id)

        return returned_configs, returned_exps


    def _do_test(self, id_config: int) -> Tuple[EliminationTestResults, List[int]]:
        """
        Perform statistical test for single configuration, generate ranks and discard configurations
        :param id_config: Id of the configuration the triggered the test

        :return: test_result is s dictionary with test results, alive_ids is a list of ids that are alive
        """

        # Get the instances executed by the triggering configuration
        trigger_instances = list(self.exps_view.get_result_by_configuration(id_config, drop_na_instances=True).keys())

        # selected intances
        instances = []
        for x in self.instances.instances_ids:
            if x in trigger_instances and x not in instances:
                instances.append(x)

        best_ins = list(self.exps_view.get_result_by_configuration(self.best_id, drop_na_instances=True).keys()) \
            if self.best_id is not None else self.instances.instances_ids
        self.n_best_instances = len(best_ins)

        experiment_results = self.alive_experiment_results

        # Perform elimination test
        if self.debug_level >= 3:
            print("# Performing elimination test")
        test_result = self.eliminator.test_elimination(
            experiment_results=experiment_results.copy(),
            instances=instances,
            ins_stream=self.instances.instances_ids,
            min_test=self.options.firstTest.value,
            n_best=self.options.eliteRef.value,
        )

        # Log statistical test
        self.race_logger.log_elimination_test(test_result, self.debug_level)

        final_best = test_result.overall_best
        ranks = test_result.overall_ranks

        #FIXME: Check if this is needed!!
        # remove the elites config and the trigger config
        elites_id_by_so_far_experiments = []
        if test_result.data.shape[0] != self.exps_view.get_n_instances():
            elites_id_by_so_far_experiments.extend(self.elite_ids)

        discarded_ids = list(set(test_result.discarded) - set(elites_id_by_so_far_experiments))
        alive_ids = test_result.alive

        if self.debug_level >= 3:
            print("# Available data:")
            print(test_result.overall_data)
            print("# On test Instances :", test_result.instances)
            print("# On test Configurations :", test_result.data.columns.values)

        if self.debug_level >= 3:
            p1_str = ",".join([f"{x}:{y}" for x, y in sorted(test_result.overall_ranks.items(), key=lambda item: item[1])])
            print("# Ranks after statistical test (id:rank): {}".format(p1_str))

        return test_result, alive_ids


    def _do_tests(self, id_configs: list):
        """
        Perform statistical test for multiple configurations, generate ranks and discard configurations
        :param id_configs: a list of configurations can be tested

        :return: test_result is s dictionary with test results, alive_ids is a list of ids that are alive
        """

        # a dict: config_id: max_ins
        id_instances = {x: max(self.exps_view.get_result_by_configuration(x, drop_na_instances=True).keys()) for x in id_configs}

        test_result = returned_results = None
        test_experiment_results = self.alive_experiment_results.copy()
        returned_discarded = []
        returned_instances = []
        returned_best = returned_means = None

        # do test from the most instances to the least
        for i, x in enumerate(reversed(sorted(set(id_instances.values())))):
            test_configs = [k for k,v in id_instances.items() if x >= self.options.firstTest.value and x <= v]

            # Get the instances executed by the triggering configuration
            trigger_instances = self.exps_view.get_results_by_configurations(test_configs, drop_na_instances=True).index.tolist()
            # selected intances
            test_instances = []
            for x in self.instances.instances_ids:
                if x in trigger_instances and x not in test_instances:
                    test_instances.append(x)
            # Perform elimination test
            if self.debug_level >= 3:
                print("# Performing elimination test")

            if test_result:
                test_experiment_results = test_experiment_results[test_result.alive]

            test_result = self.eliminator.test_elimination(
                experiment_results=test_experiment_results.copy(),
                instances=test_instances,
                ins_stream=self.instances.instances_ids,
                min_test=self.options.firstTest.value,
                n_best=self.options.eliteRef.value
            )

            returned_discarded += test_result.discarded
            returned_instances += test_result.instances
            if i == 0:
                returned_best = test_result.best
                returned_means = test_result.means

        if i > 0:
            returned_results = EliminationTestResults(best=returned_best,
                                                        means=returned_means,
                                                        ranks=test_result.overall_ranks,
                                                        p_value=test_result.p_value,
                                                        alive=test_result.alive,
                                                        discarded=list(set(returned_discarded)),
                                                        instances=list(set(returned_instances)),
                                                        data=self.alive_experiment_results.copy(),
                                                        overall_best=test_result.overall_best,
                                                        overall_mean=test_result.overall_mean,
                                                        overall_ranks=test_result.overall_ranks,
                                                        overall_data=test_result.overall_data)
        else:
            returned_results = test_result

        # Log statistical test
        self.race_logger.log_elimination_test(test_result, self.debug_level)

        if self.debug_level >= 3:
            print("# Available data:")
            print(test_result.overall_data)
            print("# On test Instances :", test_result.instances)
            print("# On test Configurations :", test_result.data.columns.values)

        if self.debug_level >= 2:
            p1_str = ",".join([f"{x}:{y}" for x, y in sorted(test_result.overall_ranks.items(), key=lambda item: item[1])])
            print("# Ranks after statistical test (id:rank): {}".format(p1_str))

        return returned_results, returned_results.alive


    def _check_budget(self):
        """
        Check budget for current slice
        """
        if self.n_pre_budget == 0:
            # priority of maxTime is greater
            self.n_pre_budget = math.ceil(
                self.budget / (self.n_all_updating + 1)
            )
            if self.options.maxTime.value != 0:
                self.n_pre_budget = math.ceil(
                    self.budget / (self.n_all_updating + 1)
                )
        else:
            remaining = self.budget - self.used_budget
            self.n_pre_budget = math.ceil(
                remaining / min(1, self.n_all_updating - self.idx_updating + 1)
            )


    def _able_to_update_ins(self):
        """
        check if update instances
        :return: a integer value
        """
        ids = []
        for k, v in self.n_elites_config.items():
            if v >= self.options.instancePatience.value:
                ids.append(k)
        return ids


    def _update_instances(self):
        """
        update instance stream when the old instance is difficult to recognise a better configuration
        """
        if self.options.instancePatience.value == 0: return

        instance_ids = self._able_to_update_ins()
        if len(instance_ids) > 0:
            new_ins = []
            self.instances.update_instances_stream(instance_ids)
            for _ in range(0, len(instance_ids)):
                new_ins.append(self.instances.new_instance())
            self.n_elites_config = {k: 0 for k in self.instances.instances_ids}
            if self.debug_level >= 2:
                print("# Update instances stream..")
                print(f"#   Current instances stream: {self.instances.instances_ids}")

            # discard experiments
            self._discard_by_ins(instance_ids)

            # if capping: update time bound for target algorithm on each used instance
            if self.capping:
                self.priori_bounds = self.experiments.calculate_priori_bounds(
                    instance_ids=self.instances.instances_ids,
                    elitist_ids=self.elitist_ids,
                    bound_max=self.priori_bounds[0],
                    domType=self.options.domType.value,
                    new_ins=new_ins)

            self._add_experiments(
                self.configurations.get_alive(),
                self.instances.get(new_ins),
                self.priori_bounds,
                challenges=self.elitist_ids,
                )

            self.race_logger.log_task(
                action=TaskType.instance_stream,
                instances=self.instances.instances_ids,
                total_instances=self.experiments.get_n_instances(),
                total_configurations=self.configurations.n_config,
                best_id=self.best_id,
                best_configuration=self.configurations.get_configuration(self.best_id),
                tbest_id=self.best_id,
                mean_best=self.mean_best,
                ranks=None,
                t_alive_ids=[],
                alive_ids=self.configurations.get_alive_ids(),
                discarded_ids=[],
                elites=self.elite_ids,
                elite_list=self.elite_list,
                exp_so_far=self.experiments.n_complete,
                pending_exp=len(self._scheduled_jobs),
                pending_config=len(set(self._scheduled_configs)),
                t_time=self.experiments.total_time,
                w_time=time.time() - self.clock,
                used_budget=self.used_budget,
                remaining_budget=self.budget-self.used_budget,
                nb_new_configs=self.c_new_configurations,
                n_pre_budget=self.n_pre_budget,
                idx_updating=self.idx_updating,
                idx_updating_restart=self.idx_updating_restart,
                n_all_updating=self.n_all_updating,
            )


    def _discard_by_ins(self, instance_ids, forbid=False):
        """
        Discard experiments based on provided instances
        """
        for ins in instance_ids:
            # get all experiments pending from the configurations
            pending_exps = self.experiments.get_pending_experiments_by_instance_id(ins)

            # get all experiments pending from the configurations
            waiting_exps = self.experiments.get_waiting_experiments_by_instance_id(ins)

            # A. Discard "pending" and "waiting" experiments from Experiment Entry
            for exp in pending_exps + waiting_exps:
                self.experiments.discard_experiment(exp.experiment_id)
            # C. Discard scheduled experiments
                if exp.experiment_id in self._scheduled_jobs:
                    index = self._scheduled_jobs.index(exp.experiment_id)
                    del self._scheduled_jobs[index]
                    del self._scheduled_configs[index]

            # B. Remove experiments from the waiting queue and executioner
            self._pool.cancel_experiments(None, pending_exps)
            if self.debug_level >= 2: print(f"Cancel experiments of selected instance [{ins}]: {[x.experiment_id for x in pending_exps]}")


    def _able_to_update_model(self):
        """
        Check if update model

        :return: A boolean value.
        """
        if self.used_budget < self.budget:
            if len(self.elite_ids) > 0:
                # check num of sampled configurations
                if self.c_new_configurations >= 2048:
                    return True

                # check budget
                if self.c_used_budget >= self.n_pre_budget:
                    return True

        return False


    def _update_models(self, experiment):
        """
        Update models of alive configurations
        """
        # check the rule used for updating the model
        if not self.options.modelUpdateBySampling.value:
            # check if update the models or not
            if self._able_to_update_model():
                if self.debug_level >= 2:
                    print("# Updating model")

                # update the counters
                self.idx_updating += 1
                self.idx_updating_restart += 1
                # check if update self.n_all_updating
                if self.idx_updating > self.n_all_updating: self.n_all_updating = self.idx_updating

                # log the model information
                # total: total times of model updating
                # current: the time of model updating in current loop (restart or not)
                logger_1.info(f"# Model update {self.idx_updating}/{self.idx_updating_restart} (total/current), used budget {self.used_budget}")

                # log the information in file slice.log
                # slice.log is for plotting
                logger_1.info(f"# Elite configs: {self.elite_list}")

                # update all the models
                # after restarting: comparing the factor to update models to those before updating:
                #                   same: nb_model_updated=self.idx_updating_restart,
                #                         total_model_update=self.options.modelUpdateByStep.value
                #                   slower: nb_model_updated=self.idx_updating_restart
                #                           total_model_update=self.n_all_updating
                self.model.update(configs=self.configurations.get_configurations(self.elite_ids),
                                  nb_new_configs=self.c_new_configurations,
                                  nb_model_updated=self.idx_updating_restart,
                                  total_model_update=self.options.modelUpdateByStep.value,
                                  alive_configs=self.configurations.get_alive(),
                                  by_sampling=False)
                
                # update slice information
                self._update_slice_info(experiment, restart=False)

    def _update_slice_info(self, experiment, restart: bool=False):
        """
        update details of each slice when updating or restarting models
        """
        task_type = TaskType.model_update if not restart else TaskType.model_restart
        task_type_str = 'update' if not restart else 'restart'

        model_info = ModelInfo(type=task_type_str,
                               n_all_updating=self.idx_updating,
                               idx_updating=self.idx_updating_restart,
                               last_config_id=self.last_config_id,
                               used_budget=self.used_budget,
                               best_so_far=self.best_id,
                               best_mean_so_far=self.mean_best,
                               elites=self.elite_ids,
                               alive_model_ids=self.configurations.get_alive_model_ids(),
                               end_time=experiment.end_time)
        logger_2.info(json.dumps(model_info.as_dict()))

        self._check_budget()

        self.race_logger.log_task(
            action=task_type,
            instances=self.instances.instances_ids,
            total_instances=self.experiments.get_n_instances(),
            total_configurations=self.configurations.n_config,
            best_id=self.best_id,
            best_configuration=self.configurations.get_configuration(self.best_id),
            tbest_id=self.best_id,
            mean_best=self.mean_best,
            ranks=None,
            t_alive_ids=[],
            alive_ids=self.configurations.get_alive_ids(),
            discarded_ids=[],
            elites=self.elite_ids,
            elite_list=self.elite_list,
            exp_so_far=self.experiments.n_complete,
            pending_exp=len(self._scheduled_jobs),
            pending_config=len(set(self._scheduled_configs)),
            t_time=self.experiments.total_time,
            w_time=time.time() - self.clock,
            used_budget=self.used_budget,
            remaining_budget=self.budget-self.used_budget,
            nb_new_configs=self.c_new_configurations,
            n_pre_budget=self.n_pre_budget,
            idx_updating=self.idx_updating,
            idx_updating_restart=self.idx_updating_restart,
            n_all_updating=self.n_all_updating,
        )

        self.c_used_budget = 0
        self.c_new_configurations = 0

    def _update_elites(self, experiments):
        """
        only elitist
        """
        sel = self.experiments.get_ncompleted_by_configurations(self.configurations.get_alive_ids())
        sel_ids = [k for k, v in sel.items() if v == max(sel.values())]

        self.elite_ids = self.old_elites[1:] if len(self.old_elites) > 1 else [random.choice(sel_ids)]
        self.elitist_ids = self.old_elites[1:] = self.elite_ids
        self.elite_list = {x: self.exps_view.get_ncompleted_by_configuration(x) for x in self.elite_ids}
        self.best_id = self.elitist_ids[0]

        if self.exps_view.get_ncompleted_by_configuration(self.best_id) > 0:
            self.mean_best = float(np.array(list(self.exps_view.get_result_by_configuration(self.best_id, True).values())).mean())

        if not isinstance(experiments, list):
            instances=list(self.exps_view.get_result_by_configuration(experiments.configuration_id, drop_na_instances=True).keys())
        else:
            instances=list(self.exps_view.get_results_by_configurations([x.configuration_id for x in experiments], drop_na_instances=True).keys())

        self.race_logger.log_task(
            action=TaskType.completion,
            instances=instances,
            total_instances=self.experiments.get_n_instances(),
            total_configurations=self.configurations.n_config,
            best_id=self.best_id,
            best_configuration=self.configurations.get_configuration(self.best_id),
            tbest_id=self.best_id,
            mean_best=self.mean_best,
            ranks=self.ranks,
            t_alive_ids=[],
            alive_ids=self.configurations.get_alive_ids(),
            discarded_ids=[],
            elites=self.elite_ids,
            elite_list=self.elite_list,
            exp_so_far=self.experiments.n_complete,
            pending_exp=len(self._scheduled_jobs),
            pending_config=len(set(self._scheduled_configs)),
            t_time=self.experiments.total_time,
            w_time=time.time() - self.clock,
            used_budget=self.used_budget,
            remaining_budget=self.budget-self.used_budget,
            nb_new_configs=self.c_new_configurations,
            n_pre_budget=self.n_pre_budget,
            idx_updating=self.idx_updating,
            idx_updating_restart=self.idx_updating_restart,
            n_all_updating=self.n_all_updating,
        )

    def _discard(self, configuration_ids, forbid=False):
        """
        Discard the configurations provided

        :param configuration_ids: Id of the configurations to be discarded
        :param forbid: Boolean indicating if the configuration should be forbidden
        """
        _exps_view = self.experiments if self.exps_view is None else self.exps_view

        if not isinstance(configuration_ids, list):
            configuration_ids = [configuration_ids]

        # STEP 1: Discard all the related experiments
        #         A. including all the unfinished experiments in the Experiment Entry
        #            experiment state: ["pending", "waiting"]
        #         B. and all the running experiments in the Execution Pool
        #            experiment: waiting in the pool (waiting queue) and running in the executioner
        #                        not completed experiments (pending exps)
        #         C. and experiments in the scheduled jobs

        #         D. experiments in the returned list

        # CHECK elite configurations
        for x in configuration_ids:
            if x in self.elite_ids: self.elite_ids.pop(self.elite_ids.index(x))
            if x in self.old_elites[1:]: self.old_elites.pop(self.old_elites.index(x))
        if len(self.elite_ids) < 1:
            completed_exps = _exps_view.get_ncompleted_by_configurations(self.configurations.get_alive_ids())
            if max(completed_exps.values(), default=0) > 0:
                self.elitist_ids = [int(k) for k, v in completed_exps.items() if v == max(completed_exps.values())]
            else:
                self.elitist_ids = [1]
            self.elite_ids = self.old_elites[1:] = self.elitist_ids
            self.best_id = self.elite_ids[0]

        # get all experiments pending from the configurations
        pending_exps = self.experiments.get_pending_experiments_by_conf_ids(configuration_ids)

        # get all experiments pending from the configurations
        waiting_exps = self.experiments.get_waiting_experiments_by_conf_ids(configuration_ids)

        # A. Discard "pending" and "waiting" experiments from Experiment Entry
        for exp in pending_exps + waiting_exps:
            self.experiments.discard_experiment(exp.experiment_id)
        # C. Discard scheduled experiments
            if exp.experiment_id in self._scheduled_jobs:
                index = self._scheduled_jobs.index(exp.experiment_id)
                del self._scheduled_jobs[index]
                del self._scheduled_configs[index]

        # B. Remove experiments from the waiting queue and executioner
        self._pool.cancel_experiments(configuration_ids, pending_exps)

        # D. remove all related experiments waiting to be tested
        if len(self.returned_exps) > 0:
            for exp in reversed(self.returned_exps):
                if exp.configuration_id in configuration_ids:
                    self.returned_exps.pop(self.returned_exps.index(exp))
                    self.n_returned -= 1

        # STEP 2:  Discard models
        # self.model.discard_models(configuration_ids)

        # STEP 3: Discard configurations
        # Mark as forbidden or simple discard
        if not forbid:
            self.configurations.discard_configurations(configuration_ids)
        else:
            for x in configuration_ids: self.configurations.discard_forbidden(x)

            # add_forbidden_configuration()
            # :param Configuration

            # get_confiigurations()
            # :param configuration_ids: Any
            # :param as_list: bool=True related to the type of return
            # :return List[Configuration] (List[Dict]) if :param as_list=True
            #         Dict[int: Configuration] (List[int: Dict])  if :param as_list=False
            for i in configuration_ids:
                self.model.add_forbidden_configuration(self.configurations.get_configuration(i))

    def _extend_race(self, experiment, config_ids=None, test_flag: bool=False):
        """
        Adds experiments to the race increasing either configurations or instances
        :param: config_ids: used to check if there is a challenge.
        :param: test_flag: a flag for test. if current task is in test, flag is True.
        :return: Task type (either configuration or instance) depending on what was added
        """
        # We enforce to have a at least 1) a minimum number pf scheduled experiments and 2) a minumum number of configurations executing
        # FIXME: we might not want to schedule configurations over the min number of experiments.
        # FIXME: add case in which no instance can be generated because of deterministic setup
        # TODO: maybe we would like to submit experiments in a different order
        return_type = []
        _exps_view = self.experiments if self.exps_view is None else self.exps_view

        # check if there is a challenge, use self.challenge_flag to indicate
        #   when current config just finished its last exp (all the sampled instance) 
        #   and it is not be discarded
        self.challenge_flag = False

        if test_flag:   # challenge can only be generated after elimination test
            if isinstance(config_ids, int):
                if (_exps_view.get_ncompleted_by_configuration(config_ids) == _exps_view.get_n_instances()
                    and len(self.elitist_ids) > 1
                ):
                    self.challenge_flag = True
            elif isinstance(config_ids, list):
                returned_exps = _exps_view.get_ncompleted_by_configurations(config_ids)
                selected = [x == _exps_view.get_n_instances() for x in returned_exps.values()]
                if (
                    len(selected) > 1 and len(self.elitist_ids) > 1
                ):
                    self.challenge_flag = True

        # FIXME: option is to control the rules for entering this function
        while (
            len(self._scheduled_jobs) < self.num_cores
            or len(self.configurations.get_alive_ids()) < self.options.nbConfigurations.value
            # check the number of pending configs to immediately sample instance when there is a challenge
            or (self.options.sampleInsStrict.value and self.challenge_flag)
            or ( # guarantee the order of finished instances for adaptive dominance test
                self.strict_pool and len(set(self._scheduled_configs)) < self.num_cores
            )
        ):
            if self.debug_level >= 3:
                print("# Sampling is required")

            # Choose to sample configurations or add new instances
            flag = self._get_sample_type()
            instance_ids = []

            if (SamplePolicy.instance_challenge in flag
                or SamplePolicy.instance_expect in flag
                ):
                # adding one instance at the time
                # TODO: maybe here we would like first to shuffle current instances... this would affect the order in which the experiments are sent
                for _ in range(0, self.options.eachTest.value):
                    instance_ids.append(self.instances.new_instance())
                    self.n_elites_config[instance_ids[-1]] = 0
                self.old_elites[0] += 1

                if self.capping: self.priori_bounds[instance_ids[-1]] = self.priori_bounds[0]
                if self.options.shuffleInstancesOnNew.value:
                    self.do_shuffle_instances = True

                if SamplePolicy.instance_challenge in flag:
                    self._add_experiments(
                        self.configurations.get_alive(),
                        self.instances.get(instance_ids),
                        self.priori_bounds,
                        challenges=self.elitist_ids,
                        )
                    return_type.append(TaskType.instance_challenge)

                else:
                    self._add_experiments(
                        self.configurations.get_alive(),
                        self.instances.get(instance_ids),
                        self.priori_bounds,
                        challenges=[self.best_id],
                        )
                    return_type.append(TaskType.instance_expect)

            # Shuffle instances before submitting experiments of new configurations
            if self.do_shuffle_instances:
                if self.debug_level >= 2:
                    print("# Shuffling instances...")
                self.instances.shuffle_instances()
                self.do_shuffle_instances = False

            if SamplePolicy.configuration in flag:
                # add configurations
                new_configurations = []
                n1 = n2 = 0
                # sample config based on alive configs (compared to nbConfigurations)
                if len(self.configurations.get_alive_ids()) < self.options.nbConfigurations.value:
                    n1 = self.options.nbConfigurations.value - len(self.configurations.get_alive_ids())
                # sample config to gurantee the configs on workers are different for adaptive dom
                if self.strict_pool and (len(set(self._scheduled_configs)) < self.num_cores):
                    n2 = self.num_cores - len(set(self._scheduled_configs))
                n_to_add = max(1, n1, n2)

                if self.debug_level >= 3:
                    print("# Sampling {} configuration(s)".format(n_to_add))

                try:
                    for i in range(1, n_to_add+1):
                        configuration = self._add_configuration_from_model(experiment)[-1]
                        new_configurations.append(configuration)
                except TypeError:
                    if self.terminated is not False: return

                self.c_new_configurations += n_to_add

                if self.capping:
                # priori bound is only related to the elite config
                # it is the same on the same ins for different configs
                    # for _ in new_configurations:
                    self.priori_bounds = self.experiments.calculate_priori_bounds(
                        instance_ids=self.instances.instances_ids,
                        elitist_ids=self.elitist_ids,
                        bound_max=self.priori_bounds[0],
                        domType=self.options.domType.value)
                self._add_experiments(new_configurations, self.instances.instances, self.priori_bounds)

                return_type.append(TaskType.configuration)

        if len(return_type) > 0: #and PRINT_CURRENT_TASK:
            unique_task = set(return_type)
            for task in unique_task:
                # Adding instances/configurations tasks
                self.race_logger.log_task(
                    action=task,
                    instances=self.instances.instances_ids,
                    total_instances=self.experiments.get_n_instances(),
                    total_configurations=self.configurations.n_config,
                    best_id=self.best_id,
                    best_configuration=self.configurations.get_configuration(self.best_id),
                    tbest_id=self.best_id,
                    mean_best=self.mean_best,
                    ranks=None,
                    t_alive_ids=[],
                    alive_ids=self.configurations.get_alive_ids(),
                    discarded_ids=[],
                    elites=self.elite_ids,
                    elite_list=self.elite_list,
                    exp_so_far=self.experiments.n_complete,
                    pending_exp=len(self._scheduled_jobs),
                    pending_config=len(set(self._scheduled_configs)),
                    t_time=self.experiments.total_time,
                    w_time=time.time() - self.clock,
                    used_budget=self.used_budget,
                    remaining_budget=self.budget-self.used_budget,
                    nb_new_configs=self.c_new_configurations,
                    n_pre_budget=self.n_pre_budget,
                    idx_updating=self.idx_updating,
                    idx_updating_restart=self.idx_updating_restart,
                    n_all_updating=self.n_all_updating,
                )

    def _get_sample_type(self) -> SamplePolicy:
        """
        Decides if we should sample configurations or add new instances. We sample
        configurations when:
        1. We have less alive configurations than the minimum defined
        2. There is no configuration aiming to join the "elite" set
        :return: Boolean indicating if we should sample configurations
        """
        flags = SamplePolicy(0)

        if self.challenge_flag:
        # when there is a challenge:
        #   configurations are tested on all instances but no one is better
        #   len(scheduled_configs) decreases
            # flags |= SamplePolicy.instance
            if random.random() <= self.options.sampleInsFrequency.value: flags |= SamplePolicy.instance_challenge
            self.challenge_flag = False

        if self.options.expectInstances.is_set() and self.options.expectInstances.value > 0:
            # get used budget (either time or experiments)
            if (self.used_budget / self.budget >= 0.25
                and int((self.used_budget-0.25*self.budget) / (0.75*self.budget) * self.expect_ins_num) > self.instances.size
            ):
                flags |= SamplePolicy.instance_expect

        # TODO: implement this
        # not enough configurations in the race
        #   1. discard the bad configuration(s) (alive config decreases)
        if (
            SamplePolicy.instance_challenge not in flags
        ):
            flags |= SamplePolicy.configuration

        
        # # maximum number of instances added to increase precision
        # # FIXME: add parameter to control this
        # if self.idx_updating > 2:
        #     return SamplePolicy.configuration

        # if self.n_best_changed >= 4:
        #     self.n_best_changed = 0
        #     # if the best has changed multiples times add a instance to increase precision
        #     return SamplePolicy.instance

        # if self.best_id and self.n_best_alive >= 2:
        #     completed_experiments = self.experiments.get_ncompleted_by_configurations(
        #         self.elite_ids
        #     )
        #     best = completed_experiments[self.best_id]

        #     adversary_ids = [
        #         value
        #         for (key, value) in completed_experiments.items()
        #         if key != self.best_id and value == best
        #     ]

        #     if len(adversary_ids) >= 2:
        #         self.n_best_alive = 0
        #         return SamplePolicy.instance | SamplePolicy.elites

            # if self.n_best_alive > 2:
            #     self.n_best_alive -= 1
            #     return SamplePolicy.instance

        # if self.it_instance_sampling > 2:
        #     self.it_instance_sampling = 0
        #     return SamplePolicy.instance

        return flags

    def _add_configuration_from_model(self, experiment):
        """
        Samples and adds to the race a new configuration
        :return: Configuration object
        """
        # select a parent configuration
        parent_id = self._select_parent()

        # FIXME: implement the rule for updating by sampling
        if self.options.modelUpdateBySampling.value:
            self.idx_updating += 1
            self.idx_updating_restart += 1
            # check if update self.n_all_updating
            if self.idx_updating > self.n_all_updating: self.n_all_updating = self.idx_updating
            logger_1.info(f"# Model update {self.idx_updating}/{self.idx_updating_restart} (total/current) id {parent_id}")
            self.model.update(configs=self.configurations.get_configuration(parent_id),
                              nb_new_configs=self.c_new_configurations,
                              nb_model_updated=self.idx_updating_restart,
                              total_model_update=self.options.modelUpdateByStep.value,
                              alive_configs=self.configurations.get_alive(),
                              by_sampling=True)
            self.c_new_configurations = 0

        logger_1.info("# Sampling new configuration from parent {}".format(parent_id))

        # only sample one new configuration
        # if id_conf is -2, it means the sampled new configuration exsits already
        # then sample a new config repeatly
        id_conf = [-2]
        repeat_time = 100
        while id_conf[0] == -2 and repeat_time >= 0:
            if repeat_time == 0:
                if self.options.softRestart.value != 0:

                    # Do restart in limited number of times
                    if (self.options.softRestart.value != -1 
                        and self.RESTART_TIMES >= self.options.softRestart.value):
                        print(f"WARNING: Already applied softRestart {self.RESTART_TIMES} times.\n"
                              f"         Terminating racing procedure..")
                        self.terminated = 'restart'
                        return None

                    # restart: FIXME (parameters for updating model) set the Xth level
                    #   model to the initial: self.idx_updating_restart = 0
                    #   model to the Xth level: self.idx_updating_restart = X - 1
                    self.idx_updating += 1            # Restarting is also a form of updating.
                    self.idx_updating_restart = 1     # Index of model updating with restart
                    # factor = None    # the factor for updating model
                    # check if update self.n_all_updating
                    if self.idx_updating > self.n_all_updating: self.n_all_updating = self.idx_updating

                    if self.debug_level >= 2:
                        print("# Doing soft restart...")

                    # restart all alive configurations
                    self.model.soft_restart(configs=self.configurations.get_alive(),
                                            elites=self.elitist_ids, model_id=self.idx_updating_restart)
                    
                    # restart only elite configurations
                    # self.model.soft_restart(configs=[self.configurations.get_configuration(parent_id)],
                    #                         elites=self.elitist_ids, model_id=self.idx_updating_restart)
                    
                    # update slice information
                    self._update_slice_info(experiment, restart=True)

                    repeat_time = 100
                    self.RESTART_TIMES += 1

                else:
                    print(f"WARNING: No new configuration generated after {repeat_time} tries. Perhaps activate softRestart?")
                    self.terminated = 'restart'
                    return None

            sampled = self.model.sample_configuration(self.configurations.get_configuration(parent_id))
            id_conf = self.configurations.add_from_model([sampled])
            repeat_time -= 1

        self.last_config_id = id_conf[-1]
        configuration = self.configurations.get_configurations(id_conf)

        return configuration

    def _select_parent(self):
        """
        Selects a configuration to use for sampling
        :return: Configuration id
        """
        # we select a configuration from the elite
        if len(self.elite_ids) > 0:
            return random.choice(self.elite_ids)
        else:
            return random.choice(self.configurations.get_alive_ids())
        # return self.elite_ids[0]

    def _clean_queue(self):
        """
        Cancels all scheduled experiments and cleans the queue
        """
        # all experiments are closed
        for exp_id in self._scheduled_jobs:
            experiment = self.experiments.get_experiment_by_id(
                exp_id, as_dataframe=False
            )
            self._pool.cancel_experiment(experiment)

            self._update_alive_elites(experiment)

        self._scheduled_jobs = []

    def _update_final_elites(self):
        """
        Delete incorrect 'elite configuration' when the budget is met.
        """
        while self.exps_view.get_ncompleted_by_configuration(self.elite_ids[-1]) < self.options.firstTest.value:
            self.elite_ids.pop()
        self.configurations.print_to_file(self.elite_ids, self.log_elite)

    def _update_alive_elites(self, exp):
        """
        Delete uncompleted configurations from the alive list
        """
        if exp.configuration_id in self.elite_ids:
            self._update_final_elites()

        if exp.configuration_id in self.configurations.get_alive_ids() and \
        self.exps_view.get_ncompleted_by_configuration(exp.configuration_id) < self.options.firstTest.value:
            self.configurations.discard_configuration(exp.configuration_id)

    def _check_race_termination(self, exp):
        """
        check if terminate
        """
        logger_1.debug(
            "# Termination check. Total experiments: {} Total time: {}".format(
                self.n_experiments, self.experiments.total_time
            )
        )
        # capping version has both/either maxTime and/or  maxExperiments
        # check maxTime first
        if self.capping and self.options.maxTime.value <= self.experiments.total_time:
            self.terminated = 'time'

        elif self.options.maxExperiments.value != 0:
            assert self.n_experiments == (
                self.experiments.n_complete + self.experiments.n_waiting
            ), (
                "discordant count of experiments "
                + str(self.n_experiments)
                + " but "
                + str(self.experiments.n_complete + self.experiments.n_waiting)
            )
            if self.options.maxExperiments.value <= self.n_experiments:
                self.terminated = 'exp'

    def _terminate_racing(self):
        """
        terminate racing
        """
        self._clean_queue()
        self._result_future.set_result("Future is done!")

    def get_elite_configurations(self, all: bool = False):
        if all:
            return self.configurations.get_alive(as_list=True)
        else:
            return self.configurations.get_configurations(self.elite_ids, as_list=True)

    def update_status_for_recovering(self):
        """
        recovering scheduled_jobs when recovery folder is not None
        """
        if self.options.onlytest.value:
            print("\n# Load from ONLYTEST results")
            return CraceResults(
                 options=self.options,
                 parameters=self.parameters,
                 instances=self.instances,
                 configurations=self.configurations,
                 models=self.model,
                 experiments=self.experiments,
                 elites=self.elite_ids,
                 best_id=self.best_id,
                 state=None,
                 race=self,
            )

        recovery_flag = True if self.options.recoveryDir.value else False
        readlogs_flag = True if self.options.readlogs.value else False

        # update counters for exp and time
        self.n_experiments = self.experiments.n_complete
        self.n_time = self.experiments.total_time

        # update budget and counter for updating models
        self.used_budget_recoverd = self.used_budget = self.n_time if self.options.maxTime.value > 0 else self.n_experiments

        # check termination
        if recovery_flag and self.used_budget_recoverd >= self.budget:
            self.terminated = 'budget'
            print("\n\n# BUDGET IS EXHAUSTED!")
            print("# Nothing to restore.\n")

        # check slice index
        if recovery_flag:
            slice_log = self.options.recoveryDir.value + "/slice.log"
        else:
            slice_log = self.options.readlogs.value + "/slice.log"

        slice = ModelInfo.load_from_log(slice_log)

        self.idx_updating = 0 if slice.empty else max(slice.iloc[:, 0].tolist())

        # update elite configurations
        RANDOM_FLAG = False
        self.experiments.update_status_for_recovering(alive=self.configurations.get_alive_ids())
        if recovery_flag:
            self.recovery_elite = self.options.recoveryDir.value + "/elite.log"
        else:
            self.recovery_elite = self.options.readlogs.value + "/elite.log"

        if os.path.exists(self.recovery_elite):
            with open(self.recovery_elite, newline='') as f:
                reader = csv.reader(f)
                data = [tuple(row) for row in reader]
                data = data[1:]
                for i in data:
                    self.elite_ids.append(int(i[-2]))

        else:
            RANDOM_FLAG = True
            ncompleted_configs = self.experiments.get_ncompleted_by_configurations(self.configurations.get_alive_ids())
            self.elite_ids = [[k for k,v in sorted(ncompleted_configs.items(), key=lambda x: x[1], reverse=True)][0]]

        self.old_elites[1:] = self.elitist_ids = self.elite_ids
        self.best_id = self.elite_ids[0]
        if self.experiments.get_ncompleted_by_configuration(self.best_id) > 0:
            self.mean_best = float(np.array(list(self.experiments.get_result_by_configuration(self.best_id, True).values())).mean())
        self.elite_list = {x: self.experiments.get_ncompleted_by_configuration(x) for x in self.elite_ids}

        # update scheduled experiments
        self._scheduled_configs, self._scheduled_jobs = self.experiments.get_scheduled_experiments()
        if recovery_flag:
            self._pool.submit_experiments([
                self.experiments.get_experiment_by_id(x, as_dataframe=False) for x in self._scheduled_jobs], self.elite_ids)

        # update status for configurations
        self.configurations.update_status_for_recovering(scheduled=self._scheduled_configs,
                                                         model_config=self.model.alive_model_config)

        # update status for experiments
        self.experiments.update_status_for_recovering(elites=self.elite_ids)

        # if capping: update time bound for target algorithm on each used instance
        if self.capping:
            self.priori_bounds = self.experiments.calculate_priori_bounds(
                instance_ids=self.instances.instances_ids,
                elitist_ids=self.elitist_ids,
                bound_max=self.priori_bounds[0],
                domType=self.options.domType.value,
                recovery=True,
                readlogs=readlogs_flag)

        # update submitted experiments
        # based on the information of configurations
        exps_by_configs = self.experiments.experiments_by_configuration.keys()
        alive_ids = self.configurations.get_alive_ids()
        new_configs = []
        for x in alive_ids:
            if x not in exps_by_configs:
                new_configs.append(self.configurations.get_configuration(x))
        if len(new_configs) > 0:
            self._add_experiments(configurations=new_configs,
                                  instances=self.instances.instances,
                                  bounds=self.priori_bounds,
                                  challenges=self.elite_ids)

        self.n_alive = len(self.configurations.get_alive_ids())

        state = (
            f"#   used budget:                                  {self.used_budget}\n"
            f"#   finished experiments:                         {self.n_experiments}\n"
            f"#   number of alive configurations:               {self.n_alive}\n"
            f"#   number of scheduled experiments:              {len(self._scheduled_jobs)}\n"
        )
        if not RANDOM_FLAG:
            state += f"#   elite configurations:                         {str(self.elite_ids).strip('[').strip(']')}\n"
        else:
            state += f"#   elite configurations(select from alive):      {str(self.elite_ids).strip('[').strip(']')}\n"
        state += f"#   number of instances in stream:                {len(self.instances.instances)}"
        if recovery_flag:
            state += f"\n# Continue crace from slice                       {self.idx_updating+1}"

        if self.options.readLogsInCplot.value < 1:
            print(f"\n# Check data consistency and then restore the state")
            print(state, '\n')

        if not recovery_flag:
            if self.options.readLogsInCplot.value < 1:
                print(f'# Return an object: CraceResults (including: ')
                print(f'#   options, parameters, instances, configurations, experiments, models)\n')
            return CraceResults(
                 options=self.options,
                 parameters=self.parameters,
                 instances=self.instances,
                 configurations=self.configurations,
                 models=self.model,
                 experiments=self.experiments,
                 elites=self.elite_ids,
                 best_id=self.best_id,
                 state=state,
                 slice=slice,
                 race=self,
            )
        else:
            if self.options.instancePatience.value !=0:
                self.exps_view = self.experiments.get_stream_experiments(stream=self.instances.instances_ids)

class RaceCheck:
    """
    Class that implements the checking procedure
    To test target algorithms several times

    """

    def __init__(
        self,
        initial_configurations: Configurations,
        instances: Instances,
        parameters: Parameters,
        options: CraceOptions
    ):
        """
            Creates the structures used in the race. This class should be instantiated form a
            the Crace class object.

            :param initial_configurations: Configurations object that describes a set of initial configurations
            :param instances: Instances object that describes a set of training instances
            :param parameters: Parameters object that describes the parameter space
            :param options: craceOptions object that contains all crace options
        """
        # ###################################################################################################################
        #                                                                                                                   #
        #                                          SCENARIO PARAMETERS / OPTIONS                                            #
        #                                                                                                                   #
        # ###################################################################################################################
        self.options = options
        self.log_level = self.options.logLevel.value
        self.debug_level = self.options.debugLevel.value
        self.capping = self.options.capping.value
        # Minumum number of experiments to be scheduled at any time
        self.num_cores = max(1, self.options.parallel.value)

        self.parameters = parameters
        self.instances = instances

        self.budget = 6 if instances.has_test() else 3

        # Initialize configurations for the race
        if initial_configurations is not None:
            # Recovered configurations should be obtained when reading the scenario
            self.configurations = copy.deepcopy(initial_configurations)
        else:
            self.configurations = Configurations(
                parameters=self.parameters,
                exec_dir=self.options.execDir.value,
                recovery_folder=self.options.recoveryDir.value,
                conf_repair=self.options.repairConfiguration.value,
                debug_level=self.debug_level,
                log_base_name=os.path.basename(options.logDir.value)
            )

        # Initialize log files
        setup_race_loggers(self.options, True)

        # Generate or recover experiments
        self.experiments = Experiments(
            log_folder=self.options.logDir.value,
            recovery_folder=self.options.recoveryDir.value,
            budget_digits=self.options.boundDigits.value,
            capping=self.capping,
            log_level=self.log_level,
            configurations=self.configurations,
        )

        # Initialize race model
        self.model = ProbabilisticModel(
            parameters=parameters,
            log_folder=self.options.logDir.value,
            log_level=self.options.logLevel.value,
            global_model=self.options.globalModel.value)

        # In the case there is already configurations in place
        self.n_alive = len(self.configurations.get_alive_ids())

        # Sample configurations
        self.model.add_models(self.model.size)

        # FIXME: add a method for this
        to_sample = max(0, 3 - self.n_alive)
        # sample configurations based randomly
        if to_sample > 0:
            if initial_configurations is None:
                configurations = self.model.sample_random_parameters(to_sample)
                self.configurations.add_from_model(configurations)
            else:
                configurations = self.model.sample_from_random_parents(to_sample, initial_configurations)
                self.configurations.add_from_model(configurations)

        self.null = 0
        self.correct = 0

        # bounds[0] is the default bound for checking
        self.priori_bounds = {}
        self.priori_bounds[0] = 2 if self.capping else 0

        # General things to be initialized
        self._pool = None

        # self._result_future = asyncio.Future()
        loop = get_loop()
        self._result_future = loop.create_future()


    def set_pool(self, pool):
        """
        Set pool variable of the race
        :param pool: Pool object
        """
        self._pool = pool

    async def asynchronous_race(self):
        """
        Starts an asynchronous race and returns a coroutine that needs to be awaited.
        :return: Configuration object containing the set of alive configurations
        """
        if self.budget == 3:
            print(f"# Testing target executioner {self.budget} times on a random selected training instance..\n#")
        else:
            print(f"# Testing target executioner {self.budget} times..")
            print(f"#  Instance 1: random selected from the training instances")
            print(f"#  Instance 2: random selected from the test instances\n#")
        self.instances.check_instances()
        if self.capping:
            print(f"# Set time bound for each target algorithm as 2 seconds")
            for ins in self.instances.instances:
                self.priori_bounds[ins.instance_id] = self.priori_bounds[0] # bound of new ins is the default value

        # Add experiments
        for x in self.instances.instances:
            self._add_experiments(
                self.configurations.get_alive(),
                [x],
                self.priori_bounds
            )

        # Wait for race to finish
        await self._result_future

        return self.configurations.get_alive()

    def _add_experiments(self, configurations: list, instances: list, bounds: dict):
        """
        Generates and adds new experiments to the race
        :param configurations: List of configurations which should be executed
        :param instances: List of instances that should be executed
        :param bound: Dict of bound for executing the experiments, keys are correspoding instance ids
        """

        if isinstance(configurations, dict):
            configurations = list(configurations.values())

        # Create experiments
        _, exps = self.experiments.new_experiments(configurations, instances, bounds)

        # Submit experiments
        self._pool.submit_experiments(exps)

    def race_callback(self, experiment: ExperimentEntry, solution_quality, solution_time):
        """
        Called when one experiment is finished. At the moment only called in the async_race.

        New race_callback includes the basic components for storing the information of the finished experiment
        then call test or tests based on option testExperimentSize.

        :param experiment: Experiment object which triggers the callback
        :param solution_quality: Result obtained by the experiment
        :param solution_time: Execution time reported by the experiment
        """
        # Discard all experiments from this configuration, if any experiment returns infinite value
        if math.isinf(solution_quality) or math.isnan(solution_quality):
            self.null += 1
        else:
            self.correct += 1

        print(f"# Experiment {experiment.experiment_id}\
        configuration {experiment.configuration_id}\
        instance {experiment.instance_id}\
        result {solution_quality} ")

        self.budget -= 1

        # If the race was terminated we stop here
        if self.budget <= 0:
            self._result_future.set_result("Future is done!")
            return
