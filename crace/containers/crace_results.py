import os
import sys
import itertools
import numpy as np
import pandas as pd
import scipy.stats as ss

from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests

import crace.settings.description as csd
from crace.containers.instances import Instances
from crace.models.model import ProbabilisticModel
from crace.containers.parameters import Parameters
from crace.containers.experiments import Experiments
from crace.containers.crace_options import CraceOptions
from crace.containers.configurations import Configurations


@staticmethod
def _print_error(info: str):
    print("\nERROR: There was an error while analysing results:")
    print(info)
    sys.tracebacklimit = 0
    raise

_unblocked_properties = ["test", "version"]
def _conditional_property(unblocked_list, flag_attr):
    """
    check if property is in unblocked_list
    if not and flag_attr is not enabled: raise
    """
    def decorator(func):
        prop_name = func.__name__

        def wrapper(self):
            enabled = getattr(self, flag_attr)

            if prop_name not in unblocked_list and not enabled:
                print(f"{prop_name} is disabled for results from ONLYTEST.")
                return
            else:
                return func(self)

        return property(wrapper)
    return decorator


class ExperimentsData:
    def __init__(self, experiments: Experiments,
                 test: bool, 
                 elites: list,
                 all_elites: list=None,
                 slice: pd.DataFrame=None,
                 ):
        """
        initialize experiment data

        :var test: boolean for test experiments
        :var elites: list of elite configurations finally returned
        :var all_elites: list of elite configurations selected during crace procedure
        :var slice: dataframe of slice information
        """
        self.experiments = experiments
        self._test = test
        if not self._test:
            self.slice = slice
        self.all_elites = all_elites
        self.elites = elites

        self.data = experiments.get_all_experiments()

        if self.all_elites:
            ncompleted = experiments.get_ncompleted_by_configurations(self.all_elites)
        else:
            ncompleted = experiments.get_ncompleted_by_configurations(self.elites)
        # select elite configurations having the most evaluations
        self._selected = [int(k) for k, v in ncompleted.items() if v == max(ncompleted.values())]

    def pairwise_test(self, test_name: str='wilcoxon'):
        try:
            self._pairwise_test(test_name)
        except Exception:
            pass

    def _pairwise_test(self, test_name: str):
        """
        pariwise test for configurations having the same evaluations

        test_name: available test method
        """
        # dict for provided test methods
        #   available test methods: ss.wilcoxon, ss.mannwhitneyu, ss.ttest_rel
        tests = {'wilcoxon': ss.wilcoxon, 'wil': ss.wilcoxon, 'wx': ss.wilcoxon,
                 'mannwhitneyu': ss.mannwhitneyu, 'mw': ss.mannwhitneyu, 'mwu': ss.mannwhitneyu,
                 'ttest': ss.ttest_rel, 'tt': ss.ttest_rel,
                }
        if test_name not in tests.keys():
            _print_error(f"Provided test name ({test_name}) is not allowed,\n you can choose "
                         f"from ({', '.join(tests.keys())})")

        if not self.experiments:
            _print_error(f"No test results.")

        # select experiment data
        data = self.experiments.get_experiments_by_configurations(self.elites, as_dataframe=True)
        data = data[['experiment_id', 'configuration_id', 'instance_id', 'quality']].dropna()

        if len(self._selected) < 2:
            if len(self.elites) < 2:
                _print_error(f'Pairwise test is not allowed for only one elite configuration {self._selected}.')
            else:
                _print_error(f'Pairwise test is not allowed when the provided elites {self.elites} are inconsistent in instances.')

        selected_data = data[data['configuration_id'].isin(self._selected)]

        print(f'Doing pairwise test (selected: {test_name})..')

        print('Elite configurations: ', self.elites)

        instances = selected_data['instance_id'].unique()
        print('The number of selected instances: ', len(instances))

        pairwise_results = []
        p_values = []

        for conf1, conf2 in itertools.combinations(self._selected, 2):
            quality1 = selected_data.loc[selected_data['configuration_id'] == conf1, 'quality'].values
            quality2 = selected_data.loc[selected_data['configuration_id'] == conf2, 'quality'].values
            
            # test
            if len(quality1) == len(quality2):
                stat, p = tests[test_name](quality1, quality2)
                p_values.append(p)
                pairwise_results.append((conf1, conf2, stat, p))

        # adjust p_values using Bonferroni
        adjusted_p_values = multipletests(p_values, alpha=0.05, method='bonferroni')[1]

        # print the output
        for (conf1, conf2, stat, p), adj_p in zip(pairwise_results, adjusted_p_values):
            print(f"Comparison between configuration {conf1} and {conf2}:\n  statistic={stat}, p-value={p}, adjusted p-value={adj_p}")

    def concordance(self, method: str='average'):
        try:
            self._concordance(method)
        except Exception:
            pass

    def _concordance(self, method: str):
        """
        calculate concordance for selected configurations
        """
        # check provided rankdata method
        methods = ['average', 'min', 'max', 'dense', 'ordinal']
        if method not in methods:
            _print_error(f"Provided rank method ({method}) is not allowed,\n you can "
                         f"choose from ({', '.join(methods)})")

        results = self.experiments.get_experiments_by_configurations(self._selected, as_dataframe=True)
        results = results[['configuration_id','quality']].dropna()
        grouped = results.groupby("configuration_id")["quality"].apply(list)

        print(f'Checking concordance..')

        data = pd.DataFrame.from_records(grouped.tolist(), index=grouped.index)
        data.dropna(axis=0, inplace=True)
        data_values = data.values

        if not isinstance(data_values, np.ndarray) or not np.issubdtype(data_values.dtype, np.number):
            _print_error("Input must be a numeric matrix.")

        n, k = data_values.shape
        if n <= 1 or k <= 1:
            print("Only the best configuration was tested.")
            return

        r = np.apply_along_axis(rankdata, 1, data_values, method=method)
        R = np.sum(r, axis=0)
        TIES = np.bincount(r.astype(int).flatten())
        
        if all(TIES == k):
            W = 1
        else:
            TIES = np.sum(TIES**3 - TIES)
            W = (12 * np.sum((R - n * (k + 1) / 2)**2)) / ((n**2 * (k**3 - k)) - (n * TIES))

        rho = (n * W - 1) / (n - 1)

        print(f"selected configurations: {', '.join(map(str,data.index.to_list()))}")
        print(f"rank method: {method}\t kendall.w: {W:.5f}\t spearman.rho: {rho:.5f}")


class CraceResults:
    def __init__(self, 
                 options: CraceOptions,
                 parameters: Parameters,
                 instances: Instances,
                 configurations: Configurations,
                 models: ProbabilisticModel,
                 experiments: Experiments,
                 elites: list,
                 best_id: int,
                 state: str,
                 slice,
                 race):
        """
        initialize crace results

        arguments prefixed with '_': load when invocation
        """

        self.options = options
        self.parameters = parameters
        self.instances = instances
        self.configurations = configurations

        self._models = models
        self._slice = slice

        self._enabled = True if not options.onlytest.value else False

        self._experiments = experiments

        self._elites = elites
        self._best_id = best_id
        self._race = race

        self._scenario = None
        self._state = None
        self._all_elites = None
        self._training = None
        self._test = None
        self._summarise = None
        self._version = csd.version

        self.__state = state


    @_conditional_property(unblocked_list=_unblocked_properties, flag_attr="_enabled")
    def models(self):
        return self._models

    @_conditional_property(unblocked_list=_unblocked_properties, flag_attr="_enabled")
    def slice(self):
        return self._slice

    @_conditional_property(unblocked_list=_unblocked_properties, flag_attr="_enabled")
    def elites(self):
        return self._elites

    @_conditional_property(unblocked_list=_unblocked_properties, flag_attr="_enabled")
    def best_id(self):
        return self._best_id

    @_conditional_property(unblocked_list=_unblocked_properties, flag_attr="_enabled")
    def version(self):
        return self._version

    @_conditional_property(unblocked_list=_unblocked_properties, flag_attr="_enabled")
    def scenario(self):
        if self._scenario is None:
            self._scenario = self.options.print_all()
        else:
            return self._scenario

    @_conditional_property(unblocked_list=_unblocked_properties, flag_attr="_enabled")
    def state(self):
        if self._state is None:
            self._state = self.__state.replace('#   ', '')
        print(self._state)

    @_conditional_property(unblocked_list=_unblocked_properties, flag_attr="_enabled")
    def all_elites(self):
        """
        load all elite configurations selected in each slice
        """
        if self._all_elites is None:
            slice = self.slice
            if 'elites' not in slice.columns:
                print("# Note: 'slice' object has no attribute 'elites'")
                elites_log = os.path.join(os.path.dirname(self.options.readlogs.value), 'elites.log')
                if os.path.exists(elites_log):
                    print("#   loading elites from 'elites.log'...")
                    with open(elites_log, newline='') as f:
                        lines = f.read().splitlines()
                    elites = list(int(x) for x in lines[:-1])
                else:
                    return None
            else:
                if self.options.elitist.value:
                    elites = list(dict.fromkeys(
                                                x[0] for x in slice.elites.tolist() + [self.elites]
                                                if isinstance(x, list) and x)
                    )
                else:
                    elites = list(dict.fromkeys(
                                                x for sub in slice.elites.tolist() + [self.elites]
                                                if isinstance(sub, list) and sub
                                                for x in sub))

            self._all_elites = elites
        return self._all_elites

    @_conditional_property(unblocked_list=_unblocked_properties, flag_attr="_enabled")
    def training(self):
        if self._training is None:
            self._training = TrainResults(experiments=self._experiments,
                                          slice=self.slice,
                                          all_elites=self.all_elites,
                                          elites=self.elites,
                                          options=self.options)
        return self._training

    @_conditional_property(unblocked_list=_unblocked_properties, flag_attr="_enabled")
    def test(self):
        if self._test is None:
            self._test = TestResults(options=self.options,
                                     configurations=self.configurations,
                                     elites=self.elites)
        return self._test
    
    @_conditional_property(unblocked_list=_unblocked_properties, flag_attr="_enabled")
    def summarise(self):
        """
        obtain brief racing information
        """
        if self._summarise is None:
            from crace.race.race import Race
            race = self._race
            experiments = self._experiments
            configurations = self.configurations
            options = self.options

            class Summary:
                def __init__(self, race: Race,
                            options: CraceOptions,
                            experiments: Experiments,
                            configurations: Configurations):
                    # crace
                    self.version = csd.version
                    self.author = csd.author
                    self.maintainers = csd.maintainers

                    # scenario
                    self.budget = race.budget
                    self.n_slice = options.modelUpdateByStep.value + 1
                    self.n_pre_budget = race.n_pre_budget

                    # budget and time
                    self.used_budget = race.used_budget
                    self.used_time = race.n_time
                    self.finished_experiments = race.n_experiments
                    self.total_experiment_time = experiments.total_time

                    # configuration and instance
                    self.n_alive_configurations = race.n_alive
                    self.total_instances = experiments.get_n_instances()
                    self.total_configurations = configurations.n_config
                    self.n_discarded_configurations = self.total_configurations - self.n_alive_configurations

                    # results
                    self.best_id = race.best_id
                    self.best_mean = float(experiments.get_experiments_by_configuration(self.best_id).dropna()['quality'].mean())

                    # all elements
                    self.all = self._print_all

                def _print_all(self):
                    lenth = max(len(x) for x in self.__dict__.keys())
                    for name, value in self.__dict__.items():
                        if name not in ['all', '_print_all']:
                            if isinstance(value, list):
                                value = ', '.join(value)
                            print(f'{name:<{lenth}}: {value}')

            self._summarise = Summary(race, options, experiments, configurations)
        return self._summarise

class TrainResults(ExperimentsData):
    """
    load training experiments
    """
    def __init__(self, experiments, slice, all_elites, elites, options: CraceOptions):
        super().__init__(experiments=experiments, slice=slice,
                         test=False, all_elites=all_elites, elites=elites)

class TestResults(ExperimentsData):
    def __init__(self, options: CraceOptions, configurations: Configurations, elites: list):
        """
        load test experiments
        """
        self._options = options
        self._configurations = configurations

        experiments = self._load_results()
        final_elites = list(map(int, experiments.get_all_experiments()['configuration_id'].unique()))

        super().__init__(experiments=experiments,
                         test=True,
                         elites=final_elites,
                         all_elites=elites)

    def _load_results(self):
        if self._options.readlogs.short in self._options.arguments:
            index = self._options.arguments.index(self._options.readlogs.short)
        else:
            index = self._options.arguments.index(self._options.readlogs.long)
        log_base_name=os.path.basename(self._options.logDir.value)
        test_dir = os.path.abspath(self._options.arguments[index+1]) + '/' + log_base_name + '/test'
        test_fin = test_dir + '/exps_fin.log'

        if not os.path.exists(test_dir) or os.path.getsize(test_fin) == 0:
            print(f"#   No test results.")
            return None

        test = Experiments(log_folder=test_dir,
                           test_folder=test_dir,
                           budget_digits=self._options.boundDigits.value,
                           capping=self._options.capping.value,
                           log_level=self._options.logLevel.value,
                           configurations=self._configurations)

        f_exp = test._read_log_data(test_fin)
        for i in f_exp.keys():
            if (f_exp is not None) and (i in f_exp.keys()):
                exp = f_exp[i]
                test._add_experiment(exp)

        return test
