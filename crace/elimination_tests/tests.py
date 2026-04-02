import math
import scipy
import logging
import statistics
import collections
import pandas as pd

from scipy import stats
from scipy.stats import chi2, wilcoxon
from statutils.multi_comparison import p_adjust

from crace.utils.const import DELTA
from crace.errors import CraceExecutionError
from crace.elimination_tests.eliminator import Eliminator, EliminationTestResults


class FriedmanTest(Eliminator):
    """
    Friedman test statistical test
    """
    def __init__(self, correction=None, paired=False, confidence_level=0.95):
        """
        Creates an Eliminator object
        :param correction: correction type name
        :param paired: Boolean that indicates if test is paired
        :param confidence_level: Real, confidence level
        """
        # super().__init__("F-test", correction, paired, confidence_level)
        super().__init__(
            test_name="F-test",
            dom_name=None,
            correction=correction,
            paired=paired,
            confidence_level=confidence_level)

    def test_elimination(self,
                         experiment_results: pd.DataFrame,
                         instances: list,
                         ins_stream: list,
                         min_test: int,
                         elite_candidates: list=[],
                         n_best=None) -> EliminationTestResults:
        """
        Performs a statistical test elimination over a set of experiments
        :param experiment_results: results data frame (configurations in columns)
        :param instances: instances to be considered in the test
        :param min_test: minimum number of instances a configuration should be executed on to be elegible for elimination
        :param elite_candidates: list of elite configurations ids
        :param n_best: number of elite configurations that should have better performance than an eliminated configuration

        :return: EliminationTestResults object with the test results
        """
        full_experiment_results = experiment_results

        # get steam results
        experiment_results = full_experiment_results.loc[ins_stream, :]

        # Select experiments by instances and get a complete matrix
        selected_results = experiment_results.loc[instances, :]
        selected_results = selected_results.dropna(axis=1, how="any")
        results = selected_results.dropna()

        if not self.can_test(results):
            return self.only_ranks(selected_results)

        nc = len(results.columns.values)
        if nc == 2:
            test_results = self._particular_case(results)
        else:
            test_results = self._general_case(results)

        alive_ids, discarded_ids = self.adjust_elimination(
            results=experiment_results,
            test_result=test_results,
            min_test=min_test,
            each_test=1,
        )

        rank_results = self.only_ranks(experiment_results)

        # Calculate final ranks
        best, final_ranks = self.final_ranks(experiment_results, rank_results, alive_ids)

        return EliminationTestResults(best=test_results.best,
                                      means=test_results.means,
                                      ranks=test_results.ranks,
                                      p_value=test_results.p_value,
                                      alive=alive_ids,
                                      discarded=discarded_ids,
                                      instances=test_results.instances,
                                      data=test_results.data,
                                      overall_best=best,
                                      overall_mean=rank_results.overall_mean,
                                      overall_ranks=final_ranks,
                                      overall_data=rank_results.overall_data)

    def only_ranks(self, experiment_results: pd.DataFrame):
        """
        Calculate only the ranks of the configurations based on a set of experiments

        :param experiment_results: results data frame (configurations in columns)

        :return: EliminationTestResults object with the test results
        """
        configurations = experiment_results.columns.values
        ranked_data = experiment_results.rank(axis=1, ascending=True)
        rank_sums = ranked_data.sum(axis=0)
        best = rank_sums.idxmin()
        means = experiment_results.mean(axis=0)
        ranks = rank_sums.to_dict()

        return EliminationTestResults(best=best,
                                      means=means,
                                      ranks=ranks,
                                      p_value=None,
                                      alive=configurations,
                                      discarded=[],
                                      instances=list(experiment_results.index),
                                      data=experiment_results,
                                      overall_best=best,
                                      overall_mean=means,
                                      overall_ranks=ranks,
                                      overall_data=experiment_results)

    def _particular_case(self, experiment_results):
        """
        Performs a statistical test elimination of two configurations

        :param experiment_results: results data frame (configurations in columns)

        :return: EliminationTestResults object with the test results
        """
        # TODO: improve comments
        conf = experiment_results.columns.values
        p_val = 0
        selected = conf.tolist()
        v1 = experiment_results[conf[0]]
        v2 = experiment_results[conf[1]]

        if all(v1 <= v2):
            ranks = {conf[0]: 1, conf[1]: 2}
            best = conf[0]
            selected = [conf[0]]

        elif all(v2 <= v1):
            ranks = {conf[0]: 2, conf[1]: 1}
            best = conf[1]
            selected = [conf[1]]

        else:
            p_val = wilcoxon(v1, v2, correction=True)[1]
            if p_val < 1 - self.confidence_level:
                # we can eliminate a configuration
                if (statistics.median(v1) - statistics.median(v2)) < 0:
                    best = conf[0]
                    ranks = {conf[0]: 1, conf[1]: 2}
                    selected = [conf[0]]

                else:
                    best = conf[1]
                    ranks = {conf[0]: 2, conf[1]: 1}
                    selected = [conf[1]]

            else:
                # we cannot eliminate a configuration
                if (statistics.median(v1) - statistics.median(v2)) < 0:
                    best = conf[0]
                    ranks = {conf[0]: 1, conf[1]: 2}

                else:
                    best = conf[1]
                    ranks = {conf[0]: 2, conf[1]: 1}

        discarded = set(conf) - set(selected)
        means = experiment_results.mean(axis=0)

        return EliminationTestResults(best=best,
                                      means=means,
                                      ranks=ranks,
                                      p_value=p_val,
                                      alive=selected,
                                      discarded=list(discarded),
                                      instances=list(experiment_results.index),
                                      data=experiment_results,
                                      overall_best=None,
                                      overall_mean=None,
                                      overall_ranks=None,
                                      overall_data=None)

    def _general_case(self, experiment_results):
        """
        Performs a statistical test elimination of N > 2 configurations

        :param experiment_results: results data frame (configurations in columns)

        :return: EliminationTestResults object with the test results
        """
        conf = experiment_results.columns.values
        assert len(set(conf)) == len(conf), "Repeated configurations in test"  
      
        nc = len(experiment_results.columns) #configurations
        ni = len(experiment_results) #instances

        ranked_data = experiment_results.rank(axis=1, ascending=True)
        rank_sums = ranked_data.sum(axis=0)
        best = rank_sums.idxmin()
        order = rank_sums.sort_values(axis=0, ascending=True).index.values

        ties = ranked_data.apply(lambda row: collections.Counter(row), axis=1)
        ties_trans = {instance: {rank: pow(count, 3)-count for rank, count in ties[instance].items()} for instance in ties.keys()}
        ties_sum = sum([sum([r for r in t.values()]) for t in ties_trans.values()]) / (nc-1)

        try:
            statistic = (12 * sum(pow(rank_sums - ni * (nc+1) / 2, 2))) / (ni * nc * (nc+1) - ties_sum)
        except ZeroDivisionError:
            raise CraceExecutionError("# ZeroDivisionError: {}".format(experiment_results))

        p_val = 1 - chi2.cdf(statistic, nc)
        alpha = 1 - self.confidence_level

        selected = conf.tolist()
        if p_val < alpha:
            sum_sqrt = pow(ranked_data, 2).sum().sum()
            te = scipy.stats.t.ppf(1-(alpha/2), ((ni-1)*(nc-1))) * math.sqrt(2*((((ni* sum_sqrt)-( pow(rank_sums,2).sum())))/((ni-1)*(nc-1))) )
            alive = {x: False for x in conf}
            selected = [best]
            assert len(alive) == len(order), "Alive must be same length as order"

            for i in range(1, nc):
                if abs(rank_sums[order[i]] - rank_sums[order[0]]) > te:
                    break
                else:
                    alive[order[i]] = True
                    if order[i] not in selected:
                        selected.append(order[i])

        discarded = list(set(conf) - set(selected))
        means = experiment_results.mean(axis=0)

        assert len(set(selected)) == len(selected), "Repeated alive configurations after statistical test"
        assert len(set(discarded)) == len(discarded), "Repeated discarded configurations after statistical test"

        return EliminationTestResults(best=best,
                                      means=means,
                                      ranks=rank_sums.to_dict(),
                                      p_value=p_val,
                                      alive=selected,
                                      discarded=discarded,
                                      instances=list(experiment_results.index),
                                      data=experiment_results,
                                      overall_best=None,
                                      overall_mean=None,
                                      overall_ranks=None,
                                      overall_data=None)

    def can_test(self, experiment_results):
        """
        Checks whether it is possible to perform a test based on the experiments provided

        :param experiment_results: results data frame (configurations in columns)

        :return: Boolean indicating if its possible to test
        """
        if len(experiment_results.index) <= 2:
            return False
        if len(experiment_results.columns.values) < 2:
            return False
        return True


class Ttest(Eliminator):
    """
    T-test statistical test
    """
    def __init__(self, correction="none", paired=False, confidence_level=0.95, first_test=1):
        """
        Creates an Eliminator object

        :param correction: correction type name
        :param paired: Boolean that indicates if test is paired
        :param confidence_level: Real, confidence level
        """
        # super().__init__("t-test", correction, paired, confidence_level)
        super().__init__(
            test_name="t-test",
            dom_name=None,
            correction=correction,
            paired=paired,
            confidence_level=confidence_level)
        self.correction = correction
        self.first_test = first_test

    def test_elimination(self,
                         experiment_results: pd.DataFrame,
                         instances: list,
                         ins_stream: list,
                         min_test: int,
                         elite_candidates: list=[],
                         n_best=None) -> EliminationTestResults:
        """
        Performs a statistical test elimination over a set of experiments

        :param experiment_results: results data frame (configurations in columns)
        :param instances: instances to be considered in the test
        :param min_test: minimum number of instances a configuration should be executed on to be elegible for elimination
        :param elite_candidates: list of elite configurations ids
        :param n_best: number of elite configurations that should have better performance than an eliminated configuration

        :return: EliminationTestResults object with the test results
        """
        full_experiment_results = experiment_results

        # get steam results
        experiment_results = full_experiment_results.loc[ins_stream, :]

        # Select experiments by instances and get a complete matrix
        selected_results = experiment_results.loc[instances, :]
        selected_results = selected_results.dropna(axis=1, how="any")
        results = selected_results.dropna()

        if len(results.columns) <= 1:
            return self.only_ranks(selected_results)

        if not self.can_test(results):
            return self.only_ranks(results)

        conf = results.columns.values
        nc = len(results.columns)

        mean_list = results.mean(axis=0)
        best = mean_list.idxmin()
        b_res = results[best]
        mean_best = b_res.mean()

        test_pvals = list()
        discarded = list()
        alive = list()
        for x in conf:
            c_res = results[x]
            if mean_best != c_res.mean():
                if self.paired:
                    test_pvals.append(stats.ttest_rel(b_res, c_res)[1])
                else:
                    test_pvals.append(stats.ttest_ind(b_res, c_res, equal_var=False)[1])
            else:
                test_pvals.append(1)

        p_adjusted = test_pvals
        if not all([x == 1 for x in test_pvals]):
            p_adjusted = p_adjust(test_pvals, method=self.correction).tolist()
        test_pvals = p_adjusted

        for i in range(len(test_pvals)):
            if test_pvals[i] < 1-self.confidence_level:
                discarded.append(conf[i])
            else:
                alive.append(conf[i])

        rank = mean_list.rank()

        test_results = EliminationTestResults(best=best,
                                              means=mean_list.to_dict(),
                                              ranks=rank,
                                              p_value=test_pvals,
                                              alive=alive,
                                              discarded=discarded,
                                              instances=instances,
                                              data=results,
                                              overall_best=None,
                                              overall_mean=None,
                                              overall_ranks=None,
                                              overall_data=None)

        alive_ids, discarded_ids = self.adjust_elimination(
            results=experiment_results,
            test_result=test_results,
            min_test=min_test,
            each_test=1,
        )

        rank_results = self.only_ranks(experiment_results)
        # Calculate final ranks
        best, final_ranks = self.final_ranks(experiment_results, rank_results, alive_ids)

        return EliminationTestResults(best=test_results.best,
                                      means=test_results.means,
                                      ranks=test_results.ranks,
                                      p_value=test_results.p_value,
                                      alive=alive_ids,
                                      discarded=discarded_ids,
                                      instances=test_results.instances,
                                      data=test_results.data,
                                      overall_best=best,
                                      overall_mean=rank_results.overall_mean,
                                      overall_ranks=final_ranks,
                                      overall_data=rank_results.overall_data)

    def only_ranks(self, experiment_results):
        """
        Calculate only the ranks of the configurations based on a set of experiments

        :param experiment_results: results data frame (configurations in columns)

        :return: EliminationTestResults object with the test results
        """
        configurations = experiment_results.columns.values
        mean_data = experiment_results.mean(axis=0)
        rank = mean_data.rank()
        best = rank.idxmin()
        ranks = rank.to_dict()
        return EliminationTestResults(best=best,
                                      means=mean_data,
                                      ranks=ranks,
                                      p_value=None,
                                      alive=configurations,
                                      discarded=[],
                                      instances=list(experiment_results.index),
                                      data=experiment_results,
                                      overall_best=best,
                                      overall_mean=mean_data,
                                      overall_ranks=ranks,
                                      overall_data=experiment_results)

    def can_test(self, experiment_results):
        """
        Checks whether it is possible to perform a test based on the experiments provided

        :param experiment_results: results data frame (configurations in columns)

        :return: Boolean indicating if its possible to test
        """
        if len(experiment_results.index) <= 2:
            return False
        return True


class AdaptiveDomTest(Eliminator):
    """
    Elimination process applied when using dominance
    """
    def __init__(self, correction=None, paired=False, confidence_level=0.95, first_test=1):
        """
        Creates an Eliminator object

        :param correction: correction type name
        :param paired: Boolean that indicates if test is paired
        :param confidence_level: Real, confidence level
        """
        # super().__init__("adaptive-dominance", correction, paired, confidence_level)
        super().__init__(
            test_name=None,
            dom_name="adaptive-dom",
            correction=correction,
            paired=paired,
            confidence_level=confidence_level)
        self.first_test = first_test
        self.old_best = []  # to storage the config id and its ins_num when this config becomes a local best

    def test_elimination(self,
                         experiment_results: pd.DataFrame,
                         instances: list,
                         ins_stream: list,
                         min_test: int,
                         elite_candidates: list=[],
                         n_best: int = 1) -> EliminationTestResults:
        """
        Performs a mean then statistical test elimination over a set of experiments

        :param experiment_results: results data frame (configurations in columns)
        :param instances: instances to consider in the test
        :param min_test: minimum number of instances a configuration should be executed on to be elegible for elimination
        :param elite_candidates: list of elite configurations ids
        :param n_best: number of elite configurations that should have better performance than an eliminated configuration

        :return: EliminationTestResults object with the test results
        """
        full_experiment_results = experiment_results

        # get steam results
        experiment_results = full_experiment_results.loc[ins_stream, :]

        # get all configuration ids
        all_conf = experiment_results.columns.values

        # get experiments of the instances considered in the test
        results = experiment_results.loc[instances, :]

        # drop the configurations that do have considered instances executed
        results = results.dropna(axis=1, how="any")
        sel_conf = results.columns.values

        if len(sel_conf) <= 1:
            #CAN THIS HAPPEN?
            print("# CHECK CODE IN DOMTEST")

        # get the number of overall experiments by configuration
        # only the selected configurations
        not_na = experiment_results.count()
        sel_not_na = experiment_results[sel_conf].count()

        mean_results = results.mean()

        # get best configuration according to considered experiments
        max_ids = sel_not_na.loc[sel_not_na == sel_not_na.max()].index
        # get the mean of the most executed configurations
        best_mean_res = mean_results[max_ids]
        # set flag for time out
        # True: all selected experiments are time out, no configuration perfoms better
        max_min = best_mean_res.min() == best_mean_res.max()

        # get best mean of these configurations
        best_id = best_mean_res.loc[best_mean_res == best_mean_res.min()].index.to_list()

        # in case there is more than one best id, choose the lastly generated
        # TODO: do we really want that to select the lastly generated?
        if len(best_id) > 1:
            best_id = max(best_id)
        else:
            best_id = best_id[0]

        # mean of the elite configurations as reference
        # ref_bounds = test_results[elite_candidates].mean() + DELTA
        ref_bounds = results[best_id].mean() + DELTA

        # count how many reference means are best than each configuration
        ref_best_count = {x: 0 for x in sel_conf}
        for conf_id in sel_conf:
            ref_best_count[conf_id] = (ref_bounds < mean_results[conf_id]).sum()

        # Apply domination criteria
        dom_alive = [x for x, v in ref_best_count.items() if v < n_best]

        # check if any old best config not in alive list
        if len(self.old_best) > 0:
            self.old_best = [x for x in self.old_best if x[0] in all_conf]
        # best_id is a new local best
        if len(self.old_best) < 1:
            self.old_best.append([best_id, not_na[best_id]])

        # if current experiments stream are all time out, 
        # DONOT update old best.
        if not max_min:
            if len(self.old_best) >= 1 and best_id not in [x[0] for x in self.old_best]:
                # the minimal Nins used to do the elimination is one more than the oldest config
                self.old_best.append([best_id, self.old_best[-1][1] + 1])

            if len(self.old_best) > 1:
                i = 0
                while i < len(self.old_best) - 1:
                    if (self.old_best[i][1] < not_na[self.old_best[i][0]]
                        and self.old_best[i+1][1] <= not_na[self.old_best[i+1][0]]):
                        # the oldest: finished ins is more than the beginning
                        # the later: finished ins meets the required minimal number
                        if best_id != self.old_best[i][0]:
                            del self.old_best[i]    # remove
                        else:
                            self.old_best[i][1] += 1
                            del self.old_best[i+1]
                    else:
                        i += 1

        # FIXME: add the info of olde_best in Dominance Test to the race_log file
        logger_1 = logging.getLogger("race_log")
        if len(self.old_best) > 1: logger_1.info("# Best configurations in Adaptive Dominance Test: {}".format(self.old_best))

        # Do not allow the elimination of configurations with less than first_test instances
        pre_alive = set(dom_alive)
        protected = set(experiment_results.columns[not_na < self.first_test]) | set([x[0] for x in self.old_best])
        if any(protected):
            alive = pre_alive.union(protected)

        # Obtain final alive configurations
        discarded = list(set(sel_conf) - alive)
        alive = list(set(all_conf) - set(discarded))

        # get configuration ranks
        rr = self.only_ranks(
            full_experiments=full_experiment_results,
            stream_experiments=experiment_results,
            max_min=max_min)
        results_rank = rr.ranks

        test_results = EliminationTestResults(best=best_id,
                                              means=mean_results,
                                              ranks=results_rank,
                                              p_value=None,
                                              alive=alive,
                                              discarded=discarded,
                                              instances=instances,
                                              data=results,
                                              overall_best=best_id,
                                              overall_mean=experiment_results.mean(),
                                              overall_ranks=results_rank,
                                              overall_data=experiment_results)

        alive_ids, discarded_ids = self.adjust_elimination(
            results=experiment_results,
            test_result=test_results,
            min_test=min_test,
            each_test=1,
        )

        best, final_ranks = self.final_ranks(experiment_results, test_results, alive_ids)

        return EliminationTestResults(best=best_id,
                                      means=mean_results,
                                      ranks=results_rank,
                                      p_value=None,
                                      alive=alive_ids,
                                      discarded=discarded_ids,
                                      instances=instances,
                                      data=results,
                                      overall_best=best,
                                      overall_mean=rr.overall_mean,
                                      overall_ranks=final_ranks,
                                      overall_data=rr.overall_data)

    def only_ranks(self, full_experiments, stream_experiments, max_min:bool=False):
        configurations = stream_experiments.columns.values
        mean_data = stream_experiments.mean(axis=0)
        rank = mean_data.rank()
        best = rank.idxmin()
        ranks = rank.to_dict()

        # update ranks when all configurations are time out
        if max_min:
            import numpy as np
            best = self.old_best[0][0]
            for k, v in ranks.items():
                if k == best:
                    ranks[k] = 1.0
                elif not pd.isna(v):
                    ranks[k] = np.inf

        return EliminationTestResults(best=best,
                                      means=mean_data,
                                      ranks=ranks,
                                      p_value=None,
                                      alive=configurations,
                                      discarded=[],
                                      instances=list(stream_experiments.index),
                                      data=stream_experiments,
                                      overall_best=best,
                                      overall_mean=mean_data,
                                      overall_ranks=ranks,
                                      overall_data=stream_experiments)

    def can_test(self, experiment_results):
        """
        Checks whether it is possible to perform a test based on the experiments provided

        :param experiment_results: results data frame (configurations in columns)

        :return: Boolean indicating if its possible to test
        """
        # experiment_results.index should be one
        if len(experiment_results.index) < 1:
            return False
        return True


class DomTest(Eliminator):
    """
    Elimination process applied when using dominance
    """
    def __init__(self, correction=None, paired=False, confidence_level=0.95, first_test=1):
        """
        Creates an Eliminator object

        :param correction: correction type name
        :param paired: Boolean that indicates if test is paired
        :param confidence_level: Real, confidence level
        """
        # super().__init__("Dominance", correction, paired, confidence_level)
        super().__init__(
            test_name=None,
            dom_name="Dominance",
            correction=correction,
            paired=paired,
            confidence_level=confidence_level)
        self.first_test = first_test
        self.old_best = []  # to storage the config id and its ins_num when this config becomes a local best

    def test_elimination(self,
                         experiment_results: pd.DataFrame,
                         instances: list,
                         ins_stream: list,
                         min_test: int,
                         elite_candidates: list=[],
                         n_best: int = 1) -> EliminationTestResults:
        """
        Performs a mean then statistical test elimination over a set of experiments

        :param experiment_results: results data frame (configurations in columns) of all alive configurations
        :param instances: instances to consider in the test
        :param min_test: minimum number of instances a configuration should be executed on to be elegible for elimination
        :param elite_candidates: list of elite configurations ids
        :param n_best: number of elite configurations that should have better performance than an eliminated configuration

        :return: EliminationTestResults object with the test results
        """
        full_experiment_results = experiment_results

        # get steam results
        experiment_results = full_experiment_results.loc[ins_stream, :]

        # get all configuration ids
        all_conf = experiment_results.columns.values

        # get experiments of the instances considered in the test
        results = experiment_results.loc[instances, :]

        # drop the configurations that do have considered instances executed
        results = results.dropna(axis=1, how="any")
        sel_conf = results.columns.values

        if len(sel_conf) <= 1:
            #CAN THIS HAPPEN?
            print("# CHECK CODE IN DOMTEST")

        # get the number of overall experiments by configuration
        # only the selected configurations
        not_na = experiment_results.count()
        sel_not_na = experiment_results[sel_conf].count()

        mean_results = results.mean()
        # sum_results = results.sum()
        mean_all_results = experiment_results.mean()

        # get best configuration according to considered experiments
        max_ids = sel_not_na.loc[sel_not_na == sel_not_na.max()].index
        # get the mean of the most executed configurations
        best_mean_res = mean_all_results[max_ids]
        # set flag for time out
        # True: all selected experiments are time out, no configuration perfoms better
        max_min = best_mean_res.min() == best_mean_res.max()

        # get best mean of these configurations
        best_id = best_mean_res.loc[best_mean_res == best_mean_res.min()].index.to_list()

        # in case there is more than one best id, choose the lastly generated
        # TODO: do we really want that to select the lastly generated?
        if len(best_id) > 1:
            best_id = max(best_id)
        else:
            best_id = best_id[0]

        # mean of the elite configurations as reference
        ref_bounds = mean_all_results[best_id] + DELTA

        # count how many reference means are best than each configuration
        ref_best_count = {x: 0 for x in sel_conf}
        for conf_id in sel_conf:
            ref_best_count[conf_id] = (ref_bounds < mean_results[conf_id]).sum()

        # Apply domination criteria
        dom_alive = [x for x, v in ref_best_count.items() if v < n_best]

        if len(self.old_best) > 0:
            # check if any old best config not in alive 
            self.old_best = [x for x in self.old_best if x[0] in all_conf]
        # best_id is a new local best
        if len(self.old_best) < 1:
            self.old_best.append([best_id, not_na[best_id]])

        # if current experiments stream are all time out, 
        # DONOT update old best.
        if not max_min:
            if len(self.old_best) >= 1 and best_id not in [x[0] for x in self.old_best]:
                # the minimal Nins used to do the elimination is one more than the oldest config
                self.old_best.append([best_id, self.old_best[-1][1] + 1])
            if len(self.old_best) > 1:
                i = 0
                while i < len(self.old_best) - 1:
                    if (self.old_best[i][1] < not_na[self.old_best[i][0]]
                        and self.old_best[i+1][1] <= not_na[self.old_best[i+1][0]]):
                        # the oldest: finished ins is more than the beginning
                        # the later: finished ins meets the required minimal number
                        if best_id != self.old_best[i][0]:
                            del self.old_best[i]    # remove
                        else:
                            self.old_best[i][1] += 1
                            del self.old_best[i+1]
                    else:
                        i += 1

        # FIXME: add the info of olde_best in Dominance Test to the race_log file
        logger_1 = logging.getLogger("race_log")
        if len(self.old_best) > 1: logger_1.info("# Best configurations in Dominance Test: {}".format(self.old_best))

        # Do not allow the elimination of configurations with less than first_test instances
        pre_alive = set(dom_alive)
        protected = set(experiment_results.columns[not_na < self.first_test]) | set([x[0] for x in self.old_best])
        if any(protected):
            alive = pre_alive.union(protected)

        # Obtain final alive configurations
        discarded = list(set(sel_conf) - alive)
        alive = list(set(all_conf) - set(discarded))

        # get configuration ranks
        rr = self.only_ranks(
            full_experiments=full_experiment_results,
            stream_experiments=experiment_results,
            max_min=max_min)
        results_rank = rr.ranks

        test_results = EliminationTestResults(best=best_id,
                                              means=mean_results,
                                              ranks=results_rank,
                                              p_value=None,
                                              alive=alive,
                                              discarded=discarded,
                                              instances=instances,
                                              data=results,
                                              overall_best=best_id,
                                              overall_mean=experiment_results.mean(),
                                              overall_ranks=results_rank,
                                              overall_data=experiment_results)

        alive_ids, discarded_ids = self.adjust_elimination(
            results=experiment_results,
            test_result=test_results,
            min_test=min_test,
            each_test=1,
        )

        best, final_ranks = self.final_ranks(experiment_results, test_results, alive_ids)

        return EliminationTestResults(best=best_id,
                                      means=mean_results,
                                      ranks=results_rank,
                                      p_value=None,
                                      alive=alive_ids,
                                      discarded=discarded_ids,
                                      instances=instances,
                                      data=results,
                                      overall_best=best,
                                      overall_mean=rr.overall_mean,
                                      overall_ranks=final_ranks,
                                      overall_data=rr.overall_data)

    def only_ranks(self, full_experiments, stream_experiments, max_min:bool=False):
        configurations = stream_experiments.columns.values
        mean_data = stream_experiments.mean(axis=0)
        rank = mean_data.rank()
        best = rank.idxmin()
        ranks = rank.to_dict()

        # update ranks when all configurations are time out
        if max_min:
            import numpy as np
            best = self.old_best[0][0]
            for k, v in ranks.items():
                if k == best:
                    ranks[k] = 1.0
                elif not pd.isna(v):
                    ranks[k] = np.inf

        return EliminationTestResults(best=best,
                                      means=mean_data,
                                      ranks=ranks,
                                      p_value=None,
                                      alive=configurations,
                                      discarded=[],
                                      instances=list(stream_experiments.index),
                                      data=stream_experiments,
                                      overall_best=best,
                                      overall_mean=mean_data,
                                      overall_ranks=ranks,
                                      overall_data=stream_experiments)

    def can_test(self, experiment_results):
        """
        Checks whether it is possible to perform a test based on the experiments provided

        :param experiment_results: results data frame (configurations in columns)

        :return: Boolean indicating if its possible to test
        """
        # experiment_results.index should be one
        if len(experiment_results.index) < 1:
            return False
        return True


class UniteTest(Eliminator):
    """
    Elimination process applied when using capping (it includes the implementation of t-test)
    :ivar ctest: complementary test
    :ivar first_test: number of instances needed to perform a test
    """

    def __init__(self, correction, paired=False, confidence_level=0.95, first_test=1, test_type="none", dom_type="none"):
        """
        Creates an Eliminator object
        :param correction: correction type name
        :param paired: Boolean that indicates if test is paired
        :param confidence_level: Real, confidence level
        """

        # super().__init__(test_type, dom_type, correction, paired, confidence_level)
        super().__init__(
            test_name=test_type,
            dom_name=dom_type,
            correction=correction,
            paired=paired,
            confidence_level=confidence_level)

        # Initialize statistical test
        if test_type == "f-test":
            self.stest = FriedmanTest(
                correction=correction,
                paired=True,
                confidence_level=confidence_level,
            )

        elif test_type == "t-test":
            self.stest = Ttest(
                correction=correction,
                paired=True,
                confidence_level=confidence_level,
                first_test=first_test
            )

        else:
            self.stest = None

        if dom_type == "dominance":
            self.dtest = DomTest(
                correction=correction,
                paired=False,
                confidence_level=confidence_level,
                first_test=first_test
            )

        elif dom_type == "adaptive-dom":
            self.dtest = AdaptiveDomTest(
                correction=correction,
                paired=False,
                confidence_level=confidence_level,
                first_test=first_test
            )

        else:
            self.dtest = None

    def test_elimination(self,
                         experiment_results: pd.DataFrame,
                         instances: list,
                         ins_stream: list,
                         min_test: int,
                         elite_candidates: list=[],
                         n_best: int = 1) -> EliminationTestResults:
        """
        Performs a mean then statistical test elimination over a set of experiments

        :param experiment_results: results data frame (configurations in columns)
        :param instances: instances to consider in the test
        :param min_test: minimum number of instances a configuration should be executed on to be elegible for elimination
        :param elite_candidates: list of elite configurations ids
        :param n_best: number of elite configurations that should have better performance than an eliminated configuration

        :return: EliminationTestResults object with the test results
        """

        # obtain dom results if dom is used
        # dom is required for time versions
        if self.dtest is not None:
            dom_results = self.dtest.test_elimination(
                experiment_results=experiment_results.copy(),
                instances=instances,
                ins_stream=ins_stream,
                min_test=min_test,
                elite_candidates=elite_candidates,
                n_best=n_best)

        # obtain statistic test results if test is used
        if self.stest is not None:
            stat_results = self.stest.test_elimination(
                experiment_results=experiment_results.copy(),
                instances=instances,
                ins_stream=ins_stream,
                min_test=min_test,
                elite_candidates=elite_candidates,
                n_best=n_best)

        # merge the results
        if self.dtest is not None:
            if self.stest is None:
                return EliminationTestResults(best=dom_results.best,
                                              means=dom_results.means,
                                              ranks=dom_results.ranks,
                                              p_value=None,
                                              alive=dom_results.alive,
                                              discarded=dom_results.discarded,
                                              instances=dom_results.instances,
                                              data=dom_results.data,
                                              overall_best=dom_results.overall_best,
                                              overall_mean=dom_results.overall_mean,
                                              overall_ranks=dom_results.overall_ranks,
                                              overall_data=dom_results.overall_data)

            else:
                _, dom_discarded_ids = self.adjust_elimination(
                    results=experiment_results,
                    test_result=dom_results,
                    min_test=min_test,
                    each_test=1,
                )

                _, stat_discarded_ids = self.adjust_elimination(
                    results=experiment_results,
                    test_result=stat_results,
                    min_test=min_test,
                    each_test=1,
                )

                # Merge test results
                # protect the old best config from dom test
                for s, t in self.dtest.old_best:
                    if s in stat_discarded_ids:
                        stat_discarded_ids.remove(s)
                discarded_ids = list(set(dom_discarded_ids).union(set(stat_discarded_ids)))
                alive_ids = list(set(experiment_results.columns.values) - set(discarded_ids))

                test_results = EliminationTestResults(best=dom_results.best,
                                                    means=dom_results.means,
                                                    ranks=dom_results.ranks,
                                                    p_value=stat_results.p_value,
                                                    alive=alive_ids,
                                                    discarded=discarded_ids,
                                                    instances=instances,
                                                    data=dom_results.data,
                                                    overall_best=dom_results.best,
                                                    overall_mean=dom_results.overall_mean,
                                                    overall_ranks=dom_results.overall_ranks,
                                                    overall_data=experiment_results)

                best, ranks = self.final_ranks(experiment_results, test_results, alive_ids)

                return EliminationTestResults(best=best,
                                            means=experiment_results.mean(),
                                            ranks=ranks,
                                            p_value=stat_results.p_value,
                                            alive=alive_ids,
                                            discarded=discarded_ids,
                                            instances=instances,
                                            data=experiment_results,
                                            overall_best=best,
                                            overall_mean=experiment_results.mean(),
                                            overall_ranks=ranks,
                                            overall_data=experiment_results)

        else:
            return EliminationTestResults(best=stat_results.best,
                                          means=stat_results.means,
                                          ranks=stat_results.ranks,
                                          p_value=stat_results.p_value,
                                          alive=stat_results.alive,
                                          discarded=stat_results.discarded,
                                          instances=stat_results.instances,
                                          data=stat_results.data,
                                          overall_best=stat_results.overall_best,
                                          overall_mean=stat_results.overall_mean,
                                          overall_ranks=stat_results.overall_ranks,
                                          overall_data=stat_results.overall_data)

    def can_test(self, experiment_results):
        pass

    def only_ranks(self, experiment_results):
        """
        Calculate only the ranks of the configurations based on a set of experiments
        If only use statistical test, use the same only_test to the stest
        if not, use the only_test from dominance test (dominance or adaptive dominance)

        :param: experiment_results: results data frame

        :return: EliminationTestResults object with the test results
        """
        configurations = experiment_results.columns.values
        not_na = experiment_results.count()
        mean_data = experiment_results.mean(axis=0)

        if self.dom_name is not None:
            if self.dom_name == "adaptive-dom":
                df = pd.DataFrame({"count": -not_na, "mean": mean_data})
                rank = df[['count', 'mean']].apply(tuple, 1).rank()

            else:
                sum_data = experiment_results.sum(axis=0)
                df = pd.DataFrame({"count": -not_na, "sum": sum_data})
                rank = df[['count', 'sum']].apply(tuple, 1).rank()

        else:
            rank = mean_data.rank()

        best = rank.idxmin()
        ranks = rank.to_dict()
        return EliminationTestResults(best=best,
                                    means=mean_data,
                                    ranks=ranks,
                                    p_value=None,
                                    alive=configurations,
                                    discarded=[],
                                    instances=list(experiment_results.index),
                                    data=experiment_results,
                                    overall_best=best,
                                    overall_mean=mean_data,
                                    overall_ranks=ranks,
                                    overall_data=experiment_results)
