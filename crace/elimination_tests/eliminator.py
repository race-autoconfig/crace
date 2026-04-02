import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from typing import Any, Dict, NamedTuple, List


class EliminationTestResults(NamedTuple):
    best: int
    means: pd.Series
    ranks: Dict[int, int]
    p_value: Any
    alive: List[int]
    discarded: List[int]
    instances: List[int]
    data: pd.DataFrame
    overall_best: int
    overall_mean: pd.Series
    overall_ranks: Dict[int, int]
    overall_data: pd.DataFrame


class Eliminator(ABC):
    """
    Class defines a elimination procedure for the race
    :ivar test_name: test type name
    :ivar correction: type of correction
    :ivar paired: Boolean that indicated if the test is paired
    :ivar confidence_level: Real, confidence level for the test
    """

    def __init__(self, test_name, dom_name, correction, paired, confidence_level):
        """
        Creates an Eliminator object

        :param test_name: test type name
        :param correction: correction type name
        :param paired: Boolean that indicates if test is paired
        :param confidence_level: Real, confidence level
        """
        self.test_name = test_name
        self.dom_name = dom_name
        self.correction = correction
        self.paired = paired
        self.confidence_level = confidence_level

    @abstractmethod
    def test_elimination(self) -> EliminationTestResults:
        pass

    @abstractmethod
    def can_test(self):
        pass

    @staticmethod
    def adjust_elimination(
        results: pd.DataFrame,
        test_result: EliminationTestResults,
        min_test: int = 1,
        each_test: int = 1,
    ):
        """
        Adjusts elimination to avoid eliminating "elite" configurations. Based on the number
        of instances used to apply the test, we avoid to eliminate configurations that
        have more instances executed than the ones used in the test.

        :param results: all results to be used in the elimination
        :param test_result: results of the statistical test
        :param min_test: minimum number of instances executed required to discard a configuration
        :param each_test: number of new instances executed required to discard a configuration

        :return: alive configuration ids, discarded ids
        """

        # number of instances in the experiments considered in the test
        # all configurations have this number of instances
        ci = len(test_result.data.index)

        # calculate number of instances to be executed by all configurations
        ti = len(results.index)
        not_na = ti - results.isna().sum(axis=0)

        all_conf = results.columns.values

        discarded_ids = []
        alive_ids = list(set(all_conf) - set(test_result.discarded))
        alive_ids = [int(x) for x in alive_ids]
        # alive_ids = test_result.alive

        # a configuration can be discarded if it does not have more executions than the ones used in the test
        for d in test_result.discarded:
            if not_na[d] <= ci:
                # less or equal instances executed
                if not_na[d] < min_test:
                    # configuration still does not reach first test
                    alive_ids.append(int(d))

                else:
                    if not_na[d] % each_test == 0:
                        # configuration has min_test + x*each_test instances
                        discarded_ids.append(int(d))
                    else:
                        # configuration does not fullfill each_test instances condition

                        alive_ids.append(int(d))
            else:
                # configuration has more instance, and thus, cannot be discarded
                alive_ids.append(d)
        return alive_ids, discarded_ids

    @staticmethod
    def final_ranks(
        results: pd.DataFrame, test_result: EliminationTestResults, alive_ids
    ):
        """
        Calculates the ranks of a set of configurations based on a set of results

        :param results: data frame of all results to be used in the elimination
        :param test_result: results of the statistical test
        :param alive_ids: configuration IDs that should be considered

        :return: ID of the best configuration, Dictionary of ranks (configuration ids as keys)
        """
        results = results[alive_ids]
        conf = results.columns.values
        nc = len(conf)
        ni = len(results.index)
        not_na = ni - results.isna().sum(axis=0)

        if nc == 1:
            return conf[0], {conf[0]: 1}

        ranks = {x: test_result.overall_ranks[x] for x in alive_ids}

        # multi-level ranking code taken from:
        # https://stackoverflow.com/questions/41974374/pandas-rank-by-multiple-columns
        # eval_dict = {x: [r, -not_na[x]] for x, r in test_result.ranks.items()}
        eval_dict = {x: [r, -not_na[x]] for x, r in ranks.items()}
        df = pd.DataFrame(eval_dict, index=["t_rank", "e_count"]).transpose()
        cols = ["e_count", "t_rank"]

        # inf: time out especially for capping versions
        # nan: no result
        if any(not np.isfinite(v) for v in ranks.values()):
            mask_finite = np.isfinite(df["t_rank"])
            tups = df.loc[mask_finite, cols].sort_values(cols, ascending=True).apply(tuple, 1)
        else:
            tups = df[cols].sort_values(cols, ascending=True).apply(tuple, 1)

        f, i = pd.factorize(tups)
        factorized = pd.Series(f + 1, tups.index, dtype='float64')
        final = df.assign(Rank=factorized)
        final.loc[np.isinf(final["t_rank"]), "Rank"] = np.inf

        best = final["Rank"].idxmin()
        ranks = final["Rank"].to_dict()

        return best, ranks
