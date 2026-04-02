import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import truncnorm
from crace.elimination_tests import UniteTest, Eliminator


from .reader import Reader
from .format import format_string, ConditionalReturn
from .logger import setup_logger, setup_race_loggers, get_logger


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def pandas_future_ctx():
    """
    Future warning detection for pandas
    """
    # 'future' is available since pandas 2.0/2.1
    try:
        has_option = (
            hasattr(pd.options, "mode")
            and hasattr(pd.options.mode, "future")
            and hasattr(pd.options.mode.future, "no_silent_downcasting")
        )
    except Exception:
        has_option = False

    class PandasFutureCtx:
        def __enter__(self):
            self.option_ctx = None

            if has_option:
                self.option_ctx = pd.option_context(
                    "future.no_silent_downcasting", True
                )
                self.option_ctx.__enter__()

            self.warn_ctx = warnings.catch_warnings()
            self.warn_ctx.__enter__()

            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message="Downcasting behavior in `replace`"
            )

            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.warn_ctx.__exit__(exc_type, exc_val, exc_tb)

            if self.option_ctx:
                self.option_ctx.__exit__(exc_type, exc_val, exc_tb)

    return PandasFutureCtx()


def truncated_normal(mean=0, sd=1, low=0, upp=10):
    #Function to use the truncated normal distribution
    a, b = (low - mean) / sd, (upp - mean) / sd
    value = truncnorm.rvs(a=a, b=b, loc=mean, scale=sd)
    return float(value)


def get_eliminator(test_type, dom_type, confidence_level, first_test=1, capping: bool = False) -> Eliminator:
    """
    Method to initial elimination test type

    :param: test_type: method for statistical test
    :param: dom_type: method for dominance test
    :param: first_test: the minimal number of executions should be involved in the elimination test
    :param: capping: boolean if for time versions

    :return: Eliminator class
    """

    if dom_type:
        dtl = dom_type.lower()
        if dtl in ("adaptive-dom", "adaptive-dominance"):
            dom_type = "adaptive-dom"
        elif dtl in ("dominance", "dom"):
            dom_type = "dominance"
    else:
        dom_type = None

    correction = None
    if test_type:
        ttl = test_type.lower()
        if ttl in ("f-test", "ftest", "friedman"):
            test_type = "f-test"
        elif ttl in ("t-test", "ttest"):
            test_type = "t-test"
            correction = "none"
        elif ttl in ("t-test-bonferroni", "t-test-bon"):
            test_type = "t-test"
            correction = "bonferroni"
        elif ttl in ("t-test-holm"):
            test_type = "t-test"
            correction = "holm"
    else:
        test_type = None

    test = UniteTest(correction=correction,
                     confidence_level=confidence_level,
                     first_test=first_test,
                     test_type=test_type,
                     dom_type=dom_type,
                     )
    return test


def get_loop():
    """
    get exsiting loop or create a new one
    """
    import asyncio

    try:
        # python 3.7+
        return asyncio.get_running_loop()
    except AttributeError:
        # python 3.6
        return asyncio.get_event_loop()
    except RuntimeError:
        # no exsiting loop, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

def drain_loop(loop):
    """
    Drain the given loop of all pending tasks
    """
    import asyncio

    tasks = asyncio.all_tasks(loop)
    if not tasks:
        return
    for t in tasks:
        if not t.done():
            t.cancel()

    loop.run_until_complete(
        asyncio.gather(*tasks, return_exceptions=True)
    )