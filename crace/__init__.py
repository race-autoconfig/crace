# crace/__init__.py

from crace.scripts import run
from crace.scripts import crace_main as main

from crace.utils import Reader
from crace.containers.scenario import Scenario
from crace.containers.parameters import Parameters
from crace.containers.instances import Instances
from crace.containers.crace_options import CraceOptions
from crace.containers.crace_results import CraceResults
from crace.containers.configurations import Configurations

import crace.settings.description as _csd


__version__ = _csd.version
__author__ = _csd.authors
__maintainers__ = _csd.maintainers
__long_description__ = _csd.long_description
__doc__ = _csd.description

__all__ = [
    'run',
    'main',
    'Reader',
    'Scenario',
    'Parameters',
    'Instances',
    'CraceOptions',
    'Configurations',
    'CraceResults'
]

def __dir__():
    return __all__
