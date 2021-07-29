"""
A graph scheduler generates the order in which the nodes of a directed
acyclic graph (DAG) are executed using the structure of the graph and
expressive conditions. Specifically, a scheduler uses a topological
ordering of the nodes as a base sequence of execution and further
restricts execution based on predefined or custom conditions provided by
the user. Patterns of execution are linked to abstract units of time and
may optionally be mapped to real time units using pint.

Source: https://github.com/kmantel/graph-scheduler
Documentation: https://kmantel.github.io/graph-scheduler/
"""

import pint

_unit_registry = pint.get_application_registry()
pint.set_application_registry(_unit_registry)
_unit_registry.precision = 8  # TODO: remove when floating point issues resolved

from . import condition  # noqa: E402
from . import scheduler  # noqa: E402
from . import time  # noqa: E402

from .condition import *  # noqa: E402
from .scheduler import *  # noqa: E402
from .time import *  # noqa: E402

__all__ = list(condition.__all__)
__all__.extend(scheduler.__all__)
__all__.extend(time.__all__)
__all__.extend([
    '_unit_registry'
])

from . import _version  # noqa: E402
__version__ = _version.get_versions()['version']

del pint
