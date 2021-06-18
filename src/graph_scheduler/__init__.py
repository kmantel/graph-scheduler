"""
This module provides utilities used to schedule the execution of psyneulink components

https://princetonuniversity.github.io/PsyNeuLink/Scheduling.html
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
