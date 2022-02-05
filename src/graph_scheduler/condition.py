"""

.. _Condition_Overview:

Overview
--------

:class:`Conditions <Condition>` are used to specify when nodes are allowed to execute.  Conditions
can be used to specify a variety of required conditions for execution, including the state of the node
itself (e.g., how many times it has already executed, or the value of one of its attributes), the state of the
:class:`Scheduler <graph_scheduler.scheduler.Scheduler>` (e.g., how many `CONSIDERATION_SET_EXECUTION` s have occurred in the current `ENVIRONMENT_STATE_UPDATE`), or the state of other
nodes in a graph (e.g., whether or how many times they have executed). This package provides a number of
`pre-specified Conditions <Condition_Pre_Specified>` that can be parametrized (e.g., how many times a node should
be executed). `Custom conditions <Condition_Custom>` can also be created, by assigning a function to a Condition that
can reference any node or its attributes, thus providing considerable flexibility for scheduling.

.. _Condition_Creation:

Creating Conditions
-------------------

.. _Condition_Pre_Specified:

*Pre-specified Conditions*
~~~~~~~~~~~~~~~~~~~~~~~~~~

`Pre-specified Conditions <Condition_Pre-Specified_List>` can be instantiated and added to a :class:`Scheduler <graph_scheduler.scheduler.Scheduler>` at any time,
and take effect immediately for the execution of that Scheduler. Most pre-specified Conditions have one or more
arguments that must be specified to achieve the desired behavior. Many Conditions are also associated with an
`owner <Condition.owner>` attribute (a node to which the Condition belongs). Schedulers maintain the data
used to test for satisfaction of Condition, independent in different `execution contexts <default_execution_id>`. The Scheduler is generally
responsible for ensuring that Conditions have access to the necessary data.
When pre-specified Conditions are instantiated within a call to the `add_condition` method of a `Scheduler` or `ConditionSet`,
the Condition's `owner <Condition.owner>` is determined through
context and assigned automatically, as in the following example::

    my_scheduler.add_condition(A, EveryNPasses(1))
    my_scheduler.add_condition(B, EveryNCalls(A, 2))
    my_scheduler.add_condition(C, EveryNCalls(B, 2))

Here, ``EveryNCalls(A, 2)`` for example, is assigned the `owner` ``B``.

.. _Condition_Custom:

*Custom Conditions*
~~~~~~~~~~~~~~~~~~~

Custom Conditions can be created by calling the constructor for the base class (`Condition()`) or one of the
`generic classes <Conditions_Generic>`,  and assigning a function to the **func** argument and any arguments it
requires to the **args** and/or **kwargs** arguments (for formal or keyword arguments, respectively). The function
is called with **args** and **kwargs** by the `Scheduler` on each `PASS` through its `consideration_queue`, and the result is
used to determine whether the associated node is allowed to execute on that `PASS`. Custom Conditions allow
arbitrary schedules to be created, in which the execution of each node can depend on one or more attributes of
any other node in the graph.

.. _Condition_Recurrent_Example:

For example, the following script fragment creates a custom Condition in which ``node_A`` is scheduled to wait to
execute until ``node_B`` has "converged" (that is, settled to the point that none of
its elements has changed in value more than a specified amount since the previous `CONSIDERATION_SET_EXECUTION`)::

    def converge(node, thresh):
        for val in node.delta:
            if abs(val) >= thresh:
                return False
        return True
    epsilon = 0.01
    my_scheduler.add_condition(node_A, NWhen(Condition(converge, node_B, epsilon), 1))

In the example, a function ``converge`` is defined that references the ``delta`` attribute of
a node (which reports the change in its ``value``). The function is assigned to
the standard :class:`Condition` with ``node_A`` and ``epsilon`` as its arguments, and `composite Condition <Conditions_Composite>`
`NWhen` (which is satisfied the first N times after its condition becomes true),  The Condition is assigned to ``node_B``,
thus scheduling it to execute one time when all of the elements of ``node_A`` have changed by less than ``epsilon``.

.. _Condition_Structure:

Structure
---------

The `Scheduler <graph_scheduler.scheduler.Scheduler>` associates every node with a Condition.  If a node has not been explicitly assigned a
Condition, it is assigned a Condition that causes it to be executed whenever it is `under consideration <Scheduler_Algorithm>`
and all its structural parents have been executed at least once since the node's last execution.
Condition subclasses (`listed below <Condition_Pre-Specified_List>`)
provide a standard set of Conditions that can be implemented simply by specifying their parameter(s). There are
six types:

  * `Generic <Conditions_Generic>` - satisfied when a `user-specified function and set of arguments <Condition_Custom>`
    evaluates to `True`;
  * `Static <Conditions_Static>` - satisfied either always or never;
  * `Composite <Conditions_Composite>` - satisfied based on one or more other Conditions;
  * `Time-based <Conditions_Time_Based>` - satisfied based on the current count of units of time at a specified
    `TimeScale`;
  * `Node-based <Conditions_Node_Based>` - based on the execution or state of other nodes.
  * `Convenience <Conditions_Convenience>` - based on other Conditions, condensed for convenience

.. _Condition_Pre-Specified_List:

*List of Pre-specified Conditions*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
    The optional `TimeScale` argument in many `Conditions <Condition>` specifies the unit of time over which the
    Condition operates;  the default value is `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>` for all Conditions except those with "EnvironmentStateUpdate"
    in their name, for which it is `ENVIRONMENT_SEQUENCE`.


.. _Conditions_Generic:

**Generic Conditions** (used to construct `custom Conditions <Condition_Custom>`):

    * `While` (func, *args, **kwargs)
      satisfied whenever the specified function (or callable) called with args and/or kwargs evaluates to :keyword:`True`.
      Equivalent to ``Condition(func, *args, **kwargs)``

    * `WhileNot` (func, *args, **kwargs)
      satisfied whenever the specified function (or callable) called with args and/or kwargs evaluates to :keyword:`False`.
      Equivalent to ``Not(Condition(func, *args, **kwargs))``

.. _Conditions_Static:

**Static Conditions** (independent of other Conditions, nodes or time):

    * `Always`
      always satisfied.

    * `Never`
      never satisfied.


.. _Conditions_Composite:

**Composite Conditions** (based on one or more other Conditions):

    * `All` (*Conditions)
      satisfied whenever all of the specified Conditions are satisfied.

    * `Any` (*Conditions)
      satisfied whenever any of the specified Conditions are satisfied.

    * `Not` (Condition)
      satisfied whenever the specified Condition is not satisfied.

    * `NWhen` (Condition, int)
      satisfied the first specified number of times the specified Condition is satisfied.


.. _Conditions_Time_Based:

**Time-Based Conditions** (based on the count of units of time at a
specified `TimeScale` or `Time <Scheduler_Absolute_Time>`):

    * `TimeInterval` ([`pint.Quantity`, `pint.Quantity`, `pint.Quantity`])
      satisfied every time an optional amount of absolute time has
      passed in between an optional specified range

    * `TimeTermination` (`pint.Quantity`)
      satisfied after the given absolute time

    * `BeforeConsiderationSetExecution` (int[, TimeScale])
      satisfied any time before the specified `CONSIDERATION_SET_EXECUTION` occurs.

    * `AtConsiderationSetExecution` (int[, TimeScale])
      satisfied only during the specified `CONSIDERATION_SET_EXECUTION`.

    * `AfterConsiderationSetExecution` (int[, TimeScale])
      satisfied any time after the specified `CONSIDERATION_SET_EXECUTION` has occurred.

    * `AfterNConsiderationSetExecutions` (int[, TimeScale])
      satisfied when or any time after the specified number of `CONSIDERATION_SET_EXECUTION`\\ s has occurred.

    * `BeforePass` (int[, TimeScale])
      satisfied any time before the specified `PASS` occurs.

    * `AtPass` (int[, TimeScale])
      satisfied only during the specified `PASS`.

    * `AfterPass` (int[, TimeScale])
      satisfied any time after the specified `PASS` has occurred.

    * `AfterNPasses` (int[, TimeScale])
      satisfied when or any time after the specified number of `PASS`\\ es has occurred.

    * `EveryNPasses` (int[, TimeScale])
      satisfied every time the specified number of `PASS`\\ es occurs.

    * `BeforeEnvironmentStateUpdate` (int[, TimeScale])
      satisfied any time before the specified `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>` occurs.

    * `AtEnvironmentStateUpdate` (int[, TimeScale])
      satisfied any time during the specified `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>`.

    * `AfterEnvironmentStateUpdate` (int[, TimeScale])
      satisfied any time after the specified `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>` occurs.

    * `AfterNEnvironmentStateUpdates` (int[, TimeScale])
      satisfied any time after the specified number of `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>`\\ s has occurred.

    * `AtEnvironmentSequence` (int)
      satisfied any time during the specified `ENVIRONMENT_SEQUENCE`.

    * `AfterEnvironmentSequence` (int)
      satisfied any time after the specified `ENVIRONMENT_SEQUENCE` occurs.

    * `AfterNEnvironmentSequences` (int)
      satisfied any time after the specified number of `ENVIRONMENT_SEQUENCE`\\ s has occurred.

.. _Conditions_Node_Based:

**Node-Based Conditions** (based on the execution or state of other nodes):


    * `BeforeNCalls` (node, int[, TimeScale])
      satisfied any time before the specified node has executed the specified number of times.

    * `AtNCalls` (node, int[, TimeScale])
      satisfied when the specified node has executed the specified number of times.

    * `AfterCall` (node, int[, TimeScale])
      satisfied any time after the node has executed the specified number of times.

    * `AfterNCalls` (node, int[, TimeScale])
      satisfied when or any time after the node has executed the specified number of times.

    * `AfterNCallsCombined` (*nodes, int[, TimeScale])
      satisfied when or any time after the specified nodes have executed the specified number
      of times among themselves, in total.

    * `EveryNCalls` (node, int[, TimeScale])
      satisfied when the specified node has executed the specified number of times since the
      last time `owner` has run.

    * `JustRan` (node)
      satisfied if the specified node was assigned to run in the previous `CONSIDERATION_SET_EXECUTION`.

    * `AllHaveRun` (*nodes)
      satisfied when all of the specified nodes have executed at least once.

    * `WhenFinished` (node)
      satisfied when the `is_finished` method of the specified node, \
      given `execution_id` returns `True`.

    * `WhenFinishedAny` (*nodes)
      satisfied when the `is_finished` method of any of the specified \
      nodes, given `execution_id` returns `True`.

    * `WhenFinishedAll` (*nodes)
      satisfied when the `is_finished` method of all of the specified \
      nodes, given `execution_id` returns `True`.

.. _Conditions_Convenience:

**Convenience Conditions** (based on other Conditions, condensed for convenience)


    * `AtEnvironmentStateUpdateStart`
      satisfied at the beginning of an `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>` (`AtPass(0) <AtPass>`)

    * `AtEnvironmentStateUpdateNStart`
      satisfied on `PASS` 0 of the specified `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>` counted using 'TimeScale`

    * `AtEnvironmentSequenceStart`
      satisfied at the beginning of an `ENVIRONMENT_SEQUENCE`

    * `AtEnvironmentSequenceNStart`
      satisfied on `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>` 0 of the specified `ENVIRONMENT_SEQUENCE` counted using 'TimeScale`


.. Condition_Execution:

Execution
---------

When the `Scheduler` `runs <Schedule_Execution>`, it makes a sequential `PASS` through its `consideration_queue`,
evaluating each `consideration_set <consideration_set>` in the queue to determine which nodes should be assigned
to execute. It evaluates the nodes in each set by calling the `is_satisfied` method of the Condition associated
with each of those nodes.  If it returns `True`, then the node is assigned to the execution set for the
`CONSIDERATION_SET_EXECUTION` of execution generated by that `PASS`.  Otherwise, the node is not executed.

.. _Condition_Class_Reference:

Class Reference
---------------

"""

import collections
import itertools
import logging
import operator
import typing
import warnings

import numpy as np
import pint

from graph_scheduler import _unit_registry
from graph_scheduler.time import TimeScale
from graph_scheduler.utilities import call_with_pruned_args

__all__ = [
    'AfterCall', 'AfterNCalls', 'AfterNCallsCombined', 'AfterNPasses', 'AfterNConsiderationSetExecutions', 'AfterNEnvironmentStateUpdates', 'AfterPass',
    'AtEnvironmentSequence', 'AfterEnvironmentSequence', 'AfterNEnvironmentSequences', 'AfterConsiderationSetExecution', 'AfterEnvironmentStateUpdate', 'All', 'AllHaveRun', 'Always', 'And', 'Any',
    'AtNCalls','AtPass', 'AtEnvironmentSequenceStart', 'AtEnvironmentSequenceNStart', 'AtConsiderationSetExecution', 'AtEnvironmentStateUpdate',
    'AtEnvironmentStateUpdateStart', 'AtEnvironmentStateUpdateNStart', 'BeforeNCalls', 'BeforePass', 'BeforeConsiderationSetExecution', 'BeforeEnvironmentStateUpdate',
    'Condition','ConditionError', 'ConditionSet', 'EveryNCalls', 'EveryNPasses',
    'JustRan', 'Never', 'Not', 'NWhen', 'Or', 'WhenFinished', 'WhenFinishedAll', 'WhenFinishedAny', 'While', 'WhileNot', 'TimeInterval', 'TimeTermination', 'Threshold'
]

logger = logging.getLogger(__name__)


_pint_all_time_units = sorted(
    set([
        getattr(_unit_registry, f'{x}s')
        for x in _unit_registry._prefixes.keys()
    ]),
    reverse=True
)

comparison_operators = {
    '<': operator.lt,
    '<=': operator.le,
    '>': operator.gt,
    '>=': operator.ge,
    '==': operator.eq,
    '!=': operator.ne
}


def _quantity_as_integer(q):
    rounded = round(q.m, _unit_registry.precision)
    if rounded == int(rounded):
        return int(rounded) * q.u
    else:
        raise ValueError(f'{q} cannot be safely converted to integer magnitude')


def _reduce_quantity_to_integer(q):
    # find the largest time unit for which q can be expressed as an integer
    for u in filter(lambda u: u <= q.u, _pint_all_time_units):
        try:
            return _quantity_as_integer(q.to(u))
        except ValueError:
            pass

    return q


def _parse_absolute_unit(n, unit=None):
    def _get_quantity(n, unit):
        if isinstance(n, _unit_registry.Quantity):
            return n

        if isinstance(n, str):
            try:
                full_quantity = _unit_registry.Quantity(n)
            except pint.errors.UndefinedUnitError:
                pass
            else:
                # n is an actual full quantity (e.g. '1ms') not just a number ('1')
                if full_quantity.u != _unit_registry.Unit('dimensionless'):
                    return full_quantity
                else:
                    try:
                        n = int(n)
                    except ValueError:
                        n = float(n)

        try:
            # handle string representation of a unit
            unit = getattr(_unit_registry, unit)
        except TypeError:
            pass

        assert isinstance(unit, _unit_registry.Unit)
        n = n * unit

        return n

    if n is None:
        return n

    # store as the base integer quantity
    return _reduce_quantity_to_integer(_get_quantity(n, unit))


class ConditionError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class ConditionSet(object):
    """Used in conjunction with a `Scheduler <graph_scheduler.scheduler.Scheduler>` to store the `Conditions <Condition>` associated with a node.

    Arguments
    ---------

    conditions : Dict[node: `Condition`]
        specifies an iterable collection of nodes and the `Conditions <Condition>` associated
        with each.

    Attributes
    ----------

    conditions : Dict[node: `Condition`]
        the key of each entry is a node, and its value is the `Condition <Condition>` associated
        with that node.  Conditions can be added to the
        ConditionSet using the ConditionSet's `add_condition` method.

    """
    def __init__(self, conditions=None):
        self.conditions = {}

        if conditions is not None:
            self.add_condition_set(conditions)

    def __contains__(self, item):
        return item in self.conditions

    def __repr__(self):
        condition_str = '\n\t'.join([f'{owner}: {condition}' for owner, condition in self.conditions.items()])
        return '{0}({1}{2}{3})'.format(
            self.__class__.__name__,
            '\n\t' if len(condition_str) > 0 else '',
            condition_str,
            '\n' if len(condition_str) > 0 else ''
        )

    def __iter__(self):
        return iter(self.conditions)

    def __getitem__(self, key):
        return self.conditions[key]

    def __setitem__(self, key, value):
        self.conditions[key] = value

    def add_condition(self, owner, condition):
        """
        Adds a `Condition` to the ConditionSet. If **owner** already has a Condition, it is overwritten
        with the new one. If you want to add multiple conditions to a single owner, use the
        `composite Conditions <Conditions_Composite>` to accurately specify the desired behavior.

        Arguments
        ---------

        owner : node
            specifies the node with which the **condition** should be associated. **condition**
            will govern the execution behavior of **owner**

        condition : Condition
            specifies the Condition, associated with the **owner** to be added to the ConditionSet.
        """
        condition.owner = owner
        self.conditions[owner] = condition

    def add_condition_set(self, conditions):
        """
        Adds a set of `Conditions <Condition>` (in the form of a dict or another ConditionSet) to the ConditionSet.
        Any Condition added here will overwrite an existing Condition for a given owner.
        If you want to add multiple conditions to a single owner, add a single `Composite Condition <Conditions_Composite>`
        to accurately specify the desired behavior.

        Arguments
        ---------

        conditions : dict[node: `Condition`], `ConditionSet`
            specifies collection of Conditions to be added to this ConditionSet,

            if a dict is provided:
                each entry should map an owner node (the node whose execution behavior will be
                governed) to a `Condition <Condition>`

        """
        for owner in conditions:
            self.add_condition(owner, conditions[owner])


class Condition:
    """
    Used in conjunction with a :class:`Scheduler` to specify the condition under which a node should be
    allowed to execute.

    Arguments
    ---------

    func : callable
        specifies function to be called when the Condition is evaluated, to determine whether it is currently satisfied.

    args : *args
        specifies formal arguments to pass to `func` when the Condition is evaluated.

    kwargs : **kwargs
        specifies keyword arguments to pass to `func` when the Condition is evaluated.

    Attributes
    ----------

    owner (node):
        the node with which the Condition is associated, and the execution of which it determines.

    """
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

        self._owner = None

        for k in kwargs:
            setattr(self, k, kwargs[k])

    def __str__(self):
        return '{0}({1}{2})'.format(
            self.__class__.__name__,
            ', '.join([str(arg) for arg in self.args]) if len(self.args) > 0 else '',
            ', {0}'.format(self.kwargs) if len(self.kwargs) > 0 else ''
        )

    @property
    def owner(self):
        return self._owner

    @owner.setter
    def owner(self, value):
        logger.debug('Condition ({0}) setting owner to {1}'.format(type(self).__name__, value))
        self._owner = value

    def is_satisfied(self, *args, execution_id=None, **kwargs):
        """
        the function called to determine satisfaction of this Condition.

        Arguments
        ---------
        args : *args
            specifies additional formal arguments to pass to `func` when the Condition is evaluated.
            these are appended to the **args** specified at instantiation of this Condition

        kwargs : **kwargs
            specifies additional keyword arguments to pass to `func` when the Condition is evaluated.
            these are added to the **kwargs** specified at instantiation of this Condition

        Returns
        -------
            True - if the Condition is satisfied
            False - if the Condition is not satisfied
        """
        # update so that kwargs can override self.kwargs
        kwargs_to_pass = self.kwargs.copy()
        kwargs_to_pass.update(kwargs)

        return call_with_pruned_args(
            self.func,
            *self.args,
            *args,
            execution_id=execution_id,
            **kwargs_to_pass
        )

    @property
    def absolute_intervals(self):
        """
        In absolute time, repeated intervals for satisfaction or
        unsatisfaction of this Condition or those it contains
        """
        return []

    @property
    def absolute_fixed_points(self):
        """
        In absolute time, specific time points for satisfaction or
        unsatisfaction of this Condition or those it contains
        """
        return []

    @property
    def is_absolute(self):
        return False


class AbsoluteCondition(Condition):
    @property
    def is_absolute(self):
        return True


class _DependencyValidation:
    @Condition.owner.setter
    def owner(self, value):
        try:
            # "dependency" or "dependencies" is always the first positional argument
            if not isinstance(self.args[0], collections.abc.Iterable) or isinstance(self.args[0], str):
                dependencies = [self.args[0]]
            else:
                dependencies = self.args[0]
        except IndexError:
            pass
        else:
            if value in dependencies:
                warnings.warn(
                    f'{self} is dependent on {value}, but you are assigning {value} as its owner.'
                    ' This may result in infinite loops or unknown behavior.',
                    stacklevel=5
                )

        self._owner = value


#########################################################################################################
# Included Conditions
#########################################################################################################

######################################################################
# Generic Conditions
#   - convenience wrappers
######################################################################


While = Condition


class WhileNot(Condition):
    """
    WhileNot

    Parameters:

        func : callable
            specifies function to be called when the Condition is evaluated, to determine whether it is currently satisfied.

        args : *args
            specifies formal arguments to pass to `func` when the Condition is evaluated.

        kwargs : **kwargs
            specifies keyword arguments to pass to `func` when the Condition is evaluated.

    Satisfied when:

        - **func** is False

    """
    def __init__(self, func, *args, **kwargs):
        def inner_func(*args, **kwargs):
            return not call_with_pruned_args(func, *args, **kwargs)
        super().__init__(inner_func, *args, **kwargs)

######################################################################
# Static Conditions
#   - independent of nodes and time
######################################################################


class Always(Condition):
    """Always

    Parameters:

        none

    Satisfied when:

        - always satisfied.

    """
    def __init__(self):
        super().__init__(lambda: True)


class Never(Condition):
    """Never

    Parameters:

        none

    Satisfied when:

        - never satisfied.
    """
    def __init__(self):
        super().__init__(lambda: False)

######################################################################
# Composite Conditions
#   - based on other Conditions
######################################################################


class CompositeCondition(Condition):
    @Condition.owner.setter
    def owner(self, value):
        super(CompositeCondition, CompositeCondition).owner.__set__(self, value)
        for cond in self.args:
            logger.debug('owner setter: Setting owner of {0} to ({1})'.format(cond, value))
            if cond.owner is None:
                cond.owner = value

    @property
    def absolute_intervals(self):
        return list(itertools.chain.from_iterable([
            c.absolute_intervals
            for c in filter(lambda a: a.is_absolute, self.args)
        ]))

    @property
    def absolute_fixed_points(self):
        return list(itertools.chain.from_iterable([
            c.absolute_fixed_points
            for c in filter(lambda a: a.is_absolute, self.args)
        ]))

    @property
    def is_absolute(self):
        return any(a.is_absolute for a in self.args)


class All(CompositeCondition):
    """All

    Parameters:

        args: one or more `Conditions <Condition>`

    Satisfied when:

        - all of the Conditions in args are satisfied.

    Notes:

        - To initialize with a list (for example)::

            conditions = [AfterNCalls(node, 5) for node in node_list]

          unpack the list to supply its members as args::

           composite_condition = All(*conditions)

    """
    def __init__(self, *args, **dependencies):
        args += tuple(*[v for k, v in dependencies.items()])
        super().__init__(self.satis, *args)

    def satis(self, *conds, **kwargs):
        for cond in conds:
            if not cond.is_satisfied(**kwargs):
                return False
        return True


class Any(CompositeCondition):
    """Any

    Parameters:

        args: one or more `Conditions <Condition>`

    Satisfied when:

        - one or more of the Conditions in **args** is satisfied.

    Notes:

        - To initialize with a list (for example)::

            conditions = [AfterNCalls(node, 5) for node in node_list]

          unpack the list to supply its members as args::

           composite_condition = Any(*conditions)

    """
    def __init__(self, *args, **dependencies):
        args += tuple(*[v for k, v in dependencies.items()])
        super().__init__(self.satis, *args)

    def satis(self, *conds, **kwargs):
        for cond in conds:
            if cond.is_satisfied(**kwargs):
                return True
        return False


And = All
Or = Any


class Not(Condition):
    """Not

    Parameters:

        condition(Condition): a `Condition`

    Satisfied when:

        - **condition** is not satisfied.

    """
    def __init__(self, condition):
        self.condition = condition

        def inner_func(condition, *args, **kwargs):
            return not condition.is_satisfied(*args, **kwargs)
        super().__init__(inner_func, condition)

    @Condition.owner.setter
    def owner(self, value):
        super(Not, Not).owner.__set__(self, value)
        self.condition.owner = value


class NWhen(Condition):
    """NWhen

    Parameters:

        condition(Condition): a `Condition`

        n(int): the maximum number of times this condition will be satisfied

    Satisfied when:

        - the first **n** times **condition** is satisfied upon evaluation

    """
    def __init__(self, condition, n=1):
        self.satisfactions = {}
        self.condition = condition

        super().__init__(self.satis, condition, n)

    @Condition.owner.setter
    def owner(self, value):
        super(NWhen, NWhen).owner.__set__(self, value)
        self.condition.owner = value

    def satis(self, condition, n, *args, scheduler=None, execution_id=None, **kwargs):
        if execution_id not in self.satisfactions:
            self.satisfactions[execution_id] = 0

        if self.satisfactions[execution_id] < n:
            if call_with_pruned_args(condition.is_satisfied, *args, scheduler=scheduler, execution_id=execution_id, **kwargs):
                self.satisfactions[execution_id] += 1
                return True
        return False


######################################################################
# Time-based Conditions
#   - satisfied based only on time
######################################################################

class TimeInterval(AbsoluteCondition):
    """TimeInterval

    Attributes:

        repeat
            the interval between *unit*s where this condition can be
            satisfied

        start
            the time at/after which this condition can be
            satisfied

        end
            the time at/fter which this condition can be
            satisfied

        unit
            the `pint.Unit` to use for scalar values of *repeat*,
            *start*, and *end*

        start_inclusive
            if True, *start* allows satisfaction exactly at the time
            corresponding to *start*. if False, satisfaction can occur
            only after *start*

        end_inclusive
            if True, *end* allows satisfaction exactly until the time
            corresponding to *end*. if False, satisfaction can occur
            only before *end*


    Satisfied when:

        Every *repeat* units of time at/after *start* and before/through
        *end*

    Notes:

        Using a `TimeInterval` as a
        `termination Condition <Scheduler_Termination_Conditions>` may
        result in unexpected behavior. The user may be inclined to
        create **TimeInterval(end=x)** to terminate at time **x**, but
        this will do the opposite and be true only and always until time
        **x**, terminating at any time before **x**. If in doubt, use
        `TimeTermination` instead.

        If the scheduler is not set to `exact_time_mode = True`,
        *start_inclusive* and *end_inclusive* may not behave as
        expected. See `Scheduler_Exact_Time` for more info.
    """
    def __init__(
        self,
        repeat: typing.Union[int, str, pint.Quantity] = None,
        start: typing.Union[int, str, pint.Quantity] = None,
        end: typing.Union[int, str, pint.Quantity] = None,
        unit: typing.Union[str, pint.Unit] = _unit_registry.ms,
        start_inclusive: bool = True,
        end_inclusive: bool = True
    ):
        if repeat is start is end is None:
            raise ConditionError(
                'At least one of "repeat", "start", "end" must be specified'
            )

        if start is not None and end is not None and start > end:
            raise ConditionError(f'Start ({start}) is later than {end}')

        repeat = _parse_absolute_unit(repeat, unit)
        start = _parse_absolute_unit(start, unit)
        end = _parse_absolute_unit(end, unit)

        def func(scheduler, execution_id):
            satisfied = True
            clock = scheduler.get_clock(execution_id)

            if scheduler._in_exact_time_mode and start is not None:
                offset = start
            else:
                for i, cs in enumerate(scheduler.consideration_queue):
                    if self.owner in cs:
                        offset = i * clock.time.absolute_interval

            if repeat is not None:
                satisfied &= round(
                    (clock.time.absolute - offset) % repeat,
                    _unit_registry.precision
                ) == 0

            if start is not None:
                if start_inclusive:
                    satisfied &= clock.time.absolute >= start
                else:
                    satisfied &= clock.time.absolute > start

            if end is not None:
                if end_inclusive:
                    satisfied &= clock.time.absolute <= end
                else:
                    satisfied &= clock.time.absolute < end

            return satisfied

        super().__init__(
            func,
            repeat=repeat,
            start=start,
            end=end,
            unit=unit,
            start_inclusive=start_inclusive,
            end_inclusive=end_inclusive
        )

    @property
    def absolute_intervals(self):
        return [self.repeat] if self.repeat is not None else []

    @property
    def absolute_fixed_points(self):
        fp = []
        if self.start_inclusive and self.start is not None:
            fp.append(self.start)
        if self.end_inclusive and self.end is not None:
            fp.append(self.end)

        return fp


class TimeTermination(AbsoluteCondition):
    """TimeTermination

    Attributes:

        t
            the time at/after which this condition is satisfied

        unit
            the `pint.Unit` to use for scalar values of *t*, *start*,
            and *end*

        start_inclusive
            if True, the condition is satisfied exactly at the time
            corresponding to *t*. if False, satisfaction can occur
            only after *t*

    Satisfied when:

        At/After time *t*
    """
    def __init__(
        self,
        t: typing.Union[int, str, pint.Quantity],
        inclusive: bool = True,
        unit: typing.Union[str, pint.Unit] = _unit_registry.ms,
    ):
        t = _parse_absolute_unit(t, unit)

        def func(scheduler, execution_id):
            clock = scheduler.get_clock(execution_id)

            if inclusive:
                return clock.time.absolute >= t
            else:
                return clock.time.absolute > t

        super().__init__(func, t=t, inclusive=inclusive)

    @property
    def absolute_fixed_points(self):
        return [self.t] if self.inclusive else []


class BeforeConsiderationSetExecution(Condition):
    """BeforeConsiderationSetExecution

    Parameters:

        n(int): the 'CONSIDERATION_SET_EXECUTION' before which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `CONSIDERATION_SET_EXECUTION`\\ s (default: TimeScale.ENVIRONMENT_STATE_UPDATE)

    Satisfied when:

        - at most n-1 `CONSIDERATION_SET_EXECUTION`\\ s have occurred within one unit of time at the `TimeScale` specified by **time_scale**.

    Notes:

        - Counts of TimeScales are zero-indexed (that is, the first `CONSIDERATION_SET_EXECUTION` is 0, the second `CONSIDERATION_SET_EXECUTION` is 1, etc.);
          so, `BeforeConsiderationSetExecution(2)` is satisfied at `CONSIDERATION_SET_EXECUTION` 0 and `CONSIDERATION_SET_EXECUTION` 1.

    """
    def __init__(self, n, time_scale=TimeScale.ENVIRONMENT_STATE_UPDATE):
        def func(n, time_scale, scheduler=None, execution_id=None):
            try:
                return scheduler.get_clock(execution_id).get_total_times_relative(TimeScale.CONSIDERATION_SET_EXECUTION, time_scale) < n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, n, time_scale)


class AtConsiderationSetExecution(Condition):
    """AtConsiderationSetExecution

    Parameters:

        n(int): the `CONSIDERATION_SET_EXECUTION` at which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `CONSIDERATION_SET_EXECUTION`\\ s (default: TimeScale.ENVIRONMENT_STATE_UPDATE)

    Satisfied when:

        - exactly n `CONSIDERATION_SET_EXECUTION`\\ s have occurred within one unit of time at the `TimeScale` specified by **time_scale**.

    Notes:

        - Counts of TimeScales are zero-indexed (that is, the first 'CONSIDERATION_SET_EXECUTION' is pass 0, the second 'CONSIDERATION_SET_EXECUTION' is 1, etc.);
          so, `AtConsiderationSetExecution(1)` is satisfied when a single `CONSIDERATION_SET_EXECUTION` (`CONSIDERATION_SET_EXECUTION` 0) has occurred, and `AtConsiderationSetExecution(2)` is satisfied
          when two `CONSIDERATION_SET_EXECUTION`\\ s have occurred (`CONSIDERATION_SET_EXECUTION` 0 and `CONSIDERATION_SET_EXECUTION` 1), etc..

    """
    def __init__(self, n, time_scale=TimeScale.ENVIRONMENT_STATE_UPDATE):
        def func(n, scheduler=None, execution_id=None):
            try:
                return scheduler.get_clock(execution_id).get_total_times_relative(TimeScale.CONSIDERATION_SET_EXECUTION, time_scale) == n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, n)


class AfterConsiderationSetExecution(Condition):
    """AfterConsiderationSetExecution

    Parameters:

        n(int): the `CONSIDERATION_SET_EXECUTION` after which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `CONSIDERATION_SET_EXECUTION`\\ s (default: TimeScale.ENVIRONMENT_STATE_UPDATE)

    Satisfied when:

        - at least n+1 `CONSIDERATION_SET_EXECUTION`\\ s have occurred within one unit of time at the `TimeScale` specified by **time_scale**.

    Notes:

        - Counts of TimeScals are zero-indexed (that is, the first `CONSIDERATION_SET_EXECUTION` is 0, the second `CONSIDERATION_SET_EXECUTION` is 1, etc.); so,
          `AfterConsiderationSetExecution(1)` is satisfied after `CONSIDERATION_SET_EXECUTION` 1 has occurred and thereafter (i.e., in `CONSIDERATION_SET_EXECUTION`\\ s 2, 3, 4, etc.).

    """
    def __init__(self, n, time_scale=TimeScale.ENVIRONMENT_STATE_UPDATE):
        def func(n, time_scale, scheduler=None, execution_id=None):
            try:
                return scheduler.get_clock(execution_id).get_total_times_relative(TimeScale.CONSIDERATION_SET_EXECUTION, time_scale) > n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, n, time_scale)


class AfterNConsiderationSetExecutions(Condition):
    """AfterNConsiderationSetExecutions

    Parameters:

        n(int): the number of `CONSIDERATION_SET_EXECUTION`\\ s after which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `CONSIDERATION_SET_EXECUTION`\\ s (default: TimeScale.ENVIRONMENT_STATE_UPDATE)


    Satisfied when:

        - at least n `CONSIDERATION_SET_EXECUTION`\\ s have occurred within one unit of time at the `TimeScale` specified by **time_scale**.

    """
    def __init__(self, n, time_scale=TimeScale.ENVIRONMENT_STATE_UPDATE):
        def func(n, time_scale, scheduler=None, execution_id=None):
            try:
                return scheduler.get_clock(execution_id).get_total_times_relative(TimeScale.CONSIDERATION_SET_EXECUTION, time_scale) >= n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, n, time_scale)


class BeforePass(Condition):
    """BeforePass

    Parameters:

        n(int): the 'PASS' before which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `PASS`\\ es (default: TimeScale.ENVIRONMENT_STATE_UPDATE)

    Satisfied when:

        - at most n-1 `PASS`\\ es have occurred within one unit of time at the `TimeScale` specified by **time_scale**.

    Notes:

        - Counts of TimeScales are zero-indexed (that is, the first `PASS` is 0, the second `PASS` is 1, etc.);
          so, `BeforePass(2)` is satisfied at `PASS` 0 and `PASS` 1.

    """
    def __init__(self, n, time_scale=TimeScale.ENVIRONMENT_STATE_UPDATE):
        def func(n, time_scale, scheduler=None, execution_id=None):
            try:
                return scheduler.get_clock(execution_id).get_total_times_relative(TimeScale.PASS, time_scale) < n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, n, time_scale)


class AtPass(Condition):
    """AtPass

    Parameters:

        n(int): the `PASS` at which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `PASS`\\ es (default: TimeScale.ENVIRONMENT_STATE_UPDATE)

    Satisfied when:

        - exactly n `PASS`\\ es have occurred within one unit of time at the `TimeScale` specified by **time_scale**.

    Notes:

        - Counts of TimeScales are zero-indexed (that is, the first 'PASS' is pass 0, the second 'PASS' is 1, etc.);
          so, `AtPass(1)` is satisfied when a single `PASS` (`PASS` 0) has occurred, and `AtPass(2)` is satisfied
          when two `PASS`\\ es have occurred (`PASS` 0 and `PASS` 1), etc..

    """
    def __init__(self, n, time_scale=TimeScale.ENVIRONMENT_STATE_UPDATE):
        def func(n, scheduler=None, execution_id=None):
            try:
                return scheduler.get_clock(execution_id).get_total_times_relative(TimeScale.PASS, time_scale) == n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, n)


class AfterPass(Condition):
    """AfterPass

    Parameters:

        n(int): the `PASS` after which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `PASS`\\ es (default: TimeScale.ENVIRONMENT_STATE_UPDATE)

    Satisfied when:

        - at least n+1 `PASS`\\ es have occurred within one unit of time at the `TimeScale` specified by **time_scale**.

    Notes:

        - Counts of TimeScales are zero-indexed (that is, the first `PASS` is 0, the second `PASS` is 1, etc.); so,
          `AfterPass(1)` is satisfied after `PASS` 1 has occurred and thereafter (i.e., in `PASS`\\ es 2, 3, 4, etc.).

    """
    def __init__(self, n, time_scale=TimeScale.ENVIRONMENT_STATE_UPDATE):
        def func(n, time_scale, scheduler=None, execution_id=None):
            try:
                return scheduler.get_clock(execution_id).get_total_times_relative(TimeScale.PASS, time_scale) > n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, n, time_scale)


class AfterNPasses(Condition):
    """AfterNPasses

    Parameters:

        n(int): the number of `PASS`\\ es after which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `PASS`\\ es (default: TimeScale.ENVIRONMENT_STATE_UPDATE)


    Satisfied when:

        - at least n `PASS`\\ es have occurred within one unit of time at the `TimeScale` specified by **time_scale**.

    """
    def __init__(self, n, time_scale=TimeScale.ENVIRONMENT_STATE_UPDATE):
        def func(n, time_scale, scheduler=None, execution_id=None):
            try:
                return scheduler.get_clock(execution_id).get_total_times_relative(TimeScale.PASS, time_scale) >= n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, n, time_scale)


class EveryNPasses(Condition):
    """EveryNPasses

    Parameters:

        n(int): the frequency of passes with which this condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `PASS`\\ es (default: TimeScale.ENVIRONMENT_STATE_UPDATE)

    Satisfied when:

        - `PASS` 0

        - the specified number of `PASS`\\ es that has occurred within a unit of time (at the `TimeScale` specified by
          **time_scale**) is evenly divisible by n.

    """
    def __init__(self, n, time_scale=TimeScale.ENVIRONMENT_STATE_UPDATE):
        def func(n, time_scale, scheduler=None, execution_id=None):
            try:
                return scheduler.get_clock(execution_id).get_total_times_relative(TimeScale.PASS, time_scale) % n == 0
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, n, time_scale)


class BeforeEnvironmentStateUpdate(Condition):
    """BeforeEnvironmentStateUpdate

    Parameters:

        n(int): the `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>` before which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>`\\ s
        (default: TimeScale.ENVIRONMENT_SEQUENCE)

    Satisfied when:

        - at most n-1 `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>`\\ s have occurred within one unit of time at the `TimeScale`
          specified by **time_scale**.

    Notes:

        - Counts of TimeScales are zero-indexed (that is, the first `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>` is 0, the second
          `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>` is 1, etc.); so, `BeforeEnvironmentStateUpdate(2)` is satisfied at `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>` 0
          and `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>` 1.

    """
    def __init__(self, n, time_scale=TimeScale.ENVIRONMENT_SEQUENCE):
        def func(n, scheduler=None, execution_id=None):
            try:
                return scheduler.get_clock(execution_id).get_total_times_relative(TimeScale.ENVIRONMENT_STATE_UPDATE, time_scale) < n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, n)


class AtEnvironmentStateUpdate(Condition):
    """AtEnvironmentStateUpdate

    Parameters:

        n(int): the `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>` at which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>`\\ s
        (default: TimeScale.ENVIRONMENT_SEQUENCE)

    Satisfied when:

        - exactly n `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>`\\ s have occurred within one unit of time at the `TimeScale`
          specified by **time_scale**.

    Notes:

        - Counts of TimeScales are zero-indexed (that is, the first `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>` is 0,
          the second `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>` is 1, etc.); so, `AtEnvironmentStateUpdate(1)` is satisfied when one
          `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>` (`ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>` 0) has already occurred.

    """
    def __init__(self, n, time_scale=TimeScale.ENVIRONMENT_SEQUENCE):
        def func(n, scheduler=None, execution_id=None):
            try:
                return scheduler.get_clock(execution_id).get_total_times_relative(TimeScale.ENVIRONMENT_STATE_UPDATE, time_scale) == n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, n)


class AfterEnvironmentStateUpdate(Condition):
    """AfterEnvironmentStateUpdate

    Parameters:

        n(int): the `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>` after which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>`\\ s.
        (default: TimeScale.ENVIRONMENT_SEQUENCE)

    Satisfied when:

        - at least n+1 `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>`\\ s have occurred within one unit of time at the `TimeScale`
          specified by **time_scale**.

    Notes:

        - Counts of TimeScales are zero-indexed (that is, the first `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>` is 0, the second
        `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>` is 1, etc.); so,  `AfterPass(1)` is satisfied after `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>` 1
        has occurred and thereafter (i.e., in `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>`\\ s 2, 3, 4, etc.).

    """
    def __init__(self, n, time_scale=TimeScale.ENVIRONMENT_SEQUENCE):
        def func(n, scheduler=None, execution_id=None):
            try:
                return scheduler.get_clock(execution_id).get_total_times_relative(TimeScale.ENVIRONMENT_STATE_UPDATE, time_scale) > n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, n)


class AfterNEnvironmentStateUpdates(Condition):
    """AfterNEnvironmentStateUpdates

    Parameters:

        n(int): the number of `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>`\\ s after which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>`\\ s
        (default: TimeScale.ENVIRONMENT_SEQUENCE)

    Satisfied when:

        - at least n `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>`\\ s have occured within one unit of time at the `TimeScale`
          specified by **time_scale**.

    """
    def __init__(self, n, time_scale=TimeScale.ENVIRONMENT_SEQUENCE):
        def func(n, time_scale, scheduler=None, execution_id=None):
            try:
                return scheduler.get_clock(execution_id).get_total_times_relative(TimeScale.ENVIRONMENT_STATE_UPDATE, time_scale) >= n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, n, time_scale)


class AtEnvironmentSequence(Condition):
    """AtEnvironmentSequence

    Parameters:

        n(int): the `ENVIRONMENT_SEQUENCE` at which the Condition is satisfied

    Satisfied when:

        - exactly n `ENVIRONMENT_SEQUENCE`\\ s have occurred.

    Notes:
        - `ENVIRONMENT_SEQUENCE`\\ s are managed by the environment \
        using the Scheduler (e.g. \
        `end_environment_sequence <Scheduler.end_environment_sequence>`\
        ) and are not automatically updated by this package.

    """
    def __init__(self, n):
        def func(n, scheduler=None, execution_id=None):
            try:
                return scheduler.get_clock(execution_id).time.environment_sequence == n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, n)


class AfterEnvironmentSequence(Condition):
    """AfterEnvironmentSequence

    Parameters:

        n(int): the `ENVIRONMENT_SEQUENCE` after which the Condition is satisfied

    Satisfied when:

        - at least n+1 `ENVIRONMENT_SEQUENCE`\\ s have occurred.

    Notes:
        - `ENVIRONMENT_SEQUENCE`\\ s are managed by the environment \
        using the Scheduler (e.g. \
        `end_environment_sequence <Scheduler.end_environment_sequence>`\
        ) and are not automatically updated by this package.

    """
    def __init__(self, n):
        def func(n, scheduler=None, execution_id=None):
            try:
                return scheduler.get_clock(execution_id).time.environment_sequence > n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, n)


class AfterNEnvironmentSequences(Condition):
    """AfterNEnvironmentSequences

    Parameters:

        n(int): the number of `ENVIRONMENT_SEQUENCE`\\ s after which the Condition is satisfied

    Satisfied when:

        - at least n `ENVIRONMENT_SEQUENCE`\\ s have occured.

    Notes:
        - `ENVIRONMENT_SEQUENCE`\\ s are managed by the environment \
        using the Scheduler (e.g. \
        `end_environment_sequence <Scheduler.end_environment_sequence>`\
        ) and are not automatically updated by this package.

    """

    def __init__(self, n):
        def func(n, scheduler=None, execution_id=None):
            try:
                return scheduler.get_clock(execution_id).time.environment_sequence >= n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, n)

######################################################################
# Node-based Conditions
#   - satisfied based on executions or state of nodes
######################################################################


class BeforeNCalls(_DependencyValidation, Condition):
    """BeforeNCalls

    Parameters:

        dependency(node):  the node on which the Condition depends

        n(int): the number of executions of **dependency** before which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting executions of **dependency**
        (default: TimeScale.ENVIRONMENT_STATE_UPDATE)

    Satisfied when:

        - the node specified in **dependency** has executed at most n-1 times
          within one unit of time at the `TimeScale` specified by **time_scale**.

    """
    def __init__(self, dependency, n, time_scale=TimeScale.ENVIRONMENT_STATE_UPDATE):
        self.time_scale = time_scale

        def func(dependency, n, scheduler=None, execution_id=None):
            try:
                num_calls = scheduler.counts_total[execution_id][time_scale][dependency]
                logger.debug('{0} has reached {1} num_calls in {2}'.format(dependency, num_calls, time_scale.name))
                return num_calls < n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, dependency, n)

# NOTE:
# The behavior of AtNCalls is not desired (i.e. depending on the order nodes are checked, B running AtNCalls(A, x))
# may run on both the xth and x+1st call of A; if A and B are not parent-child
# A fix could invalidate key assumptions and affect many other conditions
# Since this condition is unlikely to be used, it's best to leave it for now


class AtNCalls(_DependencyValidation, Condition):
    """AtNCalls

    Parameters:

        dependency(node):  the node on which the Condition depends

        n(int): the number of executions of **dependency** at which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting executions of **dependency**
        (default: TimeScale.ENVIRONMENT_STATE_UPDATE)

    Satisfied when:

        - the node specified in **dependency** has executed exactly n times
          within one unit of time at the `TimeScale` specified by **time_scale**.

    """
    def __init__(self, dependency, n, time_scale=TimeScale.ENVIRONMENT_STATE_UPDATE):
        self.time_scale = time_scale

        def func(dependency, n, scheduler=None, execution_id=None):
            try:
                num_calls = scheduler.counts_total[execution_id][time_scale][dependency]
                logger.debug('{0} has reached {1} num_calls in {2}'.format(dependency, num_calls, time_scale.name))
                return num_calls == n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, dependency, n)


class AfterCall(_DependencyValidation, Condition):
    """AfterCall

    Parameters:

        dependency(node):  the node on which the Condition depends

        n(int): the number of executions of **dependency** after which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting executions of **dependency**
        (default: TimeScale.ENVIRONMENT_STATE_UPDATE)

    Satisfied when:

        - the node specified in **dependency** has executed at least n+1 times
          within one unit of time at the `TimeScale` specified by **time_scale**.

    """
    def __init__(self, dependency, n, time_scale=TimeScale.ENVIRONMENT_STATE_UPDATE):
        def func(dependency, n, scheduler=None, execution_id=None):
            try:
                num_calls = scheduler.counts_total[execution_id][time_scale][dependency]
                logger.debug('{0} has reached {1} num_calls in {2}'.format(dependency, num_calls, time_scale.name))
                return num_calls > n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, dependency, n)


class AfterNCalls(_DependencyValidation, Condition):
    """AfterNCalls

    Parameters:

        dependency(node):  the node on which the Condition depends

        n(int): the number of executions of **dependency** after which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting executions of **dependency**
        (default: TimeScale.ENVIRONMENT_STATE_UPDATE)

    Satisfied when:

        - the node specified in **dependency** has executed at least n times
          within one unit of time at the `TimeScale` specified by **time_scale**.

    """
    def __init__(self, dependency, n, time_scale=TimeScale.ENVIRONMENT_STATE_UPDATE):
        self.time_scale = time_scale

        def func(dependency, n, scheduler=None, execution_id=None):
            try:
                num_calls = scheduler.counts_total[execution_id][time_scale][dependency]
                logger.debug('{0} has reached {1} num_calls in {2}'.format(dependency, num_calls, time_scale.name))
                return num_calls >= n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, dependency, n)


class AfterNCallsCombined(_DependencyValidation, Condition):
    """AfterNCallsCombined

    Parameters:

        *nodes(nodes):  one or more nodes on which the Condition depends

        n(int): the number of combined executions of all nodes specified in **dependencies** after which the
        Condition is satisfied (default: None)

        time_scale(TimeScale): the TimeScale used as basis for counting executions of **dependency**
        (default: TimeScale.ENVIRONMENT_STATE_UPDATE)


    Satisfied when:

        - there have been at least n+1 executions among all of the nodes specified in **dependencies**
          within one unit of time at the `TimeScale` specified by **time_scale**.

    """
    def __init__(self, *dependencies, n=None, time_scale=TimeScale.ENVIRONMENT_STATE_UPDATE):
        logger.debug('{0} args: deps {1}, n {2}, ts {3}'.format(type(self).__name__, dependencies, n, time_scale))

        def func(*dependencies, n=None, scheduler=None, execution_id=None):
            if n is None:
                raise ConditionError(f'{type(self).__name__}: required keyword argument n is None.')
            count_sum = 0
            for d in dependencies:
                try:
                    count_sum += scheduler.counts_total[execution_id][time_scale][d]
                    logger.debug('{0} has reached {1} num_calls in {2}'.
                                 format(d, scheduler.counts_total[execution_id][time_scale][d], time_scale.name))
                except AttributeError as e:
                    raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

            return count_sum >= n
        super().__init__(func, *dependencies, n=n)


class EveryNCalls(_DependencyValidation, Condition):
    """EveryNCalls

    Parameters:

        dependency(node):  the node on which the Condition depends

        n(int): the frequency of executions of **dependency** at which the Condition is satisfied


    Satisfied when:

        - the node specified in **dependency** has executed at least n times since the last time the
          Condition's owner executed.

        COMMENT:
            JDC: IS THE FOLLOWING TRUE OF ALL OF THE ABOVE AS WELL??
            K: No, EveryNCalls is tricky in how it needs to be implemented, because it's in a sense
                tracking the relative frequency of calls between two objects. So the idea is that the scheduler
                tracks how many executions of a node are "useable" by other nodes for EveryNCalls conditions.
                So, suppose you had something like add_condition(B, All(AfterNCalls(A, 10), EveryNCalls(A, 2))). You
                would want the AAB pattern to start happening after A has run 10 times. Useable counts allows B to see
                whether A has run enough times for it to run, and then B spends its "useable executions" of A. Then,
                A must run two more times for B to run again. If you didn't reset the counts of A useable by B
                to 0 (question below) when B runs, then in the
                above case B would continue to run every pass for the next 4 passes, because it would see an additional
                8 executions of A it could spend to execute.
            JDC: IS THIS A FORM OF MODULO?  IF SO, WOULD IT BE EASIER TO EXPLAIN IN THAT FORM?
        COMMENT

    Notes:

        - scheduler's count of each other node that is "useable" by the node is reset to 0 when the
          node runs

    """
    def __init__(self, dependency, n):
        def func(dependency, n, scheduler=None, execution_id=None):
            try:
                num_calls = scheduler.counts_useable[execution_id][dependency][self.owner]
                logger.debug('{0} has reached {1} num_calls'.format(dependency, num_calls))
                return num_calls >= n
            except AttributeError as e:
                raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')

        super().__init__(func, dependency, n)


class JustRan(_DependencyValidation, Condition):
    """JustRan

    Parameters:

        dependency(node):  the node on which the Condition depends

    Satisfied when:

        - the node specified in **dependency** executed in the previous `CONSIDERATION_SET_EXECUTION`.

    Notes:

        - This Condition can transcend divisions between `TimeScales <TimeScale>`.
          For example, if A runs in the final `CONSIDERATION_SET_EXECUTION` of an `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>`,
          JustRan(A) is satisfied at the beginning of the next `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>`.

    """
    def __init__(self, dependency):
        def func(dependency, scheduler=None, execution_id=None):
            logger.debug(f'checking if {dependency} in previous execution step set')
            try:
                return dependency in scheduler.execution_list[execution_id][-1]
            except TypeError:
                return dependency == scheduler.execution_list[execution_id][-1]
        super().__init__(func, dependency)


class AllHaveRun(_DependencyValidation, Condition):
    """AllHaveRun

    Parameters:

        *nodes(nodes):  an iterable of nodes on which the Condition depends

        time_scale(TimeScale): the TimeScale used as basis for counting executions of **dependency**
        (default: TimeScale.ENVIRONMENT_STATE_UPDATE)

    Satisfied when:

        - all of the nodes specified in **dependencies** have executed at least once
          within one unit of time at the `TimeScale` specified by **time_scale**.

    """
    def __init__(self, *dependencies, time_scale=TimeScale.ENVIRONMENT_STATE_UPDATE):
        self.time_scale = time_scale

        def func(*dependencies, scheduler=None, execution_id=None):
            if len(dependencies) == 0:
                dependencies = scheduler.nodes
            for d in dependencies:
                try:
                    if scheduler.counts_total[execution_id][time_scale][d] < 1:
                        return False
                except AttributeError as e:
                    raise ConditionError(f'{type(self).__name__}: scheduler must be supplied to is_satisfied: {e}.')
                except KeyError as e:
                    raise ConditionError(
                        f'{type(self).__name__}: execution_id ({execution_id}) must both be specified, and '
                        f'execution_id must be in scheduler.counts_total (scheduler: {scheduler}): {e}.')
            return True
        super().__init__(func, *dependencies)


class WhenFinished(_DependencyValidation, Condition):
    """WhenFinished

    Parameters:

        dependency(node):  the node on which the Condition depends

    Satisfied when:

        - the `is_finished` methods of the node specified in **dependencies** returns `True`.

    Notes:

        - This is a dynamic Condition: Each node is responsible for managing its finished status on its
          own, which can occur independently of the execution of other nodes.  Therefore the satisfaction of
          this Condition) can vary arbitrarily in time.

        - The is_finished method is called with `execution_id` as its \
          sole positional argument

    """
    def __init__(self, dependency):
        def func(dependency, execution_id=None):
            try:
                return dependency.is_finished(execution_id)
            except AttributeError as e:
                raise ConditionError(f'WhenFinished: Unsupported dependency type: {type(dependency)}; ({e}).')

        super().__init__(func, dependency)


class WhenFinishedAny(_DependencyValidation, Condition):
    """WhenFinishedAny

    Parameters:

        *nodes(nodes):  zero or more nodes on which the Condition depends

    Satisfied when:

        - the `is_finished` methods of any nodes specified in **dependencies** returns `True`.

    Notes:

        - This is a convenience class; WhenFinishedAny(A, B, C) is equivalent to
          Any(WhenFinished(A), WhenFinished(B), WhenFinished(C)).
          If no nodes are specified, the condition will default to checking all of scheduler's nodes.

        - This is a dynamic Condition: Each node is responsible for managing its finished status on its
          own, which can occur independently of the execution of other nodes.  Therefore the satisfaction of
          this Condition) can vary arbitrarily in time.

        - The is_finished method is called with `execution_id` as its \
          sole positional argument

    """
    def __init__(self, *dependencies):
        def func(*dependencies, scheduler=None, execution_id=None):
            if len(dependencies) == 0:
                dependencies = scheduler.nodes
            for d in dependencies:
                try:
                    if d.is_finished(execution_id):
                        return True
                except AttributeError as e:
                    raise ConditionError(f'WhenFinishedAny: Unsupported dependency type: {type(d)}; ({e}).')
            return False

        super().__init__(func, *dependencies)


class WhenFinishedAll(_DependencyValidation, Condition):
    """WhenFinishedAll

    Parameters:

        *nodes(nodes):  zero or more nodes on which the Condition depends

    Satisfied when:

        - the `is_finished` methods of all nodes specified in **dependencies** return `True`.

    Notes:

        - This is a convenience class; WhenFinishedAny(A, B, C) is equivalent to
          All(WhenFinished(A), WhenFinished(B), WhenFinished(C)).
          If no nodes are specified, the condition will default to checking all of scheduler's nodes.

        - This is a dynamic Condition: Each node is responsible for managing its finished status on its
          own, which can occur independently of the execution of other nodes.  Therefore the satisfaction of
          this Condition) can vary arbitrarily in time.

        - The is_finished method is called with `execution_id` as its \
          sole positional argument

    """
    def __init__(self, *dependencies):
        def func(*dependencies, scheduler=None, execution_id=None):
            if len(dependencies) == 0:
                dependencies = scheduler.nodes
            for d in dependencies:
                try:
                    if not d.is_finished(execution_id):
                        return False
                except AttributeError as e:
                    raise ConditionError(f'WhenFinishedAll: Unsupported dependency type: {type(d)}; ({e})')
            return True

        super().__init__(func, *dependencies)


######################################################################
# Convenience Conditions
######################################################################


class AtEnvironmentStateUpdateStart(AtPass):
    """AtEnvironmentStateUpdateStart

    Satisfied when:

        - at the beginning of an `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>`

    Notes:

        - identical to `AtPass(0) <AtPass>`
    """
    def __init__(self):
        super().__init__(0)

    def __str__(self):
        return '{0}()'.format(self.__class__.__name__)


class AtEnvironmentStateUpdateNStart(All):
    """AtEnvironmentStateUpdateNStart

    Parameters:

        n(int): the `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>` on which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>`\\ s
        (default: TimeScale.ENVIRONMENT_SEQUENCE)

    Satisfied when:

        - on `PASS` 0 of the specified `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>` counted using 'TimeScale`

    Notes:

        - identical to All(AtPass(0), AtEnvironmentStateUpdate(n, time_scale))

    """
    def __init__(self, n, time_scale=TimeScale.ENVIRONMENT_SEQUENCE):
        return super().__init__(AtPass(0), AtEnvironmentStateUpdate(n, time_scale))


class AtEnvironmentSequenceStart(AtEnvironmentStateUpdate):
    """AtEnvironmentSequenceStart

    Satisfied when:

        - at the beginning of an `ENVIRONMENT_SEQUENCE`

    Notes:

        - identical to `AtEnvironmentStateUpdate(0) <AtEnvironmentStateUpdate>`
    """
    def __init__(self):
        super().__init__(0, time_scale=TimeScale.ENVIRONMENT_SEQUENCE)

    def __str__(self):
        return '{0}()'.format(self.__class__.__name__)


class AtEnvironmentSequenceNStart(All):
    """AtEnvironmentSequenceNStart

    Parameters:

        n(int): the `ENVIRONMENT_SEQUENCE` on which the Condition is satisfied

    Satisfied when:

        - on `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>` 0 of the specified `ENVIRONMENT_SEQUENCE` counted using 'TimeScale`

    Notes:

        - identical to `All(AtEnvironmentStateUpdate(0), AtEnvironmentSequence(n))`

    """
    def __init__(self, n):
        return super().__init__(AtEnvironmentStateUpdate(0), AtEnvironmentSequence(n))


class Threshold(_DependencyValidation, Condition):
    """Threshold

    Attributes:

        dependency
            the node on which the Condition depends

        parameter
            the name of the parameter of **dependency** whose value is
            to be compared to **threshold**

        threshold
            the fixed value compared to the value of the **parameter**

        comparator
            the string comparison operator determining the direction or
            type of comparison of the value of the **parameter**
            relative to **threshold**

        indices
            if specified, a series of indices that reach the desired
            number given an iterable value for **parameter**

        atol
            absolute tolerance for the comparison

        rtol
            relative tolerance (to **threshold**) for the comparison

        custom_parameter_getter
            if specified, a function that returns the value of
            **parameter** for **dependency**; to support class
            structures other than <**dependency**>.<**parameter**>
            without subclassing

        custom_parameter_validator
            if specified, a function that throws an exception if there
            is no **parameter** for **dependency**; to support class
            structures other than <**dependency**>.<**parameter**>
            without subclassing

    Satisfied when:

        The comparison between the value of the **parameter** and
        **threshold** using **comparator** is true. If **comparator** is
        an equality (==, !=), the comparison will be considered equal
        within tolerances **atol** and **rtol**.

    Notes:

        The comparison must be done with scalars. If the value of
        **parameter** contains more than one item, **indices** must be
        specified.
    """

    def __init__(
        self,
        dependency,
        parameter,
        threshold,
        comparator,
        indices=None,
        atol=0,
        rtol=0,
        custom_parameter_getter=None,
        custom_parameter_validator=None,
    ):
        self.validate_parameter(dependency, parameter, custom_parameter_validator)

        if comparator not in comparison_operators:
            raise ConditionError(f'Operator must be one of {list(comparison_operators.keys())}')

        if atol != 0 or rtol != 0 and comparator in {'<', '<=', '>', '>='}:
            warnings.warn('Tolerances for inequality comparators are ignored')

        if isinstance(indices, TimeScale):
            indices = [indices.value]
        elif indices is not None and not isinstance(indices, collections.abc.Iterable):
            indices = [indices]

        def func(threshold, comparator, indices, atol, rtol, execution_id):
            param_value = self.get_parameter_value(execution_id)
            if indices is not None:
                for i in indices:
                    param_value = param_value[i]

            param_value = float(param_value)

            if comparator == '==':
                return np.isclose(param_value, threshold, atol=atol, rtol=rtol)
            elif comparator == '!=':
                return not np.isclose(param_value, threshold, atol=atol, rtol=rtol)
            else:
                return comparison_operators[comparator](param_value, threshold)

        super().__init__(
            func,
            dependency=dependency,
            parameter=parameter,
            threshold=threshold,
            comparator=comparator,
            indices=indices,
            atol=atol,
            rtol=rtol,
            custom_parameter_getter=custom_parameter_getter,
        )

    def get_parameter_value(self, execution_id=None):
        if self.custom_parameter_getter is not None:
            return call_with_pruned_args(
                self.custom_parameter_getter,
                self.dependency,
                self.parameter,
                execution_id=execution_id
            )
        else:
            return getattr(self.dependency, self.parameter)

    def validate_parameter(self, dependency, parameter, custom_parameter_validator=None):
        if custom_parameter_validator is not None:
            custom_parameter_validator(dependency, parameter)
        else:
            if not hasattr(dependency, parameter):
                raise ConditionError(f'{dependency} has no {parameter} attribute')
