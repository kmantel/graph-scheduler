"""

Overview
--------

A Scheduler is used to generate the order in which the nodes of a graph are
executed. By default, a Scheduler executes nodes in an order determined by the pattern of edges among the nodes in the graph, with each node executed once
per `PASS` through the graph. For example, in a ``graph`` in which a node *A* projects to a node
*B* that projects to a node *C*, *A* will execute first followed by *B*, and then *C* in each `PASS` through the
graph. However, a Scheduler can be used to implement more complex patterns of execution, by specifying
`Conditions <Condition>` that determine when and how many times individual nodes execute, and whether and how
this depends on the execution of other nodes. Any executable node in a graph can be assigned a
Condition, and Conditions can be combined in arbitrary ways to generate any pattern of execution of the nodes
in a graph that is logically possible.

.. _Scheduler_Creation:

Creating a Scheduler
--------------------

When creating a Scheduler explicitly, the set of nodes
to be executed and their order must be specified in the Scheduler's constructor using one the following:

* a *graph specification dictionary* in the **graph** argument -
  each entry of the dictionary must be a node of a graph, and the value of each entry must be a set of
  zero or more nodes that project directly to the key.  The graph must be acyclic; an error is generated if any
  cycles (e.g., recurrent dependencies) are detected.  The Scheduler computes a `toposort` from the graph that is
  used as the default order of executions, subject to any Conditions that have been specified
  (see `below <Scheduler_Algorithm>`).

Conditions can be added to a Scheduler when it is created by specifying a `ConditionSet` (a set of
`Conditions <Condition>`) in the **conditions** argument of its constructor.  Individual Conditions and/or
ConditionSets can also be added after the Scheduler has been created, using its `add_condition` and
`add_condition_set` methods, respectively.

.. _Scheduler_Algorithm:

Algorithm
---------

.. _Consideration_Set:

When a Scheduler is created, it constructs a `consideration_queue`:  a list of ``consideration sets``
that defines the order in which nodes are eligible to be executed.  This is based on the dependencies specified in the graph
specification provided in the Scheduler's constructor.  Each ``consideration_set``
is a set of nodes that are eligible to execute at the same time/`CONSIDERATION_SET_EXECUTION` (i.e.,
that appear at the same "depth" in a sequence of dependencies, and among which there are no dependencies).  The first
``consideration_set`` consists of only origin nodes. The second consists of all nodes
that receive edges from the nodes in the first ``consideration_set``.
The third consists of nodes that receive edges from nodes in the first two ``consideration sets``,
and so forth.  When the Scheduler is run, it uses the `consideration_queue` to determine which
nodes are eligible to execute in each `CONSIDERATION_SET_EXECUTION` of a `PASS`, and then evaluates the `Condition <Condition>`
associated with each node in the current ``consideration_set`` to determine which should
actually be assigned for execution.

Pseudocode::

    consideration_queue <- list(toposort(graph))

    reset TimeScale.ENVIRONMENT_STATE_UPDATE counters
    while TimeScale.ENVIRONMENT_STATE_UPDATE are not satisfied
    and TimeScale.ENVIRONMENT_SEQUENCE termination conditions are not satisfied:
        reset TimeScale.PASS counters
        cur_index <- 0

        while TimeScale.ENVIRONMENT_STATE_UPDATE termination conditions are not satisfied
        and TimeScale.ENVIRONMENT_SEQUENCE termination conditions are not satisfied
        and cur_index < len(consideration_queue):

            cur_consideration_set <- consideration_queue[cur_index]
            do:
                cur_consideration_set_has_changed <- False
                for cur_node in cur_consideration_set:
                    if  cur_node not in cur_consideration_set_execution
                        and cur_node`s Condition is satisfied:

                        cur_consideration_set_has_changed <- True
                        add cur_node to cur_consideration_set_execution
                        increment execution and time counters
            while cur_consideration_set_has_changed

            if cur_consideration_set_execution is not empty or absolute time conditions are used:
                yield cur_consideration_set_execution

            increment cur_index
            increment time counters

        if all execution sets yielded were empty:
            yield an empty execution set

.. _Scheduler_Execution:

Execution
---------

.. note::
    This section covers normal scheduler execution
    (`Scheduler.mode = SchedulingMode.STANDARD`). See
    `Scheduler_Exact_Time` below for a description of
    `exact time mode <SchedulingMode.EXACT_TIME>`.

When a Scheduler is run, it provides a set of nodes that should be run next, based on their dependencies in the
graph specification, and any `Conditions <Condition>`, specified in the Scheduler's
constructor. For each call to the `run <Scheduler.run>` method, the Scheduler sequentially evaluates its
`consideration sets <Consideration_Set>` in their order in the `consideration_queue`.  For each set, it determines
which nodes in the set are allowed to execute, based on whether their associated `Condition <Condition>` has
been met. Any node that does not have a `Condition` explicitly specified is assigned a Condition that causes it
to be executed whenever it is `under consideration <Scheduler_Algorithm>` and all its structural parents have been
executed at least once since the node's last execution. All of the nodes within a `consideration_set
<consideration_set>` that are allowed to execute comprise a `CONSIDERATION_SET_EXECUTION` of execution. These nodes are considered
as executing simultaneously.

.. note::
    The ordering of the nodes specified within a `CONSIDERATION_SET_EXECUTION` is arbitrary (and is irrelevant, as there are no
    graph dependencies among nodes within the same ``consideration_set``). However,
    the execution of a node within a `CONSIDERATION_SET_EXECUTION` may trigger the execution of another node within its
    ``consideration_set``, as in the example below::

           C
          ↗ ↖
         A   B

        scheduler.add_condition(B, graph_scheduler.EveryNCalls(A, 2))
        scheduler.add_condition(C, graph_scheduler.EveryNCalls(B, 1))

        execution sets: [{A}, {A, B}, {C}, ...]

    Since there are no graph dependencies between `A` and `B`, they may execute in the same `CONSIDERATION_SET_EXECUTION`. Morever,
    `A` and `B` are in the same ``consideration_set``. Since `B` is specified to run every two
    times `A` runs, `A`'s second execution in the second `CONSIDERATION_SET_EXECUTION` allows `B` to run within that `CONSIDERATION_SET_EXECUTION`,
    rather than waiting for the next `PASS`.

For each `CONSIDERATION_SET_EXECUTION`, the Scheduler evaluates whether any specified
`termination Conditions <Scheduler_Termination_Conditions>` have been met, and terminates if so.  Otherwise,
it returns the set of nodes that should be executed in the current `CONSIDERATION_SET_EXECUTION`. Each subsequent call to the
`run <Scheduler.run>` method returns the set of nodes in the following `CONSIDERATION_SET_EXECUTION`.

Processing of all of the `consideration_sets <consideration_set>` in the `consideration_queue` constitutes a `PASS` of
execution, over which every node in the graph has been considered for execution. Subsequent calls to the
`run <Scheduler.run>` method cycle back through the `consideration_queue`, evaluating the `consideration_sets
<consideration_set>` in the same order as previously. Different subsets of nodes within the same `consideration_set
<consideration_set>` may be assigned to execute on each `PASS`, since different Conditions may be satisfied.

The Scheduler continues to make `PASS`\\ es through the `consideration_queue` until a `termination Condition
<Scheduler_Termination_Conditions>` is satisfied. If no termination Conditions are specified, by default the Scheduler
terminates an `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>` when every node has been specified for execution at least once
(corresponding to the `AllHaveRun` Condition).  However, other termination Conditions can be specified,
that may cause the Scheduler to terminate an `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>` earlier or later (e.g., when the Condition
for a particular node or set of nodes is met).

.. _Scheduler_Termination_Conditions:

*Termination Conditions*
~~~~~~~~~~~~~~~~~~~~~~~~

Termination conditions are `Conditions <Condition>` that specify when the open-ended units of time - `ENVIRONMENT_STATE_UPDATE
<TimeScale.ENVIRONMENT_STATE_UPDATE>` and `ENVIRONMENT_SEQUENCE` - have ended.  By default, the termination condition for an `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>` is
`AllHaveRun`, which is satisfied when all nodes have run at least once within the environment state update, and the termination
condition for an `ENVIRONMENT_SEQUENCE` is when all of its constituent environment state updates have terminated.


.. _Scheduler_Absolute_Time:

Absolute Time
-------------

The scheduler supports scheduling of models of real-time systems in
modes, both of which involve mapping real-time values to
`Time`. The default mode is is most compatible with standard
scheduling, but can cause some unexpected behavior in certain cases
because it is inexact. The consideration queue remains intact, but as a
result, actions specified by fixed times of absolute-time-based
conditions (`start <TimeInterval.start>` and `end <TimeInterval.end>` of
`TimeInterval`, and `t` of `TimeTermination`) may not occur at exactly
the time specified. The simplest example of this situation involves a
linear graph with two nodes::

    >>> import graph_scheduler

    >>> graph = {'A': set(), 'B': {'A'}}
    >>> scheduler = graph_scheduler.Scheduler(graph=graph)

    >>> scheduler.add_condition('A', graph_scheduler.TimeInterval(start=10))
    >>> scheduler.add_condition('B', graph_scheduler.TimeInterval(start=10))

In standard mode, **A** and **B** are in different consideration sets,
and so can never execute at the same time. At most one of **A** and
**B** will start exactly at t=10ms, with the other starting at its next
consideration after. There are many of these examples, and while it may
be solveable in some cases, it is not a simple problem. So,
`Exact Time Mode <Scheduler_Exact_Time>` exists as an alternative
option for these cases, though it comes with its own drawbacks.

.. note::
    Due to issues with floating-point precision, absolute time values in
    conditions and `Time` are limited to 8 decimal points. If more
    precision is needed, use
    `fractions <https://docs.python.org/3/library/fractions.html>`_,
    where possible, or smaller units (e.g. microseconds instead of
    milliseconds).


.. _Scheduler_Exact_Time:

Exact Time Mode
~~~~~~~~~~~~~~~

When `Scheduler.mode` is `SchedulingMode.EXACT_TIME`, the scheduler is
capable of handling examples like the one
`above <Scheduler_Absolute_Time>`. In this mode, all nodes in the
scheduler's graph become members of the same consideration set, and may
be executed at the same time for every consideration set execution,
subject to the conditions specified. As a result, the guarantees in
`standard scheduling <Scheduler_Execution>` may not apply - that is,
that all parent nodes get a chance to execute before their children, and
that there exist no data dependencies (edges) between nodes in the
same execution set. In exact time mode, all nodes will be in one
[unordered] execution set. An ordering may be inferred by the original
graph, however, using the `indices in the original consideration queue\
<graph_scheduler.scheduler.Scheduler.consideration_queue_indices>`.
Additionally, non-absolute conditions like
`EveryNCalls` may behave unexpectedly in some cases.

Examples
--------

Please see `Condition` for a list of all supported Conditions and their behavior.

* Basic phasing in a linear process::

    >>> import graph_scheduler

    >>> graph = {'A': set(), 'B': {'A'}, 'C': {'B'}}
    >>> scheduler = graph_scheduler.Scheduler(graph=graph)

    >>> # implicit condition of Always for A
    >>> scheduler.add_condition('B', graph_scheduler.EveryNCalls('A', 2))
    >>> scheduler.add_condition('C', graph_scheduler.EveryNCalls('B', 3))

    >>> # implicit AllHaveRun Termination condition
    >>> execution_sequence = list(scheduler.run())
    >>> execution_sequence
    [{'A'}, {'A'}, {'B'}, {'A'}, {'A'}, {'B'}, {'A'}, {'A'}, {'B'}, {'C'}]

* Alternate basic phasing in a linear process::

    >>> graph = {'A': set(), 'B': {'A'}}
    >>> scheduler = graph_scheduler.Scheduler(graph=graph)

    >>> scheduler.add_condition(
    ...     'A',
    ...     graph_scheduler.Any(
    ...         graph_scheduler.AtPass(0),
    ...         graph_scheduler.EveryNCalls('B', 2)
    ...     )
    ... )

    >>> scheduler.add_condition(
    ...     'B',
    ...     graph_scheduler.Any(
    ...         graph_scheduler.EveryNCalls('A', 1),
    ...         graph_scheduler.EveryNCalls('B', 1)
    ...     )
    ... )
    >>> termination_conds = {
    ...     graph_scheduler.TimeScale.ENVIRONMENT_STATE_UPDATE: graph_scheduler.AfterNCalls('B', 4, time_scale=graph_scheduler.TimeScale.ENVIRONMENT_STATE_UPDATE)
    ... }
    >>> execution_sequence = list(scheduler.run(termination_conds=termination_conds))
    >>> execution_sequence
    [{'A'}, {'B'}, {'B'}, {'A'}, {'B'}, {'B'}]

* Basic phasing in two processes::

    >>> graph = {'A': set(), 'B': set(), 'C': {'A', 'B'}}
    >>> scheduler = graph_scheduler.Scheduler(graph=graph)

    >>> scheduler.add_condition('A', graph_scheduler.EveryNPasses(1))
    >>> scheduler.add_condition('B', graph_scheduler.EveryNCalls('A', 2))
    >>> scheduler.add_condition(
    ...     'C',
    ...     graph_scheduler.Any(
    ...         graph_scheduler.AfterNCalls('A', 3),
    ...         graph_scheduler.AfterNCalls('B', 3)
    ...     )
    ... )
    >>> termination_conds = {
    ...     graph_scheduler.TimeScale.ENVIRONMENT_STATE_UPDATE: graph_scheduler.AfterNCalls('C', 4, time_scale=graph_scheduler.TimeScale.ENVIRONMENT_STATE_UPDATE)
    ... }
    >>> execution_sequence = list(scheduler.run(termination_conds=termination_conds))
    >>> execution_sequence # doctest: +SKIP
    [{'A'}, {'A', 'B'}, {'A'}, {'C'}, {'A', 'B'}, {'C'}, {'A'}, {'C'}, {'A', 'B'}, {'C'}]

.. _Scheduler_Class_Reference

Class Reference
===============

"""

import copy
import datetime
import enum
import fractions
import logging
import typing

import numpy as np
import pint
from toposort import toposort

from graph_scheduler import _unit_registry
from graph_scheduler.condition import (
    All, AllHaveRun, Always, Condition, ConditionSet, EveryNCalls, Never,
    _parse_absolute_unit, _quantity_as_integer,
)
from graph_scheduler.time import Clock, TimeScale

__all__ = [
    'Scheduler', 'SchedulerError', 'SchedulingMode',
]

logger = logging.getLogger(__name__)

default_termination_conds = {
    TimeScale.ENVIRONMENT_SEQUENCE: Never(),
    TimeScale.ENVIRONMENT_STATE_UPDATE: AllHaveRun(),
}


class SchedulingMode(enum.Enum):
    """
        Attributes:
            STANDARD
                `Standard Mode <Scheduler_Execution>`

            EXACT_TIME
                `Exact time Mode <Scheduler_Exact_Time>`
    """
    STANDARD = enum.auto()
    EXACT_TIME = enum.auto()


class SchedulerError(Exception):

    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class Scheduler:
    """Generates an order of execution for nodes in a graph or graph
    specification dictionary, possibly determined by a set of `Conditions <Condition>`.

    Arguments
    ---------

    graph : Dict[object: set(object)], `networkx.DiGraph`
        a graph specification dictionary - each entry of the dictionary must be an object,
        and the value of each entry must be a set of zero or more objects that project directly to the key.

    conditions  : ConditionSet
        set of `Conditions <Condition>` that specify when individual nodes in **graph**
        execute and any dependencies among them.

    mode : SchedulingMode[STANDARD|EXACT_TIME] : SchedulingMode.STANDARD
        sets the mode of scheduling: `standard <Scheduler_Execution>` or
        `exact time <Scheduler_Exact_Time>`

    default_absolute_time_unit : `pint.Quantity` : ``1ms``
        if not otherwise determined by any absolute **conditions**,
        specifies the absolute duration of a `CONSIDERATION_SET_EXECUTION`

    Attributes
    ----------

    conditions : ConditionSet
        the set of Conditions the Scheduler uses when running

    default_execution_id : object
        the execution_id to use if none is supplied to `run`; a unique
        identifier to allow multiple schedulings independently. Must be
        hashable.

    execution_list : list
        the full history of consideration set executions the Scheduler has produced

    consideration_queue : list
        a list form of the Scheduler's toposort ordering of its nodes

    consideration_queue_indices : dict
        a dict mapping **nodes** to their position in the `consideration_queue`

    termination_conds : Dict[TimeScale: Condition]
        a mapping from `TimeScales <TimeScale>` to `Conditions <Condition>` that, when met, terminate the execution
        of the specified `TimeScale`. On set, update only for the
        `TimeScale`\\ s specified in the argument.

    mode
        sets the mode of scheduling: `standard <Scheduler_Execution>` or
        `exact time <Scheduler_Exact_Time>`

        :type: SchedulingMode
        :default: `SchedulingMode.STANDARD`

    default_absolute_time_unit
        if not otherwise determined by any absolute **conditions**,
        specifies the absolute duration of a `CONSIDERATION_SET_EXECUTION`

        :type: `pint.Quantity`
        :default: ``1ms``

    """
    def __init__(
        self,
        graph,
        conditions=None,
        termination_conds=None,
        default_execution_id=None,
        mode: SchedulingMode = SchedulingMode.STANDARD,
        default_absolute_time_unit: typing.Union[str, pint.Quantity] = 1 * _unit_registry.ms,
        **kwargs
    ):
        """
        :param self:
        :param conditions: (ConditionSet) - a :keyword:`ConditionSet` to be scheduled
        """
        self.conditions = ConditionSet(conditions)

        # the consideration queue is the ordered list of sets of nodes in the graph, by the
        # order in which they should be checked to ensure that all parents have a chance to run before their children
        self.consideration_queue = []
        if termination_conds is None:
            termination_conds = default_termination_conds.copy()
        else:
            termination_conds = {**default_termination_conds, **termination_conds}
        self.default_termination_conds = Scheduler._parse_termination_conditions(termination_conds)
        self._termination_conds = self.default_termination_conds.copy()

        self.cycle_nodes = set()
        self.mode = mode
        self.default_absolute_time_unit = _parse_absolute_unit(default_absolute_time_unit)

        if graph is not None:
            try:
                # networkx graph
                self.dependency_dict = {
                    child: set(parents.keys())
                    for child, parents in graph.succ.items()
                }
            except AttributeError:
                self.dependency_dict = graph
            self.consideration_queue = list(toposort(self.dependency_dict))
            self.nodes = []
            for consideration_set in self.consideration_queue:
                for node in consideration_set:
                    self.nodes.append(node)
        else:
            raise SchedulerError(
                'Must instantiate a Scheduler with a graph dependency dict or a networkx.DiGraph'
            )

        self._generate_consideration_queue_indices()

        self.default_execution_id = default_execution_id
        # stores the in order list of self.run's yielded outputs
        self.execution_list = {self.default_execution_id: []}
        self.execution_timestamps = {self.default_execution_id: []}
        self.clocks = {self.default_execution_id: Clock()}
        self.counts_total = {}
        self.counts_useable = {}
        self._init_counts(execution_id=self.default_execution_id)
        self.date_creation = datetime.datetime.now()
        self.date_last_run_end = None

    def _generate_consideration_queue_indices(self):
        self.consideration_queue_indices = {}
        for i, cs in enumerate(self.consideration_queue):
            self.consideration_queue_indices.update({
                n: i for n in cs
            })

    def _init_counts(self, execution_id, base_execution_id=NotImplemented):
        """
            Attributes
            ----------

                execution_id
                    the execution_id to initialize counts for

                base_execution_id
                    if specified, the counts for execution_id will be copied from the counts of base_execution_id
                    default : NotImplemented
        """
        # all counts are divided by execution_id, which provides a context for the scheduler's execution, so that
        # it can be reused in multiple contexts

        # stores total the number of occurrences of a node through the time scale
        # i.e. the number of times node has ran/been queued to run in a environment state update
        if execution_id not in self.counts_total:
            if base_execution_id is not NotImplemented:
                if base_execution_id not in self.counts_total:
                    raise SchedulerError('execution_id {0} not in {1}.counts_total'.format(base_execution_id, self))

                self.counts_total[execution_id] = {
                    ts: {n: self.counts_total[base_execution_id][ts][n] for n in self.nodes} for ts in TimeScale
                }
            else:
                self.counts_total[execution_id] = {
                    ts: {n: 0 for n in self.nodes} for ts in TimeScale
                }

        # counts_useable is a dictionary intended to store the number of available "instances" of a certain node that
        # are available to expend in order to satisfy conditions such as "run B every two times A runs"
        # specifically, counts_useable[a][b] = n indicates that there are n uses of a that are available for b to expend
        # so, in the previous example B would check to see if counts_useable[A][B] >= 2, in which case B can run
        # then, counts_useable[a][b] would be reset to 0, even if it was greater than 2
        if execution_id not in self.counts_useable:
            if base_execution_id is not NotImplemented:
                if base_execution_id not in self.counts_useable:
                    raise SchedulerError('execution_id {0} not in {1}.counts_useable'.format(base_execution_id, self))

                self.counts_useable[execution_id] = {
                    node: {n: self.counts_useable[base_execution_id][node][n] for n in self.nodes} for node in self.nodes
                }
            else:
                self.counts_useable[execution_id] = {
                    node: {n: 0 for n in self.nodes} for node in self.nodes
                }

        if execution_id not in self.execution_list:
            if base_execution_id is not NotImplemented:
                if base_execution_id not in self.execution_list:
                    raise SchedulerError('execution_id {0} not in {1}.execution_list'.format(base_execution_id, self))

                self.execution_list[execution_id] = list(self.execution_list[base_execution_id])
            else:
                self.execution_list[execution_id] = []

        self._init_clock(execution_id, base_execution_id)

    def _init_clock(self, execution_id, base_execution_id=NotImplemented):
        # instantiate new Clock for this execution_id if necessary
        # currently does not work with base_execution_id
        if execution_id not in self.clocks:
            if base_execution_id is not NotImplemented:
                if base_execution_id not in self.clocks:
                    raise SchedulerError('execution_id {0} not in {1}.clocks'.format(base_execution_id, self))

                self.clocks[execution_id] = copy.deepcopy(self.clocks[base_execution_id])
            else:
                self.clocks[execution_id] = Clock()

    def _delete_counts(self, execution_id):
        for obj in [
            self.counts_useable,
            self.counts_total,
            self.clocks,
            self.execution_list,
            self.execution_timestamps
        ]:
            try:
                del obj[execution_id]
            except KeyError:
                pass

    def _reset_counts_total(self, time_scale, execution_id):
        for ts in TimeScale:
            # only reset the values underneath the current scope
            # this works because the enum is set so that higher granularities of time have lower values
            if ts.value <= time_scale.value:
                for c in self.counts_total[execution_id][ts]:
                    self.counts_total[execution_id][ts][c] = 0

    def _reset_counts_useable(self, execution_id):
        self.counts_useable[execution_id] = {
            node: {n: 0 for n in self.nodes} for node in self.nodes
        }

    def _combine_termination_conditions(self, termination_conds):
        termination_conds = Scheduler._parse_termination_conditions(termination_conds)
        new_conds = self.termination_conds.copy()
        new_conds.update(termination_conds)

        return new_conds

    @staticmethod
    def _parse_termination_conditions(termination_conds):
        # parse string representation of TimeScale
        parsed_conds = {}
        delkeys = set()
        for scale in termination_conds:
            try:
                parsed_conds[getattr(TimeScale, scale.upper())] = termination_conds[scale]
                delkeys.add(scale)
            except (AttributeError, TypeError):
                pass

        termination_conds.update(parsed_conds)

        try:
            termination_conds = {
                k: termination_conds[k] for k in termination_conds
                if (
                    isinstance(k, TimeScale)
                    and isinstance(termination_conds[k], Condition)
                    and k not in delkeys
                )
            }
        except TypeError:
            raise TypeError('termination_conditions must be a dictionary of the form {TimeScale: Condition, ...}')
        else:
            return termination_conds

    def end_environment_sequence(self, execution_id=NotImplemented):
        """Signals that an `ENVIRONMENT_SEQUENCE` has completed

        Args:
            execution_id (optional): Defaults to `Scheduler.default_execution_id`
        """
        if execution_id is NotImplemented:
            execution_id = self.default_execution_id

        self._increment_time(TimeScale.ENVIRONMENT_SEQUENCE, execution_id)

    ################################################################################
    # Wrapper methods
    #   to allow the user to ignore the ConditionSet internals
    ################################################################################
    def __contains__(self, item):
        return self.conditions.__contains__(item)

    def add_condition(self, owner, condition):
        """
        Adds a `Condition` to the Scheduler. If **owner** already has a Condition, it is overwritten
        with the new one. If you want to add multiple conditions to a single owner, use the
        `composite Conditions <Conditions_Composite>` to accurately specify the desired behavior.

        Arguments
        ---------

        owner : ``node``
            specifies the node with which the **condition** should be associated. **condition**
            will govern the execution behavior of **owner**

        condition : Condition
            specifies the Condition, associated with the **owner** to be added to the ConditionSet.
        """
        self.conditions.add_condition(owner, condition)

    def add_condition_set(self, conditions):
        """
        Adds a set of `Conditions <Condition>` (in the form of a dict or another ConditionSet) to the Scheduler.
        Any Condition added here will overwrite an existing Condition for a given owner.
        If you want to add multiple conditions to a single owner, add a single `Composite Condition <Conditions_Composite>`
        to accurately specify the desired behavior.

        Arguments
        ---------

        conditions : dict[``node``: `Condition`], `ConditionSet`
            specifies collection of Conditions to be added to this ConditionSet,

            if a dict is provided:
                each entry should map an owner node (the node whose execution behavior will be
                governed) to a `Condition <Condition>`

        """
        self.conditions.add_condition_set(conditions)

    ################################################################################
    # Validation methods
    #   to provide the user with info if they do something odd
    ################################################################################
    def _validate_run_state(self):
        self._validate_conditions()

    def _validate_conditions(self):
        unspecified_nodes = []
        for node in self.nodes:
            if node not in self.conditions:
                dependencies = list(self.dependency_dict[node])
                if len(dependencies) == 0:
                    cond = Always()
                elif len(dependencies) == 1:
                    cond = EveryNCalls(dependencies[0], 1)
                else:
                    cond = All(*[EveryNCalls(x, 1) for x in dependencies])

                self.add_condition(node, cond)
                unspecified_nodes.append(node)
        if len(unspecified_nodes) > 0:
            logger.info(
                'These nodes have no Conditions specified, and will be scheduled with conditions: {0}'.format(
                    {node: self.conditions[node] for node in unspecified_nodes}
                )
            )

    ################################################################################
    # Run methods
    ################################################################################
    def run(
        self,
        termination_conds=None,
        execution_id=None,
        base_execution_id=None,
        skip_environment_state_update_time_increment=False,
        **kwargs,
    ):
        """
        run is a python generator, that when iterated over provides the next `CONSIDERATION_SET_EXECUTION` of
        executions at each iteration

        :param termination_conds: (dict) - a mapping from `TimeScale`\\s to `Condition`\\s that when met
               terminate the execution of the specified `TimeScale`
        """
        self._validate_run_state()

        if self.mode is SchedulingMode.EXACT_TIME:
            effective_consideration_queue = [set(self.nodes)]
        else:
            effective_consideration_queue = self.consideration_queue

        if termination_conds is None:
            termination_conds = self.termination_conds
        else:
            termination_conds = self._combine_termination_conditions(termination_conds)

        current_time = self.get_clock(execution_id).time
        is_satisfied_kwargs = {
            'scheduler': self,
            'execution_id': execution_id,
            **kwargs
        }

        in_absolute_time_mode = len(self.get_absolute_conditions(termination_conds)) > 0
        if in_absolute_time_mode:
            current_time.absolute_interval = self._get_absolute_consideration_set_execution_unit(termination_conds)
            # advance absolute clock time to first necessary time
            current_time.absolute = max(
                current_time.absolute,
                min([
                    min(c.absolute_fixed_points)
                    if len(c.absolute_fixed_points) > 0 else 0
                    for c in self.get_absolute_conditions(termination_conds).values()
                ])
            )
            # convert to interval time unit to avoid pint floating point issues
            current_time.absolute = current_time.absolute.to(current_time.absolute_interval.u)

            # .to auto converts to float even if magnitude is an int, undo
            try:
                current_time.absolute = _quantity_as_integer(current_time.absolute)
            except ValueError:
                pass

        current_time.absolute_enabled = in_absolute_time_mode

        self._init_counts(execution_id, base_execution_id)
        self._reset_counts_useable(execution_id)
        self._reset_counts_total(TimeScale.ENVIRONMENT_STATE_UPDATE, execution_id)

        while (
            not termination_conds[TimeScale.ENVIRONMENT_STATE_UPDATE].is_satisfied(**is_satisfied_kwargs)
            and not termination_conds[TimeScale.ENVIRONMENT_SEQUENCE].is_satisfied(**is_satisfied_kwargs)
        ):
            self._reset_counts_total(TimeScale.PASS, execution_id)

            execution_list_has_changed = False
            cur_index_consideration_queue = 0

            while (
                cur_index_consideration_queue < len(effective_consideration_queue)
                and not termination_conds[TimeScale.ENVIRONMENT_STATE_UPDATE].is_satisfied(**is_satisfied_kwargs)
                and not termination_conds[TimeScale.ENVIRONMENT_SEQUENCE].is_satisfied(**is_satisfied_kwargs)
            ):
                # all nodes to be added during this consideration set execution
                cur_consideration_set_execution_exec = set()
                # the current "layer/group" of nodes that MIGHT be added during this consideration set execution
                cur_consideration_set = effective_consideration_queue[cur_index_consideration_queue]

                try:
                    iter(cur_consideration_set)
                except TypeError as e:
                    raise SchedulerError('cur_consideration_set is not iterable, did you ensure that this Scheduler was instantiated with an actual toposort output for param toposort_ordering? err: {0}'.format(e))

                # do-while, on cur_consideration_set_has_changed
                # we check whether each node in the current consideration set is allowed to run,
                # and nodes can cause cascading adds within this set
                while True:
                    cur_consideration_set_has_changed = False
                    for current_node in cur_consideration_set:
                        # only add each node once during a single consideration set execution, this also serves
                        # to prevent infinitely cascading adds
                        if current_node not in cur_consideration_set_execution_exec:
                            if self.conditions.conditions[current_node].is_satisfied(**is_satisfied_kwargs):
                                cur_consideration_set_execution_exec.add(current_node)
                                execution_list_has_changed = True
                                cur_consideration_set_has_changed = True

                                for ts in TimeScale:
                                    self.counts_total[execution_id][ts][current_node] += 1
                                # current_node's node is added to the execution queue, so we now need to
                                # reset all of the counts useable by current_node's node to 0
                                for n in self.counts_useable[execution_id]:
                                    self.counts_useable[execution_id][n][current_node] = 0
                                # and increment all of the counts of current_node's node useable by other
                                # nodes by 1
                                for n in self.counts_useable[execution_id]:
                                    self.counts_useable[execution_id][current_node][n] += 1
                    # do-while condition
                    if not cur_consideration_set_has_changed:
                        break

                # add a new consideration set execution at each step in a pass, if the consideration set execution would not be empty
                if len(cur_consideration_set_execution_exec) >= 1 or in_absolute_time_mode:
                    self.execution_list[execution_id].append(cur_consideration_set_execution_exec)
                    if in_absolute_time_mode:
                        self.execution_timestamps[execution_id].append(
                            copy.copy(current_time)
                        )
                    yield self.execution_list[execution_id][-1]

                    self.get_clock(execution_id)._increment_time(TimeScale.CONSIDERATION_SET_EXECUTION)

                cur_index_consideration_queue += 1

            # if an entire pass occurs with nothing running, add an empty consideration set execution
            if not execution_list_has_changed and not in_absolute_time_mode:
                self.execution_list[execution_id].append(set())
                yield self.execution_list[execution_id][-1]

                self.get_clock(execution_id)._increment_time(TimeScale.CONSIDERATION_SET_EXECUTION)

            self.get_clock(execution_id)._increment_time(TimeScale.PASS)

        if not skip_environment_state_update_time_increment:
            self.get_clock(execution_id)._increment_time(TimeScale.ENVIRONMENT_STATE_UPDATE)

        if termination_conds[TimeScale.ENVIRONMENT_SEQUENCE].is_satisfied(**is_satisfied_kwargs):
            self.date_last_run_end = datetime.datetime.now()

        return self.execution_list[execution_id]

    def _increment_time(self, time_scale, execution_id):
        self.get_clock(execution_id)._increment_time(time_scale)

    def _get_absolute_consideration_set_execution_unit(self, termination_conds=None):
        """Computes the time length of the gap between two consideration set executions
        """
        if termination_conds is None:
            termination_conds = self.termination_conds

        # all of the units of time that must occur on a consideration set execution
        intervals = []
        for c in self.get_absolute_conditions(termination_conds).values():
            intervals.extend(c.absolute_intervals)
            if self.mode is SchedulingMode.EXACT_TIME:
                intervals.extend(c.absolute_fixed_points)

        if len(intervals) == 0:
            return self.default_absolute_time_unit
        else:
            min_time_unit = min(intervals, key=lambda x: x.u).u

            # convert all intervals into the same unit
            for i, a in enumerate(intervals):
                unit_conversion = round(np.log10((1 * a.u).to(min_time_unit).m))
                intervals[i] = (int(a.m) * 10 ** unit_conversion) * min_time_unit

            # numerator is the largest possible length of a pass
            # denominator evenly divides this length by number of consideration set executions
            numerator = np.gcd.reduce([interval.m for interval in intervals])
            if self.mode == SchedulingMode.STANDARD:
                denominator = len(self.consideration_queue)
            elif self.mode == SchedulingMode.EXACT_TIME:
                denominator = 1

            if denominator == 1:
                res = numerator
            else:
                res = fractions.Fraction(numerator=numerator, denominator=denominator)
            return res * min_time_unit

    def get_absolute_conditions(self, termination_conds=None):
        if termination_conds is None:
            termination_conds = self.termination_conds

        return {
            owner: cond
            for owner, cond
            in [*self.conditions.conditions.items(), *termination_conds.items()]
            if cond.is_absolute
        }

    def get_clock(self, execution_id):
        try:
            return self.clocks[execution_id.default_execution_id]
        except AttributeError:
            if execution_id not in self.clocks:
                self._init_clock(execution_id)
            return self.clocks[execution_id]
        except KeyError:
            self._init_clock(execution_id.default_execution_id)
            return self.clocks[execution_id.default_execution_id]

    @property
    def termination_conds(self):
        return self._termination_conds

    @termination_conds.setter
    def termination_conds(self, termination_conds):
        """Updates this Scheduler's base `termination conditions
        <Scheduler.termination_conds>`_ to be used on future `run
        <Scheduler.run>`\\ s for which termination conditions are not
        specified.

        Arguments:
            termination_conds : dict[[TimeScale, str]: Condition]
                the dictionary of termination Conditions to overwrite
                the current base
                `termination conditions <Scheduler.termination_conds>`
        """
        if termination_conds is None:
            self._termination_conds = self.default_termination_conds.copy()
        else:
            self._termination_conds = self._combine_termination_conditions(
                termination_conds
            )

    @property
    def _in_exact_time_mode(self):
        return self.mode is SchedulingMode.EXACT_TIME or len(self.consideration_queue) == 1
