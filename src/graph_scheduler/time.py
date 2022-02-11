"""

.. _Time_Overview:

Overview
--------

:doc:`Scheduler`\\ s maintain `Clock` objects to track time. The current time in \
relation to a :doc:`Scheduler` is stored in :class:`Clock.time <Time>` \
or :class:`Clock.simple_time <SimpleTime>`

"""

import copy
import enum
import functools
import keyword
import re
import types
import typing

from graph_scheduler import _unit_registry

__all__ = [
    'Clock', 'TimeScale', 'Time', 'SimpleTime', 'TimeHistoryTree', 'TimeScaleError', 'set_time_scale_alias', 'remove_time_scale_alias'
]


_time_scale_aliases = {}
_alias_docs_warning_str = """

.. note:: This documentation was modified from the original due to environment-specific TimeScale renamings. If there is any confusion, please see the original documentation at https://www.github.com/kmantel/graph-scheduler

"""


class TimeScaleError(Exception):

    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


# Time scale modes
@functools.total_ordering
class TimeScale(enum.Enum):
    """Represents divisions of time used by the `Scheduler` and `Conditions <Condition>`.

    The values of TimeScale are defined as follows (in order of increasingly coarse granularity):

    Attributes
    ----------

    CONSIDERATION_SET_EXECUTION
        the nuclear unit of time, corresponding to the execution of all nodes allowed to execute
        from a single `consideration set <Consideration_Set>` of a `Scheduler <graph_scheduler.scheduler.Scheduler>`, and which are considered to have
        executed simultaneously.

    PASS
        a full iteration through all of the consideration sets in a `Scheduler's <graph_scheduler.scheduler.Scheduler>`
        `consideration_queue`, consisting of one or more `CONSIDERATION_SET_EXECUTIONs <CONSIDERATION_SET_EXECUTION>`, over which every node
        specified to a `Scheduler <Scheduler_Creation>` is considered for execution at least once.

    ENVIRONMENT_STATE_UPDATE
        an open-ended unit of time consisting of all actions that occurs within the scope of a single
        call to `run <Scheduler.run>`

    ENVIRONMENT_SEQUENCE
        the scope of a batch of one or more `ENVIRONMENT_STATE_UPDATE
        <TimeScale.ENVIRONMENT_STATE_UPDATE>`_\\s, managed by the
        environment using the Scheduler.

    LIFE
        the scope of time since the creation of an object.
    """
    CONSIDERATION_SET_EXECUTION = 0
    PASS = 1
    ENVIRONMENT_STATE_UPDATE = 2
    ENVIRONMENT_SEQUENCE = 3
    LIFE = 4

    # ordering based on enum.OrderedEnum example
    # https://docs.python.org/3/library/enum.html#orderedenum
    # https://stackoverflow.com/a/39269589/3131666
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

    @classmethod
    def get_parent(cls, time_scale):
        """
        Returns
        -------
            the TimeScale one level wider in scope than time_scale : :class:`TimeScale`
        """
        return cls(time_scale.value + 1)

    @classmethod
    def get_child(cls, time_scale):
        """
        Returns
        -------
            the TimeScale one level smaller in scope than time_scale : :class:`TimeScale`
        """
        return cls(time_scale.value - 1)


class Clock:
    """
    Stores a history of :class:`TimeScale`\\ s that have occurred, and keep track of a \
    current `Time`. Used in relation to a :doc:`Scheduler`

    Attributes
    ----------
        history : `TimeHistoryTree`
            a root `TimeHistoryTree` associated with this Clock

        simple_time : `SimpleTime`
            the current time in simple format
    """
    def __init__(self):
        self.history = TimeHistoryTree()
        self.simple_time = SimpleTime(self.time)

    def __repr__(self):
        return 'Clock({0})'.format(self.time.__repr__())

    def _increment_time(self, time_scale):
        """
        Calls `self.history.increment_time <TimeHistoryTree.increment_time>`
        """
        self.history.increment_time(time_scale)

    def get_total_times_relative(self, query_time_scale, base_time_scale, base_index=None):
        """
        Convenience simplified wrapper for `TimeHistoryTree.get_total_times_relative`

        Arguments
        ---------
            query_time_scale : :class:`TimeScale`
                the unit of time whose number of ticks to be returned

            base_time_scale : :class:`TimeScale`
                the unit of time over which the number of **query_time_scale** ticks
                should be returned

            base_index : int
                the **base_index**\\ th **base_time_scale** over which the number of
                **query_time_scale** ticks should be returned
        Returns
        -------
            the number of query_time_scale s that have occurred during the scope \
            of the base_index'th base_time_scale : int
        """
        if base_index is None:
            base_index = self.get_time_by_time_scale(base_time_scale)

        return self.history.get_total_times_relative(
            query_time_scale,
            {base_time_scale: base_index}
        )

    def get_time_by_time_scale(self, time_scale):
        """
        Arguments
        ---------
            time_scale : :class:`TimeScale`

        Returns
        -------
            the current value of the time unit corresponding to time_scale \
            for this Clock : int
        """
        return self.time._get_by_time_scale(time_scale)

    @property
    def time(self):
        """
        the current time : `Time`
        """
        return self.history.current_time

    @property
    def previous_time(self):
        """
        the time that has occurred last : `Time`
        """
        return self.history.previous_time


class Time(types.SimpleNamespace):
    """
    Represents an instance of time, having values for each :class:`TimeScale`

    Attributes
    ----------
        life : int : 0
            the `TimeScale.LIFE` value

        environment_sequence : int : 0
            the `TimeScale.ENVIRONMENT_SEQUENCE` value

        environment_state_update : int : 0
            the `TimeScale.ENVIRONMENT_STATE_UPDATE` value

        pass_ : int : 0
            the `TimeScale.PASS` value

        consideration_set_execution : int : 0
            the `TimeScale.CONSIDERATION_SET_EXECUTION` value

        absolute : `pint.Quantity` : 0ms
            the absolute time value

        absolute_interval : `pint.Quantity` : 1ms
            the interval between units of absolute time

        absolute_time_unit_scale : `TimeScale` : TimeScale.CONSIDERATION_SET_EXECUTION
            the `TimeScale` that corresponds to an interval of absolute time

        absolute_enabled : bool : False
            whether absolute time is used for this Time object

    """
    def __init__(
        self,
        consideration_set_execution=0,
        pass_=0,
        environment_state_update=0,
        environment_sequence=0,
        life=0,
        absolute=0 * _unit_registry.ms,
        absolute_interval=1 * _unit_registry.ms,
        absolute_time_unit_scale=TimeScale.CONSIDERATION_SET_EXECUTION,
        absolute_enabled=False,
        **alias_time_values,
    ):
        time_scale_values = {}
        for ts in TimeScale:
            time_scale_values[ts] = locals()[_time_scale_to_attr_str(ts)]

        time_scale_values.update({
            ts: alias_time_values[_time_scale_aliases[ts].lower()]
            for ts in _time_scale_aliases
            if _time_scale_aliases[ts].lower() in alias_time_values
        })

        super().__init__(
            **{
                _time_scale_to_attr_str(ts): v for ts, v in time_scale_values.items()
            },
            absolute=0 * _unit_registry.ms,
            absolute_interval=1 * _unit_registry.ms,
            absolute_time_unit_scale=TimeScale.CONSIDERATION_SET_EXECUTION,
            absolute_enabled=False,
        )

    def __repr__(self):
        abs_str = f'{self.absolute}, ' if self.absolute_enabled else ''
        ts_str = self._time_repr(exclusions={TimeScale.LIFE})
        return f'Time({abs_str}{ts_str})'

    def __getitem__(self, item):
        try:
            item = TimeScale(item)
        except ValueError:
            pass

        return self._get_by_time_scale(item)

    def __setitem__(self, item, value):
        try:
            item = TimeScale(item)
        except ValueError:
            pass

        return self._set_by_time_scale(item, value)

    def _time_repr(self, exclusions=()):
        ts_strs = []

        for ts in sorted(TimeScale, reverse=True):
            if ts not in exclusions:
                if ts not in _time_scale_aliases:
                    name = ts.name.lower()
                else:
                    name = _time_scale_aliases[ts].lower()

                ts_strs.append(
                    f'{name}: {getattr(self, _time_scale_to_attr_str(ts))}'
                )

        return ', '.join(ts_strs)

    def _get_by_time_scale(self, time_scale):
        """
        Arguments
        ---------
            time_scale : :class:`TimeScale`

        Returns
        -------
            this Time's value of a TimeScale by the TimeScale enum, rather \
            than by attribute : int
        """
        return getattr(self, _time_scale_to_attr_str(time_scale))

    def _set_by_time_scale(self, time_scale, value):
        """
        Arguments
        ---------
            time_scale : :class:`TimeScale`

        Sets this Time's value of a **time_scale** by the TimeScale enum,
        rather than by attribute
        """
        setattr(self, _time_scale_to_attr_str(time_scale), value)

    def _increment_by_time_scale(self, time_scale):
        """
        Increments the value of **time_scale** in this Time by one
        """
        self._set_by_time_scale(time_scale, self._get_by_time_scale(time_scale) + 1)
        self._reset_by_time_scale(time_scale)

        if (
            self.absolute_enabled
            and time_scale is self.absolute_time_unit_scale
        ):
            self.absolute += self.absolute_interval
            if (
                self.absolute_interval < 1 * self.absolute.u
                and isinstance(self.absolute.m, float)
            ):
                # only round for floating point interval increases,
                # until pint fixes precision errors
                # see https://github.com/hgrecco/pint/issues/1263
                self.absolute = round(self.absolute, _unit_registry.precision)

    def _reset_by_time_scale(self, time_scale):
        """
        Resets all the times for the time scale scope up to **time_scale**
        e.g. _reset_by_time_scale(TimeScale.ENVIRONMENT_STATE_UPDATE) will set the values for
        TimeScale.PASS and TimeScale.CONSIDERATION_SET_EXECUTION to 0
        """
        for relative_time_scale in sorted(TimeScale):
            # this works because the enum is set so that higher granularities of time have lower values
            if relative_time_scale >= time_scale:
                continue

            self._set_by_time_scale(relative_time_scale, 0)


class SimpleTime:
    """
    A subset class of `Time`, used to provide simple access to only
    `environment_sequence <Time.environment_sequence>`, `environment_state_update <Time.environment_state_update>`, and `consideration_set_execution <Time.consideration_set_execution>`
    """
    def __init__(self, time_ref):
        self._time_ref = time_ref

    # override __repr__ because this class is used only for cosmetic simplicity
    # based on a Time object
    def __repr__(self):
        abs_str = f'{self.absolute}, ' if self._time_ref.absolute_enabled else ''
        ts_str = self._time_ref._time_repr(exclusions={TimeScale.LIFE, TimeScale.PASS})
        return f'Time({abs_str}{ts_str})'

    def __getitem__(self, item):
        return self._time_ref[item]

    def __setitem__(self, item, value):
        self._time_ref[item] = value

    @property
    def absolute(self):
        return self._time_ref.absolute

    @property
    def environment_sequence(self):
        return self._time_ref.environment_sequence

    @property
    def environment_state_update(self):
        return self._time_ref.environment_state_update

    @property
    def consideration_set_execution(self):
        return self._time_ref.consideration_set_execution


class TimeHistoryTree:
    """
    A tree object that stores a history of time that has occurred at various
    :class:`TimeScale`\\ s, typically used in conjunction with a `Clock`

    Attributes
    ----------
        time_scale : :class:`TimeScale` : `TimeScale.LIFE`
            the TimeScale unit this tree/node represents

        child_time_scale : :class:`TimeScale` : `TimeScale.ENVIRONMENT_SEQUENCE`
            the TimeScale unit for this tree's children

        children : list[`TimeHistoryTree`]
            an ordered list of this tree's children

        max_depth : :class:`TimeScale` : `TimeScale.ENVIRONMENT_STATE_UPDATE`
            the finest grain TimeScale that should be created as a subtree
            Setting this value lower allows for more precise measurements
            (by default, you cannot query the number of
            `TimeScale.CONSIDERATION_SET_EXECUTION`\\ s in a certain `TimeScale.PASS`), but
            this may use a large amount of memory in large simulations

        index : int
            the index this tree has in its parent's children list

        parent : `TimeHistoryTree` : None
            the parent node of this tree, if it exists. \
            None represents no parent (i.e. root node)

        previous_time : `Time`
            a `Time` object that represents the last time that has occurred in the tree

        current_time : `Time`
            a `Time` object that represents the current time in the tree

        total_times : dict{:class:`TimeScale`: int}
            stores the total number of units of :class:`TimeScale`\\ s that have \
            occurred over this tree's scope. Only contains entries for \
            :class:`TimeScale`\\ s of finer grain than **time_scale**

    Arguments
    ---------
        enable_current_time : bool : True
            sets this tree to maintain a `Time` object. If this tree is not
            a root (i.e. **time_scale** is `TimeScale.LIFE`)
    """
    def __init__(
        self,
        time_scale=TimeScale.LIFE,
        max_depth=TimeScale.ENVIRONMENT_STATE_UPDATE,
        index=0,
        parent=None,
        enable_current_time=True
    ):
        if enable_current_time:
            self.current_time = Time()
            self.previous_time = None
        self.index = index
        self.time_scale = time_scale
        self.max_depth = max_depth
        self.parent = parent

        self.child_time_scale = TimeScale.get_child(time_scale)

        if self.child_time_scale >= max_depth:
            self.children = [
                TimeHistoryTree(
                    self.child_time_scale,
                    max_depth=max_depth,
                    index=0,
                    parent=self,
                    enable_current_time=False
                )
            ]
        else:
            self.children = []

        self.total_times = {ts: 0 for ts in TimeScale if ts < self.time_scale}

    def increment_time(self, time_scale):
        """
        Increases this tree's **current_time** by one **time_scale**

        Arguments
        ---------
            time_scale : :class:`TimeScale`
                the unit of time to increment
        """
        if self.child_time_scale >= self.max_depth:
            if time_scale == self.child_time_scale:
                self.children.append(
                    TimeHistoryTree(
                        self.child_time_scale,
                        max_depth=self.max_depth,
                        index=len(self.children),
                        parent=self,
                        enable_current_time=False
                    )
                )
            else:
                self.children[-1].increment_time(time_scale)
        self.total_times[time_scale] += 1
        try:
            self.previous_time = copy.copy(self.current_time)
            self.current_time._increment_by_time_scale(time_scale)
        except AttributeError:
            # not all of these objects have time tracking
            pass

    def get_total_times_relative(
        self,
        query_time_scale,
        base_indices=None
    ):
        """
        Arguments
        ---------
            query_time_scale : :class:`TimeScale`
                the :class:`TimeScale` of units to be returned

            base_indices : dict{:class:`TimeScale`: int}
                a dictionary specifying what scope of time query_time_scale \
                is over. e.g.

                    base_indices = {TimeScale.ENVIRONMENT_SEQUENCE: 1, TimeScale.ENVIRONMENT_STATE_UPDATE: 5}

                gives the number of **query_time_scale**\\ s that have occurred \
                in the 5th `ENVIRONMENT_STATE_UPDATE <TimeScale.ENVIRONMENT_STATE_UPDATE>` of the 1st `ENVIRONMENT_SEQUENCE`. If an entry for a :class:`TimeScale` \
                is not specified but is coarser than **query_time_scale**, the latest \
                value for that entry will be used

        Returns
        -------
            the number of units of query_time_scale that have occurred within \
            the scope of time specified by base_indices : int
        """
        if query_time_scale >= self.time_scale:
            raise TimeScaleError(
                'query_time_scale (given: {0}) must be of finer grain than {1}.time_scale ({2})'.format(
                    query_time_scale, self, self.time_scale
                )
            )

        try:
            self.current_time
        except AttributeError:
            raise TimeScaleError(
                'get_total_times_relative should only be called on a TimeHistoryTree with enable_current_time set to True'
            )

        default_base_indices = {
            TimeScale.LIFE: 0,
            TimeScale.ENVIRONMENT_SEQUENCE: None,
            TimeScale.ENVIRONMENT_STATE_UPDATE: None,
            TimeScale.PASS: None,
        }

        # overwrite defaults with dictionary passed in argument
        if base_indices is None:
            base_indices = default_base_indices
        else:
            default_base_indices.update(base_indices)
            base_indices = default_base_indices

        base_time_scale = TimeScale.LIFE
        # base_time_scale is set as the finest grain TimeScale that is specified,
        # but more coarse than query_time_scale
        # this will be where the query to attribute times will be made
        for ts in sorted(base_indices, reverse=True):
            if base_indices[ts] is not None and ts > query_time_scale:
                base_time_scale = ts

        if base_time_scale > self.time_scale:
            raise TimeScaleError(
                'base TimeScale set by base_indices ({0}) must be at least as fine as this TimeHistoryTree\'s time_scale ({1})'.format(
                    base_time_scale,
                    self.time_scale
                )
            )

        # get the root node, which will (and should) have TimeScale.LIFE
        node = self
        while node.parent is not None:
            node = node.parent

        try:
            # for all non-specified (i.e. set to None) TimeScales coarser than base_time_scale,
            # assign them to their latest time values as default
            while node.time_scale > base_time_scale:
                if base_indices[node.child_time_scale] is None:
                    base_indices[node.child_time_scale] = len(node.children) - 1
                node = node.children[base_indices[node.child_time_scale]]

            # attempt to retrieve the correct time count given the base_indices dictionary
            node = self
            while node.time_scale != base_time_scale:
                node = node.children[base_indices[node.child_time_scale]]
            return node.total_times[query_time_scale]
        except IndexError:
            raise TimeScaleError(
                'TimeHistoryTree {0}: {1} {2} does not exist in {3} {4}'.format(
                    self,
                    node.child_time_scale,
                    base_indices[node.child_time_scale],
                    node.time_scale,
                    node.index
                )
            )


@functools.lru_cache(maxsize=None)
def _time_scale_to_attr_str(time_scale: TimeScale) -> str:
    attr = time_scale.name.lower()

    if keyword.iskeyword(attr):
        return f'{attr}_'
    else:
        return attr


@functools.lru_cache(maxsize=None)
def _time_scale_to_class_str(time_scale: typing.Union[TimeScale, str]) -> str:
    try:
        name = time_scale.name
    except AttributeError:
        name = time_scale

    return ''.join([f'{x[0].upper()}{x[1:].lower()}' for x in name.split('_')])


def _attr_str_to_time_scale(attr_str):
    return getattr(TimeScale, attr_str.rstrip('_'))


def _multi_substitute_docstring(cls, subs: typing.Dict[str, str]):
    new_docstring = cls.__doc__
    try:
        for prev, new in subs.items():
            new_docstring = re.sub(prev, new, new_docstring)
    except TypeError:
        pass
    else:
        cls._old_docstring = cls.__doc__
        cls.__doc__ = new_docstring


def set_time_scale_alias(name: str, target: TimeScale):
    """Sets an alias named **name** of TimeScale **target**

    Args:
        name (str): name of the alias
        target (TimeScale): TimeScale that **name** will refer to
    """
    import graph_scheduler

    name_aliased_time_scales = list(filter(
        lambda e: _time_scale_aliases[e] == name,
        _time_scale_aliases
    ))
    if len(name_aliased_time_scales) > 0:
        raise ValueError(f"'{name}' is already aliased to {name_aliased_time_scales[0]}")

    try:
        target = getattr(TimeScale, target)
    except TypeError:
        pass
    except AttributeError as e:
        raise ValueError(f'Invalid TimeScale {target}') from e

    _time_scale_aliases[target] = name
    setattr(TimeScale, name, target)

    def getter(self):
        return getattr(self, _time_scale_to_attr_str(target))

    def setter(self, value):
        setattr(self, _time_scale_to_attr_str(target), value)

    prop = property(getter).setter(setter)
    setattr(Time, name.lower(), prop)
    setattr(SimpleTime, name.lower(), prop)

    # alias name in style of a class name
    new_class_segment_name = _time_scale_to_class_str(name)
    for cls_name, cls in graph_scheduler.__dict__.copy().items():
        # make aliases of conditions that contain a TimeScale name (e.g. AtEnvironmentStateUpdate)
        target_class_segment_name = _time_scale_to_class_str(target)

        if isinstance(cls, (type, types.ModuleType)):
            if isinstance(cls, types.ModuleType):
                try:
                    if _alias_docs_warning_str not in cls.__doc__:
                        cls.__doc__ = f'{_alias_docs_warning_str}{cls.__doc__}'
                except TypeError:
                    pass

            _multi_substitute_docstring(
                cls,
                {
                    target.name: name,
                    target_class_segment_name: new_class_segment_name,
                }
            )

        if target_class_segment_name in cls_name:
            new_cls_name = cls_name.replace(
                target_class_segment_name,
                new_class_segment_name
            )

            setattr(graph_scheduler.condition, new_cls_name, cls)
            setattr(graph_scheduler, new_cls_name, cls)

            graph_scheduler.condition.__all__.append(new_cls_name)
            graph_scheduler.__all__.append(new_cls_name)


def remove_time_scale_alias(name: str):
    """Removes an alias previously set by `set_time_scale_alias`

    Args:
        name (str): name of the TimeScale alias to remove
    """
    import graph_scheduler

    name_aliased_time_scales = list(filter(
        lambda e: _time_scale_aliases[e] == name,
        _time_scale_aliases
    ))
    assert len(name_aliased_time_scales) <= 1

    try:
        del _time_scale_aliases[name_aliased_time_scales[0]]
    except (IndexError, KeyError):
        return
    else:
        delattr(TimeScale, name)
        delattr(Time, name.lower())
        delattr(SimpleTime, name.lower())

        new_class_segment_name = _time_scale_to_class_str(name)
        for cls_name, cls in graph_scheduler.__dict__.copy().items():

            if isinstance(cls, (type, types.ModuleType)):
                if isinstance(cls, types.ModuleType):
                    if len(_time_scale_aliases) == 0:
                        try:
                            re.sub(_alias_docs_warning_str, '', cls.__doc__)
                        except TypeError:
                            pass

                try:
                    cls.__doc__ = cls._old_docstring
                    del cls._old_docstring
                except AttributeError:
                    # not all modules have docstrings
                    pass

            # NOTE: this could accidentally remove some unintended conditions if
            # an alias is constructed such that its determined class name is
            # a substring of some other Condition class name, but this is unlikely
            if new_class_segment_name in cls_name:
                delattr(graph_scheduler.condition, cls_name)
                delattr(graph_scheduler, cls_name)

                graph_scheduler.condition.__all__.remove(cls_name)
                graph_scheduler.__all__.remove(cls_name)
