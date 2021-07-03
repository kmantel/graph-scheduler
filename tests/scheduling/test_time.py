import psyneulink as pnl
import pytest

import graph_scheduler
from graph_scheduler.time import (
    SimpleTime, Time, TimeHistoryTree, TimeScale, _time_scale_aliases,
    remove_time_scale_alias, set_time_scale_alias,
)


class TestTime:
    @pytest.mark.parametrize(
        'base, increment_time_scale, expected',
        [
            (Time(run=0, trial=0, pass_=0, consideration_set_execution=0), TimeScale.TRIAL, Time(run=0, trial=1, pass_=0, consideration_set_execution=0)),
            (Time(run=0, trial=0, pass_=5, consideration_set_execution=9), TimeScale.TRIAL, Time(run=0, trial=1, pass_=0, consideration_set_execution=0)),
            (Time(run=1, trial=0, pass_=5, consideration_set_execution=9), TimeScale.TRIAL, Time(run=1, trial=1, pass_=0, consideration_set_execution=0)),
            (Time(run=1, trial=0, pass_=5, consideration_set_execution=9), TimeScale.CONSIDERATION_SET_EXECUTION, Time(run=1, trial=0, pass_=5, consideration_set_execution=10)),
        ]
    )
    def test_increment(self, base, increment_time_scale, expected):
        base._increment_by_time_scale(increment_time_scale)
        assert base == expected

    def test_multiple_runs(self):
        t1 = pnl.TransferMechanism()
        t2 = pnl.TransferMechanism()

        C = pnl.Composition(pathways=[t1, t2])

        C.run(inputs={t1: [[1.0], [2.0], [3.0]]})
        assert C.scheduler.get_clock(C).time == pnl.Time(run=1, trial=0, pass_=0, time_step=0)

        C.run(inputs={t1: [[4.0], [5.0], [6.0]]})
        assert C.scheduler.get_clock(C).time == pnl.Time(run=2, trial=0, pass_=0, time_step=0)


class TestTimeHistoryTree:
    def test_defaults(self):
        h = TimeHistoryTree()

        for node in [h, h.children[0]]:
            assert len(node.children) == 1
            assert all([node.total_times[ts] == 0 for ts in node.total_times])
            assert node.time_scale == TimeScale.get_parent(node.children[0].time_scale)
            assert node.time_scale >= TimeScale.TRIAL

    @pytest.mark.parametrize(
        'max_depth',
        [
            (TimeScale.RUN),
            (TimeScale.TRIAL)
        ])
    def test_max_depth(self, max_depth):
        h = TimeHistoryTree(max_depth=max_depth)

        node = h
        found_max_depth = h.time_scale == max_depth
        while len(node.children) > 0:
            node = node.children[0]
            found_max_depth = found_max_depth or node.time_scale == max_depth
            assert node.time_scale >= max_depth

        assert found_max_depth


class TestAliasTimeScale:
    @pytest.fixture(scope='class', autouse=True)
    def setup_alias(cls):
        # must save and replace psyneulink aliases as long as tests still
        # depend on psyneulink
        existing_aliases = _time_scale_aliases.copy()
        for ts, alias in existing_aliases.items():
            remove_time_scale_alias(alias)

        set_time_scale_alias('MY_TRIAL_ALIAS', TimeScale.TRIAL)

        yield

        remove_time_scale_alias('MY_TRIAL_ALIAS')
        for ts, alias in existing_aliases.items():
            set_time_scale_alias(alias, ts)

    @staticmethod
    def assert_trial_and_alias_equals(time_obj, value):
        assert time_obj.my_trial_alias == value
        assert time_obj.trial == value

        if isinstance(time_obj, Time):
            assert time_obj._get_by_time_scale(TimeScale.TRIAL) == value
            assert time_obj._get_by_time_scale(TimeScale.MY_TRIAL_ALIAS) == value

    def test_alias_references_timescale(self):
        assert TimeScale.MY_TRIAL_ALIAS is TimeScale.TRIAL
        assert TimeScale.get_parent(TimeScale.MY_TRIAL_ALIAS) is TimeScale.get_parent(TimeScale.TRIAL)
        assert TimeScale.get_child(TimeScale.MY_TRIAL_ALIAS) is TimeScale.get_child(TimeScale.TRIAL)

    @pytest.mark.parametrize(
        'kwargs, value',
        [
            ({'trial': 1}, 1),
            ({'my_trial_alias': 1}, 1)
        ]
    )
    def test_time_change_in_init(self, kwargs, value):
        t = Time(**kwargs)
        st = SimpleTime(t)

        self.assert_trial_and_alias_equals(t, value)
        self.assert_trial_and_alias_equals(st, value)

    def test_time_change_by_original_attr(self):
        t = Time()
        st = SimpleTime(t)

        t.trial = 1

        self.assert_trial_and_alias_equals(t, 1)
        self.assert_trial_and_alias_equals(st, 1)

    def test_time_change_by_alias_attr(self):
        t = Time()
        st = SimpleTime(t)

        t.my_trial_alias = 1

        self.assert_trial_and_alias_equals(t, 1)
        self.assert_trial_and_alias_equals(st, 1)

    @pytest.mark.parametrize('time_scale', [TimeScale.TRIAL, 'MY_TRIAL_ALIAS'])
    def test_time_change_by_method_set(self, time_scale):
        # alias does not exist at parametrization time
        try:
            time_scale = getattr(TimeScale, time_scale)
        except TypeError:
            pass

        t = Time()
        st = SimpleTime(t)
        t._set_by_time_scale(time_scale, 1)

        self.assert_trial_and_alias_equals(t, 1)
        self.assert_trial_and_alias_equals(st, 1)

    @pytest.mark.parametrize('time_scale', [TimeScale.TRIAL, 'MY_TRIAL_ALIAS'])
    def test_time_change_by_method_increment(self, time_scale):
        # alias does not exist at parametrization time
        try:
            time_scale = getattr(TimeScale, time_scale)
        except TypeError:
            pass

        t = Time()
        st = SimpleTime(t)
        t._increment_by_time_scale(time_scale)

        self.assert_trial_and_alias_equals(t, 1)
        self.assert_trial_and_alias_equals(st, 1)

    def test_time_repr(self):
        t = Time(my_trial_alias=1)
        st = SimpleTime(t)

        assert repr(t) == 'Time(run: 0, my_trial_alias: 1, pass: 0, consideration_set_execution: 0)'
        assert repr(st) == 'Time(run: 0, my_trial_alias: 1, consideration_set_execution: 0)'

    def test_aliased_conditions(self):
        graph = {'A': set()}
        sched_orig = graph_scheduler.Scheduler(
            graph=graph,
            termination_conds={TimeScale.TRIAL: graph_scheduler.AtPass(2)}
        )
        sched_alias = graph_scheduler.Scheduler(
            graph=graph,
            termination_conds={TimeScale.TRIAL: graph_scheduler.AtPass(2)}
        )

        sched_orig.add_condition('A', graph_scheduler.BeforeTrial(3))
        sched_alias.add_condition('A', graph_scheduler.BeforeMyTrialAlias(3))

        # should run in trials 0-2 and not in 3
        for _ in range(4):
            assert list(sched_orig.run()) == list(sched_alias.run())
