import pytest

import graph_scheduler as gs

SimpleTestNode = pytest.helpers.get_test_node()


class TestTime:
    @pytest.mark.parametrize(
        'base, increment_time_scale, expected',
        [
            (
                gs.Time(environment_sequence=0, environment_state_update=0, pass_=0, consideration_set_execution=0),
                gs.TimeScale.ENVIRONMENT_STATE_UPDATE,
                gs.Time(environment_sequence=0, environment_state_update=1, pass_=0, consideration_set_execution=0),
            ),
            (
                gs.Time(environment_sequence=0, environment_state_update=0, pass_=5, consideration_set_execution=9),
                gs.TimeScale.ENVIRONMENT_STATE_UPDATE,
                gs.Time(environment_sequence=0, environment_state_update=1, pass_=0, consideration_set_execution=0),
            ),
            (
                gs.Time(environment_sequence=1, environment_state_update=0, pass_=5, consideration_set_execution=9),
                gs.TimeScale.ENVIRONMENT_STATE_UPDATE,
                gs.Time(environment_sequence=1, environment_state_update=1, pass_=0, consideration_set_execution=0),
            ),
            (
                gs.Time(environment_sequence=1, environment_state_update=0, pass_=5, consideration_set_execution=9),
                gs.TimeScale.CONSIDERATION_SET_EXECUTION,
                gs.Time(environment_sequence=1, environment_state_update=0, pass_=5, consideration_set_execution=10),
            ),
        ],
    )
    def test_increment(self, base, increment_time_scale, expected):
        base._increment_by_time_scale(increment_time_scale)
        assert base == expected

    def test_multiple_runs(self):
        t1 = SimpleTestNode('t1')
        t2 = SimpleTestNode('t2')

        sched = gs.Scheduler(graph={t1: set(), t2: {t1}})

        sched.run()
        sched.end_environment_sequence()
        assert sched.get_clock().time == gs.Time(environment_sequence=1, environment_state_update=0, pass_=0, consideration_set_execution=0)

        sched.run()
        sched.end_environment_sequence()
        assert sched.get_clock().time == gs.Time(environment_sequence=2, environment_state_update=0, pass_=0, consideration_set_execution=0)

    def test_get_set_item_time(self):
        t = gs.Time(environment_sequence=1, environment_state_update=2, pass_=3, consideration_set_execution=4)
        st = gs.SimpleTime(t)

        assert t[0] == 4
        assert t[1] == 3
        assert t[2] == 2
        assert t[3] == 1
        assert t[4] == 0

        assert st[0] == 4
        assert st[2] == 2
        assert st[3] == 1

        t[3] = 5
        assert t[0] == 4
        assert t[1] == 3
        assert t[2] == 2
        assert t[3] == 5
        assert t[4] == 0

        assert st[0] == 4
        assert st[2] == 2
        assert st[3] == 5

        st[0] = 6
        assert t[0] == 6
        assert t[1] == 3
        assert t[2] == 2
        assert t[3] == 5
        assert t[4] == 0

        assert st[0] == 6
        assert st[2] == 2
        assert st[3] == 5


class TestTimeHistoryTree:
    def test_defaults(self):
        h = gs.TimeHistoryTree()

        for node in [h, h.children[0]]:
            assert len(node.children) == 1
            assert all([node.total_times[ts] == 0 for ts in node.total_times])
            assert node.time_scale == gs.TimeScale.get_parent(node.children[0].time_scale)
            assert node.time_scale >= gs.TimeScale.ENVIRONMENT_STATE_UPDATE

    @pytest.mark.parametrize(
        'max_depth',
        [
            (gs.TimeScale.ENVIRONMENT_SEQUENCE),
            (gs.TimeScale.ENVIRONMENT_STATE_UPDATE)
        ])
    def test_max_depth(self, max_depth):
        h = gs.TimeHistoryTree(max_depth=max_depth)

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
        # save and replace other aliases in case integration packages
        # apply them (psyneulink does)
        existing_aliases = gs.time._time_scale_aliases.copy()
        for ts, alias in existing_aliases.items():
            gs.remove_time_scale_alias(alias)

        gs.set_time_scale_alias('MY_ENVIRONMENT_STATE_UPDATE_ALIAS', gs.TimeScale.ENVIRONMENT_STATE_UPDATE)

        yield

        gs.remove_time_scale_alias('MY_ENVIRONMENT_STATE_UPDATE_ALIAS')
        for ts, alias in existing_aliases.items():
            gs.set_time_scale_alias(alias, ts)

    @staticmethod
    def assert_environment_state_update_and_alias_equals(time_obj, value):
        assert time_obj.my_environment_state_update_alias == value
        assert time_obj.environment_state_update == value

        if isinstance(time_obj, gs.Time):
            assert time_obj._get_by_time_scale(gs.TimeScale.ENVIRONMENT_STATE_UPDATE) == value
            assert time_obj._get_by_time_scale(gs.TimeScale.MY_ENVIRONMENT_STATE_UPDATE_ALIAS) == value

    def test_alias_references_timescale(self):
        assert gs.TimeScale.MY_ENVIRONMENT_STATE_UPDATE_ALIAS is gs.TimeScale.ENVIRONMENT_STATE_UPDATE
        assert gs.TimeScale.get_parent(gs.TimeScale.MY_ENVIRONMENT_STATE_UPDATE_ALIAS) is gs.TimeScale.get_parent(gs.TimeScale.ENVIRONMENT_STATE_UPDATE)
        assert gs.TimeScale.get_child(gs.TimeScale.MY_ENVIRONMENT_STATE_UPDATE_ALIAS) is gs.TimeScale.get_child(gs.TimeScale.ENVIRONMENT_STATE_UPDATE)

    @pytest.mark.parametrize(
        'kwargs, value',
        [
            ({'environment_state_update': 1}, 1),
            ({'my_environment_state_update_alias': 1}, 1)
        ]
    )
    def test_time_change_in_init(self, kwargs, value):
        t = gs.Time(**kwargs)
        st = gs.SimpleTime(t)

        self.assert_environment_state_update_and_alias_equals(t, value)
        self.assert_environment_state_update_and_alias_equals(st, value)

    def test_time_change_by_original_attr(self):
        t = gs.Time()
        st = gs.SimpleTime(t)

        t.environment_state_update = 1

        self.assert_environment_state_update_and_alias_equals(t, 1)
        self.assert_environment_state_update_and_alias_equals(st, 1)

    def test_time_change_by_alias_attr(self):
        t = gs.Time()
        st = gs.SimpleTime(t)

        t.my_environment_state_update_alias = 1

        self.assert_environment_state_update_and_alias_equals(t, 1)
        self.assert_environment_state_update_and_alias_equals(st, 1)

    @pytest.mark.parametrize('time_scale', [gs.TimeScale.ENVIRONMENT_STATE_UPDATE, 'MY_ENVIRONMENT_STATE_UPDATE_ALIAS'])
    def test_time_change_by_method_set(self, time_scale):
        # alias does not exist at parametrization time
        try:
            time_scale = getattr(gs.TimeScale, time_scale)
        except TypeError:
            pass

        t = gs.Time()
        st = gs.SimpleTime(t)
        t._set_by_time_scale(time_scale, 1)

        self.assert_environment_state_update_and_alias_equals(t, 1)
        self.assert_environment_state_update_and_alias_equals(st, 1)

    @pytest.mark.parametrize('time_scale', [gs.TimeScale.ENVIRONMENT_STATE_UPDATE, 'MY_ENVIRONMENT_STATE_UPDATE_ALIAS'])
    def test_time_change_by_method_increment(self, time_scale):
        # alias does not exist at parametrization time
        try:
            time_scale = getattr(gs.TimeScale, time_scale)
        except TypeError:
            pass

        t = gs.Time()
        st = gs.SimpleTime(t)
        t._increment_by_time_scale(time_scale)

        self.assert_environment_state_update_and_alias_equals(t, 1)
        self.assert_environment_state_update_and_alias_equals(st, 1)

    def test_time_repr(self):
        t = gs.Time(my_environment_state_update_alias=1)
        st = gs.SimpleTime(t)

        assert repr(t) == 'Time(environment_sequence: 0, my_environment_state_update_alias: 1, pass: 0, consideration_set_execution: 0)'
        assert repr(st) == 'Time(environment_sequence: 0, my_environment_state_update_alias: 1, consideration_set_execution: 0)'

    def test_aliased_conditions(self):
        graph = {'A': set()}
        sched_orig = gs.Scheduler(
            graph=graph,
            termination_conds={gs.TimeScale.ENVIRONMENT_STATE_UPDATE: gs.AtPass(2)}
        )
        sched_alias = gs.Scheduler(
            graph=graph,
            termination_conds={gs.TimeScale.ENVIRONMENT_STATE_UPDATE: gs.AtPass(2)}
        )

        sched_orig.add_condition('A', gs.BeforeEnvironmentStateUpdate(3))
        sched_alias.add_condition('A', gs.BeforeMyEnvironmentStateUpdateAlias(3))

        # should run in environment state updates 0-2 and not in 3
        for _ in range(4):
            assert list(sched_orig.run()) == list(sched_alias.run())
