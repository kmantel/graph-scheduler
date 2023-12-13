import logging

import numpy as np
import pytest

import graph_scheduler as gs

logger = logging.getLogger(__name__)

SimpleTestNode = pytest.helpers.get_test_node()


class TestConditionSet:
    # maintain through v1.x
    class TestConstructorInterface:
        @pytest.fixture
        def conds(self):
            return {
                'A': gs.Never(),
                'B': gs.EveryNCalls('A', 1),
                'C': gs.And(
                    gs.Or(
                        gs.TimeInterval(repeat=1), gs.Always()
                    ),
                    gs.JustRan('B')
                )
            }

        def test_positional_arg(self, conds):
            cond_set = gs.ConditionSet(conds)
            assert cond_set.conditions == conds

        def test_keyword_arg(self, conds):
            cond_set = gs.ConditionSet(conditions=conds)
            assert cond_set.conditions == conds


class TestCondition:

    def test_invalid_input_WhenFinished(self):
        with pytest.raises(gs.ConditionError):
            gs.WhenFinished(None).is_satisfied()

    def test_invalid_input_WhenFinishedAny_1(self):
        with pytest.raises(gs.ConditionError):
            gs.WhenFinished(None).is_satisfied()

    def test_invalid_input_WhenFinishedAny_2(self):
        with pytest.raises(gs.ConditionError):
            gs.WhenFinished({None}).is_satisfied()

    def test_invalid_input_WhenFinishedAll_1(self):
        with pytest.raises(gs.ConditionError):
            gs.WhenFinished(None).is_satisfied()

    def test_invalid_input_WhenFinishedAll_2(self):
        with pytest.raises(gs.ConditionError):
            gs.WhenFinished({None}).is_satisfied()

    def test_additional_args(self):
        class OneSatisfied(gs.Condition):
            def __init__(self, a):
                def func(a, b):
                    return a or b
                super().__init__(func, a)

        cond = OneSatisfied(True)
        assert cond.is_satisfied(True)
        assert cond.is_satisfied(False)

        cond = OneSatisfied(False)
        assert cond.is_satisfied(True)
        assert not cond.is_satisfied(False)

    def test_additional_kwargs(self):
        class OneSatisfied(gs.Condition):
            def __init__(self, a, c=True):
                def func(a, b, c=True):
                    return a or b or c
                super().__init__(func, a, c=True)

        cond = OneSatisfied(True)
        assert cond.is_satisfied(True)
        assert cond.is_satisfied(False, c=True)
        assert cond.is_satisfied(False, c=False)

        cond = OneSatisfied(True, c=False)
        assert cond.is_satisfied(True)
        assert cond.is_satisfied(False, c=True)
        assert cond.is_satisfied(False, c=False)

        cond = OneSatisfied(False)
        assert cond.is_satisfied(True)
        assert cond.is_satisfied(False, c=True)
        assert not cond.is_satisfied(False, c=False)
        assert not cond.is_satisfied(False, c=False, extra_arg=True)

    class TestGeneric:
        def test_WhileNot_AtPass(self):
            A = 'A'
            sched = gs.Scheduler(graph={'A': set()})
            sched.add_condition(A, gs.WhileNot(lambda sched: sched.get_clock(sched.default_execution_id).get_total_times_relative(gs.TimeScale.PASS, gs.TimeScale.ENVIRONMENT_STATE_UPDATE) == 0, sched))

            termination_conds = {}
            termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
            termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [set(), A, A, A, A]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_WhileNot_AtPass_in_middle(self):
            A = 'A'
            sched = gs.Scheduler(graph={'A': set()})
            sched.add_condition(A, gs.WhileNot(lambda sched: sched.get_clock(sched.default_execution_id).get_total_times_relative(gs.TimeScale.PASS, gs.TimeScale.ENVIRONMENT_STATE_UPDATE) == 2, sched))

            termination_conds = {}
            termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
            termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [A, A, set(), A, A]
            assert output == pytest.helpers.setify_expected_output(expected_output)

    class TestRelative:

        def test_Any_end_before_one_finished(self):
            A = 'A'
            sched = gs.Scheduler(graph={A: set()})

            sched.add_condition(A, gs.EveryNPasses(1))

            termination_conds = {}
            termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
            termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.Any(gs.AfterNCalls(A, 10), gs.AtPass(5))
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [A for _ in range(5)]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_All_end_after_one_finished(self):
            A = 'A'
            sched = gs.Scheduler(graph={A: set()})

            sched.add_condition(A, gs.EveryNPasses(1))

            termination_conds = {}
            termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
            termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.Any(gs.AfterNCalls(A, 5), gs.AtPass(10))
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [A for _ in range(5)]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_Not_AtPass(self):
            A = 'A'
            sched = gs.Scheduler(graph={'A': set()})
            sched.add_condition(A, gs.Not(gs.AtPass(0)))

            termination_conds = {}
            termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
            termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [set(), A, A, A, A]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_Not_AtPass_in_middle(self):
            A = 'A'
            sched = gs.Scheduler(graph={'A': set()})
            sched.add_condition(A, gs.Not(gs.AtPass(2)))

            termination_conds = {}
            termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
            termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [A, A, set(), A, A]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        @pytest.mark.parametrize(
            'n,expected_output', [
                (0, ['A', 'A', 'A', 'A', 'A', 'A']),
                (1, ['A', 'A', 'A', 'B', 'A', 'A', 'A']),
                (2, ['A', 'A', 'A', 'B', 'A', 'B', 'A', 'A']),
            ]
        )
        def test_NWhen_AfterNCalls(self, n, expected_output):
            A = 'A'
            B = 'B'
            sched = gs.Scheduler(graph={'A': set(), 'B': {'A'}})
            sched.add_condition(A, gs.Always())
            sched.add_condition(B, gs.NWhen(gs.AfterNCalls(A, 3), n))

            termination_conds = {}
            termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
            termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AfterNCalls(A, 6)
            output = list(sched.run(termination_conds=termination_conds))

            assert output == pytest.helpers.setify_expected_output(expected_output)

    class TestTimePNL:

        def test_BeforeConsiderationSetExecution(self):
            A = 'A'
            sched = gs.Scheduler(graph={'A': set()})
            sched.add_condition(A, gs.BeforeConsiderationSetExecution(2))

            termination_conds = {}
            termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
            termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [A, A, set(), set(), set()]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_BeforeConsiderationSetExecution_2(self):
            A = 'A'
            B = 'B'
            sched = gs.Scheduler(graph={'A': set(), 'B': {'A'}})
            sched.add_condition(A, gs.BeforeConsiderationSetExecution(2))
            sched.add_condition(B, gs.Always())

            termination_conds = {}
            termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
            termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [A, B, B, B, B, B]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_AtConsiderationSetExecution(self):
            A = 'A'
            sched = gs.Scheduler(graph={'A': set()})
            sched.add_condition(A, gs.AtConsiderationSetExecution(0))

            termination_conds = {}
            termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
            termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [A, set(), set(), set(), set()]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_BeforePass(self):
            A = 'A'
            sched = gs.Scheduler(graph={'A': set()})
            sched.add_condition(A, gs.BeforePass(2))

            termination_conds = {}
            termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
            termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [A, A, set(), set(), set()]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_AtPass(self):
            A = 'A'
            sched = gs.Scheduler(graph={A: set()})
            sched.add_condition(A, gs.AtPass(0))

            termination_conds = {}
            termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
            termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [A, set(), set(), set(), set()]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_AtPass_underconstrained(self):
            A = 'A'
            B = 'B'
            C = 'C'
            sched = gs.Scheduler(graph={A: set(), B: {A}, C: {B}})
            sched.add_condition(A, gs.AtPass(0))
            sched.add_condition(B, gs.Always())
            sched.add_condition(C, gs.Always())

            termination_conds = {}
            termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
            termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AfterNCalls(C, 2)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [A, B, C, B, C]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_AtPass_in_middle(self):
            A = 'A'
            sched = gs.Scheduler(graph={A: set()})
            sched.add_condition(A, gs.AtPass(2))

            termination_conds = {}
            termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
            termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [set(), set(), A, set(), set()]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_AtPass_at_end(self):
            A = 'A'
            sched = gs.Scheduler(graph={A: set()})
            sched.add_condition(A, gs.AtPass(5))

            termination_conds = {}
            termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
            termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [set(), set(), set(), set(), set()]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_AtPass_after_end(self):
            A = 'A'
            sched = gs.Scheduler(graph={A: set()})
            sched.add_condition(A, gs.AtPass(6))

            termination_conds = {}
            termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
            termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [set(), set(), set(), set(), set()]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_AfterPass(self):
            A = 'A'
            sched = gs.Scheduler(graph={A: set()})
            sched.add_condition(A, gs.AfterPass(0))

            termination_conds = {}
            termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
            termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [set(), A, A, A, A]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_AfterNPasses(self):
            A = 'A'
            sched = gs.Scheduler(graph={A: set()})
            sched.add_condition(A, gs.AfterNPasses(1))

            termination_conds = {}
            termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
            termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [set(), A, A, A, A]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_BeforeEnvironmentStateUpdate(self):
            A = 'A'
            sched = gs.Scheduler(graph={A: set()})
            sched.add_condition(A, gs.BeforeEnvironmentStateUpdate(4))

            termination_conds = {}
            termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(5)
            termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AtPass(1)
            for _ in range(6):
                list(sched.run(termination_conds))

            output = sched.execution_list[None]
            expected_output = [A, A, A, A, set()]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_AtEnvironmentStateUpdate(self):
            A = 'A'
            sched = gs.Scheduler(graph={A: set()})
            sched.add_condition(A, gs.Always())

            termination_conds = {}
            termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AtEnvironmentStateUpdate(4)
            termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AtPass(1)
            for _ in range(5):
                list(sched.run(termination_conds))

            output = sched.execution_list[None]
            expected_output = [A, A, A, A]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_AfterEnvironmentStateUpdate(self):
            A = 'A'
            sched = gs.Scheduler(graph={A: set()})
            sched.add_condition(A, gs.Always())

            termination_conds = {}
            termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterEnvironmentStateUpdate(4)
            termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AtPass(1)
            for _ in range(6):
                list(sched.run(termination_conds))

            output = sched.execution_list[None]
            expected_output = [A, A, A, A, A]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_AfterNEnvironmentStateUpdates(self):
            A = 'A'
            sched = gs.Scheduler(graph={A: set()})
            sched.add_condition(A, gs.AfterNPasses(1))

            termination_conds = {}
            termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
            termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [set(), A, A, A, A]
            assert output == pytest.helpers.setify_expected_output(expected_output)

    class TestTime:

        @pytest.mark.parametrize(
            'node_condition, termination_conditions, expected_output',
            [
                pytest.param(
                    gs.AfterNPasses(1),
                    {
                        gs.TimeScale.ENVIRONMENT_SEQUENCE: gs.AfterNEnvironmentStateUpdates(1),
                        gs.TimeScale.ENVIRONMENT_STATE_UPDATE: gs.AfterConsiderationSetExecution(4)
                    },
                    [set(), 'A', 'A', 'A', 'A'],
                    id='AfterConsiderationSetExecution'
                ),
                pytest.param(
                    gs.AfterNPasses(1),
                    {
                        gs.TimeScale.ENVIRONMENT_SEQUENCE: gs.AfterNEnvironmentStateUpdates(1),
                        gs.TimeScale.ENVIRONMENT_STATE_UPDATE: gs.AfterNConsiderationSetExecutions(5)
                    },
                    [set(), 'A', 'A', 'A', 'A'],
                    id='AfterNConsiderationSetExecutions'
                ),
            ]
        )
        def test_single_node(
            self, node_condition, termination_conditions, expected_output
        ):
            graph = {'A': set()}

            sched = gs.Scheduler(graph)
            sched.add_condition('A', node_condition)
            output = list(sched.run(termination_conds=termination_conditions))

            assert output == pytest.helpers.setify_expected_output(expected_output)

        @pytest.mark.parametrize(
            'node_condition, termination_conditions, expected_output, n_sequences, n_state_updates_per_sequence',
            [
                pytest.param(
                    gs.AtEnvironmentStateUpdateNStart(2),
                    {gs.TimeScale.ENVIRONMENT_STATE_UPDATE: gs.AfterNPasses(1)},
                    [[set(), set(), 'A', set()]],
                    1,
                    4,
                    id='AtEnvironmentStateUpdateNStart'
                ),
                pytest.param(
                    gs.AtEnvironmentSequence(4),
                    {gs.TimeScale.ENVIRONMENT_STATE_UPDATE: gs.AfterNPasses(1)},
                    [[set()], [set()], [set()], [set()], ['A'], [set()]],
                    6,
                    1,
                    id='AtEnvironmentSequence'
                ),
                pytest.param(
                    gs.AfterEnvironmentSequence(3),
                    {gs.TimeScale.ENVIRONMENT_STATE_UPDATE: gs.AfterNPasses(1)},
                    [[set()], [set()], [set()], [set()], ['A'], ['A']],
                    6,
                    1,
                    id='AfterEnvironmentSequence'
                ),
                pytest.param(
                    gs.AfterNEnvironmentSequences(4),
                    {gs.TimeScale.ENVIRONMENT_STATE_UPDATE: gs.AfterNPasses(1)},
                    [[set()], [set()], [set()], [set()], ['A'], ['A']],
                    6,
                    1,
                    id='AfterNEnvironmentSequences'
                ),
                pytest.param(
                    gs.AtEnvironmentSequenceStart(),
                    {gs.TimeScale.ENVIRONMENT_STATE_UPDATE: gs.AfterNPasses(1)},
                    [['A', set()], ['A', set()]],
                    2,
                    2,
                    id='AtEnvironmentSequenceStart'
                ),
                pytest.param(
                    gs.AtEnvironmentSequenceNStart(1),
                    {gs.TimeScale.ENVIRONMENT_STATE_UPDATE: gs.AfterNPasses(1)},
                    [[set(), set()], ['A', set()], [set(), set()]],
                    3,
                    2,
                    id='AtEnvironmentSequenceNStart'
                ),
            ]
        )
        def test_single_node_n_sequences(
            self,
            node_condition,
            termination_conditions,
            expected_output,
            n_sequences,
            n_state_updates_per_sequence
        ):
            graph = {'A': set()}

            sched = gs.Scheduler(graph)
            sched.add_condition('A', node_condition)
            output = []

            for _ in range(n_sequences):
                su = []
                for i in range(n_state_updates_per_sequence):
                    su.extend(
                        list(sched.run(termination_conds=termination_conditions))
                    )
                output.append(su)
                sched.end_environment_sequence()

            for i in range(n_sequences):
                assert output[i] == pytest.helpers.setify_expected_output(expected_output[i]), f'ENVIRONMENT_SEQUENCE {i}'

    class TestComponentBased:

        def test_BeforeNCalls(self):
            A = 'A'
            sched = gs.Scheduler(graph={'A': set()})
            sched.add_condition(A, gs.BeforeNCalls(A, 3))

            termination_conds = {}
            termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
            termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [A, A, A, set(), set()]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        # NOTE:
        # The behavior is not desired (i.e. depending on the order mechanisms are checked, B running AtCall(A, x))
        # may run on both the xth and x+1st call of A; if A and B are not parent-child
        # A fix could invalidate key assumptions and affect many other conditions
        # Since this condition is unlikely to be used, it's best to leave it for now
        # def test_AtCall(self):
        #     A = 'A'
        #     B = 'B'

        #     sched = gs.Scheduler(graph={A: set(), B: set()})
        #     sched.add_condition(A, gs.Always())
        #     sched.add_condition(B, gs.AtCall(A, 3))

        #     termination_conds = {}
        #     termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        #     termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AtPass(5)
        #     output = list(sched.run(termination_conds=termination_conds))

        #     expected_output = [A, A, set([A, B]), A, A]
        #     assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_AfterCall(self):
            A = 'A'
            B = 'B'
            sched = gs.Scheduler(graph={A: set(), B: set()})
            sched.add_condition(B, gs.AfterCall(A, 3))

            termination_conds = {}
            termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
            termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [A, A, A, set([A, B]), set([A, B])]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_AfterNCalls(self):
            A = 'A'
            B = 'B'
            sched = gs.Scheduler(graph={A: set(), B: set()})
            sched.add_condition(A, gs.Always())
            sched.add_condition(B, gs.AfterNCalls(A, 3))

            termination_conds = {}
            termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
            termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [A, A, set([A, B]), set([A, B]), set([A, B])]
            assert output == pytest.helpers.setify_expected_output(expected_output)

    class TestConvenience:

        def test_AtEnvironmentStateUpdateStart(self):
            A = 'A'
            B = 'B'

            sched = gs.Scheduler(graph={A: set(), B: {A}})
            sched.add_condition(B, gs.AtEnvironmentStateUpdateStart())

            termination_conds = {
                gs.TimeScale.ENVIRONMENT_STATE_UPDATE: gs.AtPass(3)
            }
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [A, B, A, A]
            assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_composite_condition_multi(self):
        A = 'A'
        B = 'B'
        C = 'C'
        sched = gs.Scheduler(graph={A: set(), B: {A}, C: {B}})

        sched.add_condition(A, gs.EveryNPasses(1))
        sched.add_condition(B, gs.EveryNCalls(A, 2))
        sched.add_condition(C, gs.All(
            gs.Any(
                gs.AfterPass(6),
                gs.AfterNCalls(B, 2)
            ),
            gs.Any(
                gs.AfterPass(2),
                gs.AfterNCalls(B, 3)
            )
        )
        )

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AfterNCalls(C, 3)
        output = list(sched.run(termination_conds=termination_conds))
        expected_output = [
            A, A, B, A, A, B, C, A, C, A, B, C
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_AfterNCallsCombined(self):
        A = SimpleTestNode('A')
        B = SimpleTestNode('B')
        C = SimpleTestNode('C')

        sched = gs.Scheduler(graph={A: set(), B: {A}, C: {B}})
        sched.add_condition(A, gs.EveryNPasses(1))
        sched.add_condition(B, gs.EveryNCalls(A, 2))
        sched.add_condition(C, gs.EveryNCalls(B, 2))

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AfterNCallsCombined(B, C, n=4)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            A, A, B, A, A, B, C, A, A, B
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_AllHaveRun(self):
        A = SimpleTestNode('A', False)
        B = SimpleTestNode('B')
        C = SimpleTestNode('C')

        sched = gs.Scheduler(graph={A: set(), B: {A}, C: {B}})
        sched.add_condition(A, gs.EveryNPasses(1))
        sched.add_condition(B, gs.EveryNCalls(A, 2))
        sched.add_condition(C, gs.EveryNCalls(B, 2))

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AllHaveRun(A, B, C)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            A, A, B, A, A, B, C
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_AllHaveRun_2(self):
        A = 'A'
        B = 'B'
        C = 'C'
        sched = gs.Scheduler(graph={A: set(), B: {A}, C: {B}})

        sched.add_condition(A, gs.EveryNPasses(1))
        sched.add_condition(B, gs.EveryNCalls(A, 2))
        sched.add_condition(C, gs.EveryNCalls(B, 2))

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AllHaveRun()
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            A, A, B, A, A, B, C
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    @pytest.mark.parametrize(
        'parameter, indices, default_variable, integration_rate, expected_results',
        [
            ('value', None, None, 1, [10]),
        ]
    )
    @pytest.mark.parametrize('threshold', [10, 10.0])
    def test_Threshold_parameters(
        self, parameter, indices, default_variable, integration_rate, expected_results, threshold,
    ):
        A = SimpleTestNode('A')
        sched = gs.Scheduler(graph={A: set()})

        sched.termination_conds = {
            gs.TimeScale.ENVIRONMENT_STATE_UPDATE: gs.Threshold(A, parameter, threshold, '>=', indices=indices)
        }

        results = pytest.helpers.run_scheduler(sched, lambda node: node.add(integration_rate))
        np.testing.assert_array_equal(results, expected_results)

    @pytest.mark.parametrize(
        'comparator, increment, threshold, expected_results',
        [
            ('>', 1, 5, [6]),
            ('>=', 1, 5, [5]),
            ('<', -1, -5, [-6]),
            ('<=', -1, -5, [-5]),
            ('==', 1, 5, [5]),
            ('!=', 1, 0, [1]),
            ('!=', -1, 0, [-1]),
        ]
    )
    def test_Threshold_comparators(
        self, comparator, increment, threshold, expected_results
    ):
        A = SimpleTestNode('A')
        sched = gs.Scheduler(graph={A: set()})

        sched.termination_conds = {
            gs.TimeScale.ENVIRONMENT_STATE_UPDATE: gs.Threshold(A, 'value', threshold, comparator)
        }

        results = pytest.helpers.run_scheduler(sched, lambda node: node.add(increment))
        np.testing.assert_array_equal(results, expected_results)

    def test_Threshold_custom_methods(self):
        class CustomDependency:
            def __init__(self, **parameters):
                self.parameters = parameters

            def has_parameter(self, parameter):
                return parameter in self.parameters

        d = CustomDependency(a=0, b=5)
        cond = gs.Threshold(
            d, 'a', 2, '>',
            custom_parameter_getter=lambda o, p: o.parameters[p],
            custom_parameter_validator=lambda o, p: o.has_parameter(p)
        )
        scheduler = gs.Scheduler({d: set()}, {})
        for _ in scheduler.run(
            termination_conds={gs.TimeScale.ENVIRONMENT_STATE_UPDATE: cond}
        ):
            d.parameters['a'] += 1

        assert d.parameters['a'] == 3

    @pytest.mark.parametrize(
        'comparator, increment, threshold, atol, rtol, expected_results',
        [
            ('==', 1, 10, 1, 0.1, [8]),
            ('==', 1, 10, 1, 0, [9]),
            ('==', 1, 10, 0, 0.1, [9]),
            ('!=', 1, 2, 1, 0.5, [5]),
            ('!=', 1, 1, 1, 0, [3]),
            ('!=', 1, 1, 0, 1, [3]),
            ('!=', -1, -2, 1, 0.5, [-5]),
            ('!=', -1, -1, 1, 0, [-3]),
            ('!=', -1, -1, 0, 1, [-3]),
        ]
    )
    def test_Threshold_tolerances(
        self, comparator, increment, threshold, atol, rtol, expected_results
    ):
        A = SimpleTestNode('A')
        sched = gs.Scheduler(graph={A: set()})

        sched.termination_conds = {
            gs.TimeScale.ENVIRONMENT_STATE_UPDATE: gs.Threshold(A, 'value', threshold, comparator, atol=atol, rtol=rtol)
        }

        results = pytest.helpers.run_scheduler(sched, lambda node: node.add(increment))
        np.testing.assert_array_equal(results, expected_results)


class TestWhenFinished:
    def test_WhenFinishedAny_1(self):
        A = SimpleTestNode('A')
        B = SimpleTestNode('B')
        C = SimpleTestNode('C')

        sched = gs.Scheduler(graph={A: set(), B: set(), C: {A, B}})
        sched.add_condition(A, gs.EveryNPasses(1))
        sched.add_condition(B, gs.EveryNPasses(1))
        sched.add_condition(C, gs.WhenFinishedAny(A, B))

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AfterNCalls(C, 1)
        output = list(sched.run(termination_conds=termination_conds))
        expected_output = [
            set([A, B]), C
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_WhenFinishedAny_2(self):
        A = SimpleTestNode('A', False)
        B = SimpleTestNode('B')
        C = SimpleTestNode('C')

        sched = gs.Scheduler(graph={A: set(), B: set(), C: {A, B}})
        sched.add_condition(A, gs.EveryNPasses(1))
        sched.add_condition(B, gs.EveryNPasses(1))
        sched.add_condition(C, gs.WhenFinishedAny(A, B))

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AfterNCalls(A, 5)
        output = list(sched.run(termination_conds=termination_conds))
        expected_output = [
            set([A, B]), C, set([A, B]), C, set([A, B]), C, set([A, B]), C, set([A, B])
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_WhenFinishedAny_noargs(self):
        A = SimpleTestNode('A', False)
        B = SimpleTestNode('B', False)
        C = SimpleTestNode('C', False)

        sched = gs.Scheduler(graph={A: set(), B: set(), C: {A, B}})
        sched.add_condition(A, gs.Always())
        sched.add_condition(B, gs.Always())
        sched.add_condition(C, gs.Always())

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.WhenFinishedAny()
        output = []
        i = 0
        for step in sched.run(termination_conds=termination_conds):
            if i == 3:
                A.is_finished_flag = True
                B.is_finished_flag = True
            if i == 4:
                C.is_finished_flag = True
            output.append(step)
            i += 1
        expected_output = [
            set([A, B]), C, set([A, B]), C,
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_WhenFinishedAll_1(self):
        A = SimpleTestNode('A')
        B = SimpleTestNode('B')
        C = SimpleTestNode('C')
        sched = gs.Scheduler(graph={A: set(), B: set(), C: {A, B}})
        sched.add_condition(A, gs.EveryNPasses(1))
        sched.add_condition(B, gs.EveryNPasses(1))
        sched.add_condition(C, gs.WhenFinishedAll(A, B))

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AfterNCalls(C, 1)
        output = list(sched.run(termination_conds=termination_conds))
        expected_output = [
            set([A, B]), C
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_WhenFinishedAll_2(self):
        A = SimpleTestNode('A', False)
        B = SimpleTestNode('B')
        C = SimpleTestNode('C')
        sched = gs.Scheduler(graph={A: set(), B: set(), C: {A, B}})
        sched.add_condition(A, gs.EveryNPasses(1))
        sched.add_condition(B, gs.EveryNPasses(1))
        sched.add_condition(C, gs.WhenFinishedAll(A, B))

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AfterNCalls(A, 5)
        output = list(sched.run(termination_conds=termination_conds))
        expected_output = [
            set([A, B]), set([A, B]), set([A, B]), set([A, B]), set([A, B]),
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_WhenFinishedAll_noargs(self):
        A = SimpleTestNode('A', False)
        B = SimpleTestNode('B', False)
        C = SimpleTestNode('C', False)

        sched = gs.Scheduler(graph={A: set(), B: set(), C: {A, B}})
        sched.add_condition(A, gs.Always())
        sched.add_condition(B, gs.Always())
        sched.add_condition(C, gs.Always())

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.WhenFinishedAll()
        output = []
        i = 0
        for step in sched.run(termination_conds=termination_conds):
            if i == 3:
                A.is_finished_flag = True
                B.is_finished_flag = True
            if i == 4:
                C.is_finished_flag = True
            output.append(step)
            i += 1
        expected_output = [
            set([A, B]), C, set([A, B]), C, set([A, B]),
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)


class TestAbsolute:
    A = SimpleTestNode('A')
    B = SimpleTestNode('B')
    C = SimpleTestNode('C')

    @pytest.mark.parametrize(
        'conditions, termination_conds',
        [
            (
                {A: gs.TimeInterval(repeat=8), B: gs.TimeInterval(repeat=4), C: gs.TimeInterval(repeat=2)},
                {gs.TimeScale.ENVIRONMENT_STATE_UPDATE: gs.AfterNCalls(A, 2)},
            ),
            (
                {A: gs.TimeInterval(repeat=5), B: gs.TimeInterval(repeat=3), C: gs.TimeInterval(repeat=1)},
                {gs.TimeScale.ENVIRONMENT_STATE_UPDATE: gs.AfterNCalls(A, 2)},
            ),
            (
                {A: gs.TimeInterval(repeat=3), B: gs.TimeInterval(repeat=2)},
                {gs.TimeScale.ENVIRONMENT_STATE_UPDATE: gs.AfterNCalls(A, 2)},
            ),
            (
                {A: gs.TimeInterval(repeat=5), B: gs.TimeInterval(repeat=7)},
                {gs.TimeScale.ENVIRONMENT_STATE_UPDATE: gs.AfterNCalls(B, 2)},
            ),
            (
                {A: gs.TimeInterval(repeat=1200), B: gs.TimeInterval(repeat=1000)},
                {gs.TimeScale.ENVIRONMENT_STATE_UPDATE: gs.AfterNCalls(A, 3)},
            ),
            (
                {A: gs.TimeInterval(repeat=0.33333), B: gs.TimeInterval(repeat=0.66666)},
                {gs.TimeScale.ENVIRONMENT_STATE_UPDATE: gs.AfterNCalls(B, 3)},
            ),
            # smaller than default units cause floating point issue without mitigation
            (
                {A: gs.TimeInterval(repeat=2 * gs._unit_registry.us), B: gs.TimeInterval(repeat=4 * gs._unit_registry.us)},
                {gs.TimeScale.ENVIRONMENT_STATE_UPDATE: gs.AfterNCalls(B, 3)},
            ),
        ]
    )
    def test_TimeInterval_linear_everynms(self, conditions, termination_conds):
        sched = gs.Scheduler(graph=pytest.helpers.create_graph_from_pathways([self.A, self.B, self.C]))
        sched.add_condition_set(conditions)

        list(sched.run(termination_conds=termination_conds))

        for node, cond in conditions.items():
            executions = [
                sched.execution_timestamps[sched.default_execution_id][i].absolute
                for i in range(len(sched.execution_list[sched.default_execution_id]))
                if node in sched.execution_list[sched.default_execution_id][i]
            ]

            for i in range(1, len(executions)):
                assert (executions[i] - executions[i - 1]) == cond.repeat

    @pytest.mark.parametrize(
        'conditions, termination_conds',
        [
            (
                {
                    A: gs.TimeInterval(repeat=10, start=100),
                    B: gs.TimeInterval(repeat=10, start=300),
                    C: gs.TimeInterval(repeat=10, start=400)
                },
                {gs.TimeScale.ENVIRONMENT_STATE_UPDATE: gs.TimeInterval(start=500)}
            ),
            (
                {
                    A: gs.TimeInterval(start=100),
                    B: gs.TimeInterval(start=300),
                    C: gs.TimeInterval(start=400)
                },
                {gs.TimeScale.ENVIRONMENT_STATE_UPDATE: gs.TimeInterval(start=500)}
            ),
            (
                {
                    A: gs.TimeInterval(repeat=2, start=105),
                    B: gs.TimeInterval(repeat=7, start=317),
                    C: gs.TimeInterval(repeat=11, start=431)
                },
                {gs.TimeScale.ENVIRONMENT_STATE_UPDATE: gs.TimeInterval(start=597)}
            ),
            (
                {
                    A: gs.TimeInterval(repeat=10, start=100, start_inclusive=False),
                    B: gs.TimeInterval(repeat=10, start=300),
                    C: gs.TimeInterval(repeat=10, start=400)
                },
                {gs.TimeScale.ENVIRONMENT_STATE_UPDATE: gs.TimeInterval(start=500)}
            ),
            (
                {
                    A: gs.TimeInterval(repeat=10, start=100, start_inclusive=False),
                    B: gs.TimeInterval(repeat=10, start=300),
                    C: gs.TimeInterval(repeat=10, start=400)
                },
                {gs.TimeScale.ENVIRONMENT_STATE_UPDATE: gs.TimeInterval(start=500, start_inclusive=False)}
            ),
            (
                {
                    A: gs.TimeInterval(repeat=10, start=100),
                    B: gs.TimeInterval(repeat=10, start=100, end=200),
                    C: gs.TimeInterval(repeat=10, start=400)
                },
                {gs.TimeScale.ENVIRONMENT_STATE_UPDATE: gs.TimeInterval(start=500, start_inclusive=False)}
            ),
            (
                {
                    A: gs.TimeInterval(repeat=10, start=100),
                    B: gs.TimeInterval(repeat=10, start=100, end=200, end_inclusive=False),
                    C: gs.TimeInterval(repeat=10, start=400)
                },
                {gs.TimeScale.ENVIRONMENT_STATE_UPDATE: gs.TimeInterval(start=500)}
            ),
        ]
    )
    def test_TimeInterval_no_dependencies(self, conditions, termination_conds):
        sched = gs.Scheduler(graph={self.A: set(), self.B: set(), self.C: set()})
        sched.add_condition_set(conditions)
        consideration_set_execution_abs_value = sched._get_absolute_consideration_set_execution_unit(termination_conds)

        list(sched.run(termination_conds=termination_conds))

        for node, cond in conditions.items():
            executions = [
                sched.execution_timestamps[sched.default_execution_id][i].absolute
                for i in range(len(sched.execution_list[sched.default_execution_id]))
                if node in sched.execution_list[sched.default_execution_id][i]
            ]

            for i in range(1, len(executions)):
                interval = (executions[i] - executions[i - 1])

                if cond.repeat is not None:
                    assert interval == cond.repeat
                else:
                    assert interval == consideration_set_execution_abs_value

            if cond.start is not None:
                if cond.start_inclusive:
                    assert cond.start in executions
                else:
                    assert cond.start + consideration_set_execution_abs_value in executions

        # this test only runs a single ENVIRONMENT_STATE_UPDATE, so this
        # timestamp corresponds to its last
        final_timestamp = sched.execution_timestamps[sched.default_execution_id][-1].absolute
        term_cond = termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE]

        if term_cond.start_inclusive:
            assert term_cond.start - consideration_set_execution_abs_value == final_timestamp
        else:
            assert term_cond.start == final_timestamp

    @pytest.mark.parametrize(
        'repeat, unit, expected_repeat',
        [
            (1, None, 1 * gs._unit_registry.ms),
            ('1ms', None, 1 * gs._unit_registry.ms),
            (1 * gs._unit_registry.ms, None, 1 * gs._unit_registry.ms),
            (1, 'ms', 1 * gs._unit_registry.ms),
            (1, gs._unit_registry.ms, 1 * gs._unit_registry.ms),
            ('1', gs._unit_registry.ms, 1 * gs._unit_registry.ms),
            (1 * gs._unit_registry.ms, gs._unit_registry.ns, 1 * gs._unit_registry.ms),
            (1000 * gs._unit_registry.ms, None, 1000 * gs._unit_registry.ms),
        ]
    )
    def test_TimeInterval_time_specs(self, repeat, unit, expected_repeat):
        if unit is None:
            c = gs.TimeInterval(repeat=repeat)
        else:
            c = gs.TimeInterval(repeat=repeat, unit=unit)

        assert c.repeat == expected_repeat

    @pytest.mark.parametrize(
        'repeat, inclusive, last_time',
        [
            (10, True, 10 * gs._unit_registry.ms),
            (10, False, 11 * gs._unit_registry.ms),
        ]
    )
    def test_TimeTermination(
        self,
        three_node_linear_scheduler,
        repeat,
        inclusive,
        last_time
    ):
        _, sched = three_node_linear_scheduler

        sched.termination_conds = {
            gs.TimeScale.ENVIRONMENT_STATE_UPDATE: gs.TimeTermination(repeat, inclusive)
        }
        list(sched.run())

        assert sched.get_clock(sched.default_execution_id).time.absolute == last_time
