import logging

import graph_scheduler as gs
import numpy as np
import psyneulink as pnl
import pytest


logger = logging.getLogger(__name__)


class TestCondition:

    def test_invalid_input_WhenFinished(self):
        with pytest.raises(pnl.ConditionError):
            pnl.WhenFinished(None).is_satisfied()

    def test_invalid_input_WhenFinishedAny_1(self):
        with pytest.raises(pnl.ConditionError):
            pnl.WhenFinished(None).is_satisfied()

    def test_invalid_input_WhenFinishedAny_2(self):
        with pytest.raises(pnl.ConditionError):
            pnl.WhenFinished({None}).is_satisfied()

    def test_invalid_input_WhenFinishedAll_1(self):
        with pytest.raises(pnl.ConditionError):
            pnl.WhenFinished(None).is_satisfied()

    def test_invalid_input_WhenFinishedAll_2(self):
        with pytest.raises(pnl.ConditionError):
            pnl.WhenFinished({None}).is_satisfied()

    def test_additional_args(self):
        class OneSatisfied(pnl.Condition):
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
        class OneSatisfied(pnl.Condition):
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

    @pytest.mark.psyneulink
    class TestGeneric:
        def test_WhileNot_AtPass(self):
            comp = pnl.Composition()
            A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A')
            comp.add_node(A)

            sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))
            sched.add_condition(A, pnl.WhileNot(lambda sched: sched.get_clock(sched.default_execution_id).get_total_times_relative(pnl.TimeScale.PASS, pnl.TimeScale.ENVIRONMENT_STATE_UPDATE) == 0, sched))

            termination_conds = {}
            termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
            termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [set(), A, A, A, A]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_WhileNot_AtPass_in_middle(self):
            comp = pnl.Composition()
            A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A')
            comp.add_node(A)

            sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))
            sched.add_condition(A, pnl.WhileNot(lambda sched: sched.get_clock(sched.default_execution_id).get_total_times_relative(pnl.TimeScale.PASS, pnl.TimeScale.ENVIRONMENT_STATE_UPDATE) == 2, sched))

            termination_conds = {}
            termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
            termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [A, A, set(), A, A]
            assert output == pytest.helpers.setify_expected_output(expected_output)

    @pytest.mark.psyneulink
    class TestRelative:

        def test_Any_end_before_one_finished(self):
            comp = pnl.Composition()
            A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A')
            for m in [A]:
                comp.add_node(m)
            sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

            sched.add_condition(A, pnl.EveryNPasses(1))

            termination_conds = {}
            termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
            termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.Any(pnl.AfterNCalls(A, 10), pnl.AtPass(5))
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [A for _ in range(5)]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_All_end_after_one_finished(self):
            comp = pnl.Composition()
            A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A')
            for m in [A]:
                comp.add_node(m)
            sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

            sched.add_condition(A, pnl.EveryNPasses(1))

            termination_conds = {}
            termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
            termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.Any(pnl.AfterNCalls(A, 5), pnl.AtPass(10))
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [A for _ in range(5)]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_Not_AtPass(self):
            comp = pnl.Composition()
            A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A')
            comp.add_node(A)

            sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))
            sched.add_condition(A, pnl.Not(pnl.AtPass(0)))

            termination_conds = {}
            termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
            termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [set(), A, A, A, A]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_Not_AtPass_in_middle(self):
            comp = pnl.Composition()
            A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A')
            comp.add_node(A)

            sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))
            sched.add_condition(A, pnl.Not(pnl.AtPass(2)))

            termination_conds = {}
            termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
            termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AtPass(5)
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
            comp = pnl.Composition()
            A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A')
            B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='B')
            for m in [A, B]:
                comp.add_node(m)
            comp.add_projection(pnl.MappingProjection(), A, B)

            sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))
            sched.add_condition(A, pnl.Always())
            sched.add_condition(B, pnl.NWhen(pnl.AfterNCalls(A, 3), n))

            termination_conds = {}
            termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
            termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AfterNCalls(A, 6)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [A if x == 'A' else B for x in expected_output]

            assert output == pytest.helpers.setify_expected_output(expected_output)

    @pytest.mark.psyneulink
    class TestTimePNL:

        def test_BeforeConsiderationSetExecution(self):
            comp = pnl.Composition()
            A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A')
            comp.add_node(A)

            sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))
            sched.add_condition(A, pnl.BeforeConsiderationSetExecution(2))

            termination_conds = {}
            termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
            termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [A, A, set(), set(), set()]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_BeforeConsiderationSetExecution_2(self):
            comp = pnl.Composition()
            A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A')
            B = pnl.TransferMechanism(name='B')
            comp.add_node(A)
            comp.add_node(B)

            comp.add_projection(pnl.MappingProjection(), A, B)

            sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))
            sched.add_condition(A, pnl.BeforeConsiderationSetExecution(2))
            sched.add_condition(B, pnl.Always())

            termination_conds = {}
            termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
            termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [A, B, B, B, B, B]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_AtConsiderationSetExecution(self):
            comp = pnl.Composition()
            A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A')
            comp.add_node(A)

            sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))
            sched.add_condition(A, pnl.AtConsiderationSetExecution(0))

            termination_conds = {}
            termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
            termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [A, set(), set(), set(), set()]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_BeforePass(self):
            comp = pnl.Composition()
            A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A')
            comp.add_node(A)

            sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))
            sched.add_condition(A, pnl.BeforePass(2))

            termination_conds = {}
            termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
            termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [A, A, set(), set(), set()]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_AtPass(self):
            comp = pnl.Composition()
            A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A')
            comp.add_node(A)

            sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))
            sched.add_condition(A, pnl.AtPass(0))

            termination_conds = {}
            termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
            termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [A, set(), set(), set(), set()]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_AtPass_underconstrained(self):
            comp = pnl.Composition()
            A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A')
            B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='B')
            C = pnl.TransferMechanism(function=pnl.Linear(intercept=1.5), name='C')
            for m in [A, B, C]:
                comp.add_node(m)
            comp.add_projection(pnl.MappingProjection(), A, B)
            comp.add_projection(pnl.MappingProjection(), B, C)

            sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))
            sched.add_condition(A, pnl.AtPass(0))
            sched.add_condition(B, pnl.Always())
            sched.add_condition(C, pnl.Always())

            termination_conds = {}
            termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
            termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AfterNCalls(C, 2)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [A, B, C, B, C]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_AtPass_in_middle(self):
            comp = pnl.Composition()
            A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A')
            comp.add_node(A)

            sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))
            sched.add_condition(A, pnl.AtPass(2))

            termination_conds = {}
            termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
            termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [set(), set(), A, set(), set()]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_AtPass_at_end(self):
            comp = pnl.Composition()
            A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A')
            comp.add_node(A)

            sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))
            sched.add_condition(A, pnl.AtPass(5))

            termination_conds = {}
            termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
            termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [set(), set(), set(), set(), set()]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_AtPass_after_end(self):
            comp = pnl.Composition()
            A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A')
            comp.add_node(A)

            sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))
            sched.add_condition(A, pnl.AtPass(6))

            termination_conds = {}
            termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
            termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [set(), set(), set(), set(), set()]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_AfterPass(self):
            comp = pnl.Composition()
            A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A')
            comp.add_node(A)

            sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))
            sched.add_condition(A, pnl.AfterPass(0))

            termination_conds = {}
            termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
            termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [set(), A, A, A, A]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_AfterNPasses(self):
            comp = pnl.Composition()
            A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A')
            comp.add_node(A)

            sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))
            sched.add_condition(A, pnl.AfterNPasses(1))

            termination_conds = {}
            termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
            termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [set(), A, A, A, A]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_BeforeEnvironmentStateUpdate(self):
            comp = pnl.Composition()
            A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A')
            comp.add_node(A)

            sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))
            sched.add_condition(A, pnl.BeforeEnvironmentStateUpdate(4))

            termination_conds = {}
            termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(5)
            termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AtPass(1)
            comp.run(
                inputs={A: range(6)},
                scheduler=sched,
                termination_processing=termination_conds
            )
            output = sched.execution_list[comp.default_execution_id]

            expected_output = [A, A, A, A, set()]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_AtEnvironmentStateUpdate(self):
            comp = pnl.Composition()
            A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A')
            comp.add_node(A)

            sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))
            sched.add_condition(A, pnl.Always())

            termination_conds = {}
            termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AtEnvironmentStateUpdate(4)
            termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AtPass(1)
            comp.run(
                inputs={A: range(6)},
                scheduler=sched,
                termination_processing=termination_conds
            )
            output = sched.execution_list[comp.default_execution_id]

            expected_output = [A, A, A, A]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_AfterEnvironmentStateUpdate(self):
            comp = pnl.Composition()
            A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A')
            comp.add_node(A)

            sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))
            sched.add_condition(A, pnl.Always())

            termination_conds = {}
            termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterEnvironmentStateUpdate(4)
            termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AtPass(1)
            comp.run(
                inputs={A: range(6)},
                scheduler=sched,
                termination_processing=termination_conds
            )
            output = sched.execution_list[comp.default_execution_id]

            expected_output = [A, A, A, A, A]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_AfterNEnvironmentStateUpdates(self):
            comp = pnl.Composition()
            A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A')
            comp.add_node(A)

            sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))
            sched.add_condition(A, pnl.AfterNPasses(1))

            termination_conds = {}
            termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
            termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AtPass(5)
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
                        pnl.TimeScale.ENVIRONMENT_SEQUENCE: gs.AfterNEnvironmentStateUpdates(1),
                        pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: gs.AfterConsiderationSetExecution(4)
                    },
                    [set(), 'A', 'A', 'A', 'A'],
                    id='AfterConsiderationSetExecution'
                ),
                pytest.param(
                    gs.AfterNPasses(1),
                    {
                        pnl.TimeScale.ENVIRONMENT_SEQUENCE: gs.AfterNEnvironmentStateUpdates(1),
                        pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: gs.AfterNConsiderationSetExecutions(5)
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
                    {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: gs.AfterNPasses(1)},
                    [[set(), set(), 'A', set()]],
                    1,
                    4,
                    id='AtEnvironmentStateUpdateNStart'
                ),
                pytest.param(
                    gs.AtEnvironmentSequence(4),
                    {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: gs.AfterNPasses(1)},
                    [[set()], [set()], [set()], [set()], ['A'], [set()]],
                    6,
                    1,
                    id='AtEnvironmentSequence'
                ),
                pytest.param(
                    gs.AfterEnvironmentSequence(3),
                    {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: gs.AfterNPasses(1)},
                    [[set()], [set()], [set()], [set()], ['A'], ['A']],
                    6,
                    1,
                    id='AfterEnvironmentSequence'
                ),
                pytest.param(
                    gs.AfterNEnvironmentSequences(4),
                    {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: gs.AfterNPasses(1)},
                    [[set()], [set()], [set()], [set()], ['A'], ['A']],
                    6,
                    1,
                    id='AfterNEnvironmentSequences'
                ),
                pytest.param(
                    gs.AtEnvironmentSequenceStart(),
                    {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: gs.AfterNPasses(1)},
                    [['A', set()], ['A', set()]],
                    2,
                    2,
                    id='AtEnvironmentSequenceStart'
                ),
                pytest.param(
                    gs.AtEnvironmentSequenceNStart(1),
                    {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: gs.AfterNPasses(1)},
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

    @pytest.mark.psyneulink
    class TestComponentBased:

        def test_BeforeNCalls(self):
            comp = pnl.Composition()
            A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A')
            comp.add_node(A)

            sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))
            sched.add_condition(A, pnl.BeforeNCalls(A, 3))

            termination_conds = {}
            termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
            termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [A, A, A, set(), set()]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        # NOTE:
        # The behavior is not desired (i.e. depending on the order mechanisms are checked, B running AtCall(A, x))
        # may run on both the xth and x+1st call of A; if A and B are not parent-child
        # A fix could invalidate key assumptions and affect many other conditions
        # Since this condition is unlikely to be used, it's best to leave it for now
        # def test_AtCall(self):
        #     comp = pnl.Composition()
        #     A = pnl.TransferMechanism(function = pnl.Linear(slope=5.0, intercept = 2.0), name = 'A')
        #     B = pnl.TransferMechanism(function = pnl.Linear(intercept = 4.0), name = 'B')
        #     C = pnl.TransferMechanism(function = pnl.Linear(intercept = 1.5), name = 'C')
        #     for m in [A,B]:
        #         comp.add_node(m)

        #     sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))
        #     sched.add_condition(A, pnl.Always())
        #     sched.add_condition(B, AtCall(A, 3))

        #     termination_conds = {}
        #     termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
        #     termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AtPass(5)
        #     output = list(sched.run(termination_conds=termination_conds))

        #     expected_output = [A, A, set([A, B]), A, A]
        #     assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_AfterCall(self):
            comp = pnl.Composition()
            A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A')
            B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='B')
            for m in [A, B]:
                comp.add_node(m)

            sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))
            sched.add_condition(B, pnl.AfterCall(A, 3))

            termination_conds = {}
            termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
            termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [A, A, A, set([A, B]), set([A, B])]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_AfterNCalls(self):
            comp = pnl.Composition()
            A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A')
            B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='B')
            for m in [A, B]:
                comp.add_node(m)

            sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))
            sched.add_condition(A, pnl.Always())
            sched.add_condition(B, pnl.AfterNCalls(A, 3))

            termination_conds = {}
            termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
            termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [A, A, set([A, B]), set([A, B]), set([A, B])]
            assert output == pytest.helpers.setify_expected_output(expected_output)

    @pytest.mark.psyneulink
    class TestConvenience:

        def test_AtEnvironmentStateUpdateStart(self):
            comp = pnl.Composition()
            A = pnl.TransferMechanism(name='A')
            B = pnl.TransferMechanism(name='B')
            comp.add_linear_processing_pathway([A, B])

            sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))
            sched.add_condition(B, pnl.AtEnvironmentStateUpdateStart())

            termination_conds = {
                pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.AtPass(3)
            }
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [A, B, A, A]
            assert output == pytest.helpers.setify_expected_output(expected_output)

    @pytest.mark.psyneulink
    def test_composite_condition_multi(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A')
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='B')
        C = pnl.TransferMechanism(function=pnl.Linear(intercept=1.5), name='C')
        for m in [A, B, C]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, B)
        comp.add_projection(pnl.MappingProjection(), B, C)
        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(A, pnl.EveryNPasses(1))
        sched.add_condition(B, pnl.EveryNCalls(A, 2))
        sched.add_condition(C, pnl.All(
            pnl.Any(
                pnl.AfterPass(6),
                pnl.AfterNCalls(B, 2)
            ),
            pnl.Any(
                pnl.AfterPass(2),
                pnl.AfterNCalls(B, 3)
            )
        )
        )

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AfterNCalls(C, 3)
        output = list(sched.run(termination_conds=termination_conds))
        expected_output = [
            A, A, B, A, A, B, C, A, C, A, B, C
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    @pytest.mark.psyneulink
    def test_AfterNCallsCombined(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A')
        A.is_finished_flag = True
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='B')
        B.is_finished_flag = True
        C = pnl.TransferMechanism(function=pnl.Linear(intercept=1.5), name='C')
        for m in [A, B, C]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, B)
        comp.add_projection(pnl.MappingProjection(), B, C)
        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(A, pnl.EveryNPasses(1))
        sched.add_condition(B, pnl.EveryNCalls(A, 2))
        sched.add_condition(C, pnl.EveryNCalls(B, 2))

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AfterNCallsCombined(B, C, n=4)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            A, A, B, A, A, B, C, A, A, B
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    @pytest.mark.psyneulink
    def test_AllHaveRun(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A')
        A.is_finished_flag = False
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='B')
        B.is_finished_flag = True
        C = pnl.TransferMechanism(function=pnl.Linear(intercept=1.5), name='C')
        for m in [A, B, C]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, B)
        comp.add_projection(pnl.MappingProjection(), B, C)
        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(A, pnl.EveryNPasses(1))
        sched.add_condition(B, pnl.EveryNCalls(A, 2))
        sched.add_condition(C, pnl.EveryNCalls(B, 2))

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AllHaveRun(A, B, C)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            A, A, B, A, A, B, C
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    @pytest.mark.psyneulink
    def test_AllHaveRun_2(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A')
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='B')
        C = pnl.TransferMechanism(function=pnl.Linear(intercept=1.5), name='C')
        for m in [A, B, C]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, B)
        comp.add_projection(pnl.MappingProjection(), B, C)
        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(A, pnl.EveryNPasses(1))
        sched.add_condition(B, pnl.EveryNCalls(A, 2))
        sched.add_condition(C, pnl.EveryNCalls(B, 2))

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AllHaveRun()
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            A, A, B, A, A, B, C
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    @pytest.mark.psyneulink
    @pytest.mark.parametrize(
        'parameter, indices, default_variable, integration_rate, expected_results',
        [
            ('value', None, None, 1, [[[10]]]),
            ('value', (0, 0), [[0, 0]], [1, 2], [[[10, 20]]]),
            ('value', (0, 1), [[0, 0]], [1, 2], [[[5, 10]]]),
            ('num_executions', pnl.TimeScale.TRIAL, None, 1, [[[10]]]),
        ]
    )
    @pytest.mark.parametrize('threshold', [10, 10.0])
    def test_Threshold_parameters(
        self, parameter, indices, default_variable, integration_rate, expected_results, threshold,
    ):

        A = pnl.TransferMechanism(
            default_variable=default_variable,
            integrator_mode=True,
            integrator_function=pnl.SimpleIntegrator,
            integration_rate=integration_rate,
        )
        comp = pnl.Composition(pathways=[A])

        comp.termination_processing = {
            pnl.TimeScale.TRIAL: gs.Threshold(A, parameter, threshold, '>=', indices=indices)
        }

        comp.run(inputs={A: np.ones(A.defaults.variable.shape)})

        np.testing.assert_array_equal(comp.results, expected_results)

    @pytest.mark.psyneulink
    @pytest.mark.parametrize(
        'comparator, increment, threshold, expected_results',
        [
            ('>', 1, 5, [[[6]]]),
            ('>=', 1, 5, [[[5]]]),
            ('<', -1, -5, [[[-6]]]),
            ('<=', -1, -5, [[[-5]]]),
            ('==', 1, 5, [[[5]]]),
            ('!=', 1, 0, [[[1]]]),
            ('!=', -1, 0, [[[-1]]]),
        ]
    )
    def test_Threshold_comparators(
        self, comparator, increment, threshold, expected_results
    ):
        A = pnl.TransferMechanism(
            integrator_mode=True,
            integrator_function=pnl.AccumulatorIntegrator(rate=1, increment=increment),
        )
        comp = pnl.Composition(pathways=[A])

        comp.termination_processing = {
            pnl.TimeScale.TRIAL: gs.Threshold(A, 'value', threshold, comparator)
        }

        comp.run(inputs={A: np.ones(A.defaults.variable.shape)})

        np.testing.assert_array_equal(comp.results, expected_results)

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

    @pytest.mark.psyneulink
    @pytest.mark.parametrize(
        'comparator, increment, threshold, atol, rtol, expected_results',
        [
            ('==', 1, 10, 1, 0.1, [[[8]]]),
            ('==', 1, 10, 1, 0, [[[9]]]),
            ('==', 1, 10, 0, 0.1, [[[9]]]),
            ('!=', 1, 2, 1, 0.5, [[[5]]]),
            ('!=', 1, 1, 1, 0, [[[3]]]),
            ('!=', 1, 1, 0, 1, [[[3]]]),
            ('!=', -1, -2, 1, 0.5, [[[-5]]]),
            ('!=', -1, -1, 1, 0, [[[-3]]]),
            ('!=', -1, -1, 0, 1, [[[-3]]]),
        ]
    )
    def test_Threshold_tolerances(
        self, comparator, increment, threshold, atol, rtol, expected_results
    ):
        A = pnl.TransferMechanism(
            integrator_mode=True,
            integrator_function=pnl.AccumulatorIntegrator(rate=1, increment=increment),
        )
        comp = pnl.Composition(pathways=[A])

        comp.termination_processing = {
            pnl.TimeScale.TRIAL: gs.Threshold(A, 'value', threshold, comparator, atol=atol, rtol=rtol)
        }

        comp.run(inputs={A: np.ones(A.defaults.variable.shape)})

        np.testing.assert_array_equal(comp.results, expected_results)


@pytest.mark.psyneulink
class TestWhenFinished:

    @classmethod
    def setup_class(self):
        self.orig_is_finished_flag = pnl.TransferMechanism.is_finished_flag
        self.orig_is_finished = pnl.TransferMechanism.is_finished
        pnl.TransferMechanism.is_finished_flag = True
        pnl.TransferMechanism.is_finished = lambda self, context: self.is_finished_flag

    @classmethod
    def teardown_class(self):
        del pnl.TransferMechanism.is_finished_flag
        del pnl.TransferMechanism.is_finished
        pnl.TransferMechanism.is_finished_flag = self.orig_is_finished_flag
        pnl.TransferMechanism.is_finished = self.orig_is_finished

    def test_WhenFinishedAny_1(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A')
        A.is_finished_flag = True
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='B')
        B.is_finished_flag = True
        C = pnl.TransferMechanism(function=pnl.Linear(intercept=1.5), name='C')
        for m in [A, B, C]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, C)
        comp.add_projection(pnl.MappingProjection(), B, C)
        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(A, pnl.EveryNPasses(1))
        sched.add_condition(B, pnl.EveryNPasses(1))
        sched.add_condition(C, pnl.WhenFinishedAny(A, B))

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AfterNCalls(C, 1)
        output = list(sched.run(termination_conds=termination_conds))
        expected_output = [
            set([A, B]), C
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_WhenFinishedAny_2(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A')
        A.is_finished_flag = False
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='B')
        B.is_finished_flag = True
        C = pnl.TransferMechanism(function=pnl.Linear(intercept=1.5), name='C')
        for m in [A, B, C]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, C)
        comp.add_projection(pnl.MappingProjection(), B, C)
        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(A, pnl.EveryNPasses(1))
        sched.add_condition(B, pnl.EveryNPasses(1))
        sched.add_condition(C, pnl.WhenFinishedAny(A, B))

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AfterNCalls(A, 5)
        output = list(sched.run(termination_conds=termination_conds))
        expected_output = [
            set([A, B]), C, set([A, B]), C, set([A, B]), C, set([A, B]), C, set([A, B])
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_WhenFinishedAny_noargs(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A')
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='B')
        C = pnl.TransferMechanism(function=pnl.Linear(intercept=1.5), name='C')
        for m in [A, B, C]:
            m.is_finished_flag = False
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, C)
        comp.add_projection(pnl.MappingProjection(), B, C)
        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(A, pnl.Always())
        sched.add_condition(B, pnl.Always())
        sched.add_condition(C, pnl.Always())

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.WhenFinishedAny()
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
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A')
        A.is_finished_flag = True
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='B')
        B.is_finished_flag = True
        C = pnl.TransferMechanism(function=pnl.Linear(intercept=1.5), name='C')
        for m in [A, B, C]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, C)
        comp.add_projection(pnl.MappingProjection(), B, C)
        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(A, pnl.EveryNPasses(1))
        sched.add_condition(B, pnl.EveryNPasses(1))
        sched.add_condition(C, pnl.WhenFinishedAll(A, B))

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AfterNCalls(C, 1)
        output = list(sched.run(termination_conds=termination_conds))
        expected_output = [
            set([A, B]), C
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_WhenFinishedAll_2(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A')
        A.is_finished_flag = False
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='B')
        B.is_finished_flag = True
        C = pnl.TransferMechanism(function=pnl.Linear(intercept=1.5), name='C')
        for m in [A, B, C]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, C)
        comp.add_projection(pnl.MappingProjection(), B, C)
        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(A, pnl.EveryNPasses(1))
        sched.add_condition(B, pnl.EveryNPasses(1))
        sched.add_condition(C, pnl.WhenFinishedAll(A, B))

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AfterNCalls(A, 5)
        output = list(sched.run(termination_conds=termination_conds))
        expected_output = [
            set([A, B]), set([A, B]), set([A, B]), set([A, B]), set([A, B]),
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_WhenFinishedAll_noargs(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A')
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='B')
        C = pnl.TransferMechanism(function=pnl.Linear(intercept=1.5), name='C')
        for m in [A, B, C]:
            comp.add_node(m)
            m.is_finished_flag = False
        comp.add_projection(pnl.MappingProjection(), A, C)
        comp.add_projection(pnl.MappingProjection(), B, C)
        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(A, pnl.Always())
        sched.add_condition(B, pnl.Always())
        sched.add_condition(C, pnl.Always())

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.WhenFinishedAll()
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


@pytest.mark.psyneulink
class TestAbsolute:
    A = pnl.TransferMechanism(name='scheduler-pytests-A')
    B = pnl.TransferMechanism(name='scheduler-pytests-B')
    C = pnl.TransferMechanism(name='scheduler-pytests-C')

    @pytest.mark.parametrize(
        'conditions, termination_conds',
        [
            (
                {A: pnl.TimeInterval(repeat=8), B: pnl.TimeInterval(repeat=4), C: pnl.TimeInterval(repeat=2)},
                {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.AfterNCalls(A, 2)},
            ),
            (
                {A: pnl.TimeInterval(repeat=5), B: pnl.TimeInterval(repeat=3), C: pnl.TimeInterval(repeat=1)},
                {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.AfterNCalls(A, 2)},
            ),
            (
                {A: pnl.TimeInterval(repeat=3), B: pnl.TimeInterval(repeat=2)},
                {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.AfterNCalls(A, 2)},
            ),
            (
                {A: pnl.TimeInterval(repeat=5), B: pnl.TimeInterval(repeat=7)},
                {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.AfterNCalls(B, 2)},
            ),
            (
                {A: pnl.TimeInterval(repeat=1200), B: pnl.TimeInterval(repeat=1000)},
                {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.AfterNCalls(A, 3)},
            ),
            (
                {A: pnl.TimeInterval(repeat=0.33333), B: pnl.TimeInterval(repeat=0.66666)},
                {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.AfterNCalls(B, 3)},
            ),
            # smaller than default units cause floating point issue without mitigation
            (
                {A: pnl.TimeInterval(repeat=2 * pnl._unit_registry.us), B: pnl.TimeInterval(repeat=4 * pnl._unit_registry.us)},
                {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.AfterNCalls(B, 3)},
            ),
        ]
    )
    def test_TimeInterval_linear_everynms(self, conditions, termination_conds):
        comp = pnl.Composition()

        comp.add_linear_processing_pathway([self.A, self.B, self.C])
        comp.scheduler.add_condition_set(conditions)

        list(comp.scheduler.run(termination_conds=termination_conds))

        for node, cond in conditions.items():
            executions = [
                comp.scheduler.execution_timestamps[comp.default_execution_id][i].absolute
                for i in range(len(comp.scheduler.execution_list[comp.default_execution_id]))
                if node in comp.scheduler.execution_list[comp.default_execution_id][i]
            ]

            for i in range(1, len(executions)):
                assert (executions[i] - executions[i - 1]) == cond.repeat

    @pytest.mark.parametrize(
        'conditions, termination_conds',
        [
            (
                {
                    A: pnl.TimeInterval(repeat=10, start=100),
                    B: pnl.TimeInterval(repeat=10, start=300),
                    C: pnl.TimeInterval(repeat=10, start=400)
                },
                {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.TimeInterval(start=500)}
            ),
            (
                {
                    A: pnl.TimeInterval(start=100),
                    B: pnl.TimeInterval(start=300),
                    C: pnl.TimeInterval(start=400)
                },
                {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.TimeInterval(start=500)}
            ),
            (
                {
                    A: pnl.TimeInterval(repeat=2, start=105),
                    B: pnl.TimeInterval(repeat=7, start=317),
                    C: pnl.TimeInterval(repeat=11, start=431)
                },
                {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.TimeInterval(start=597)}
            ),
            (
                {
                    A: pnl.TimeInterval(repeat=10, start=100, start_inclusive=False),
                    B: pnl.TimeInterval(repeat=10, start=300),
                    C: pnl.TimeInterval(repeat=10, start=400)
                },
                {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.TimeInterval(start=500)}
            ),
            (
                {
                    A: pnl.TimeInterval(repeat=10, start=100, start_inclusive=False),
                    B: pnl.TimeInterval(repeat=10, start=300),
                    C: pnl.TimeInterval(repeat=10, start=400)
                },
                {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.TimeInterval(start=500, start_inclusive=False)}
            ),
            (
                {
                    A: pnl.TimeInterval(repeat=10, start=100),
                    B: pnl.TimeInterval(repeat=10, start=100, end=200),
                    C: pnl.TimeInterval(repeat=10, start=400)
                },
                {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.TimeInterval(start=500, start_inclusive=False)}
            ),
            (
                {
                    A: pnl.TimeInterval(repeat=10, start=100),
                    B: pnl.TimeInterval(repeat=10, start=100, end=200, end_inclusive=False),
                    C: pnl.TimeInterval(repeat=10, start=400)
                },
                {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.TimeInterval(start=500)}
            ),
        ]
    )
    def test_TimeInterval_no_dependencies(self, conditions, termination_conds):
        comp = pnl.Composition()
        comp.add_nodes([self.A, self.B, self.C])
        comp.scheduler.add_condition_set(conditions)
        consideration_set_execution_abs_value = comp.scheduler._get_absolute_consideration_set_execution_unit(termination_conds)

        list(comp.scheduler.run(termination_conds=termination_conds))

        for node, cond in conditions.items():
            executions = [
                comp.scheduler.execution_timestamps[comp.default_execution_id][i].absolute
                for i in range(len(comp.scheduler.execution_list[comp.default_execution_id]))
                if node in comp.scheduler.execution_list[comp.default_execution_id][i]
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
        final_timestamp = comp.scheduler.execution_timestamps[comp.default_execution_id][-1].absolute
        term_cond = termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE]

        if term_cond.start_inclusive:
            assert term_cond.start - consideration_set_execution_abs_value == final_timestamp
        else:
            assert term_cond.start == final_timestamp

    @pytest.mark.parametrize(
        'repeat, unit, expected_repeat',
        [
            (1, None, 1 * pnl._unit_registry.ms),
            ('1ms', None, 1 * pnl._unit_registry.ms),
            (1 * pnl._unit_registry.ms, None, 1 * pnl._unit_registry.ms),
            (1, 'ms', 1 * pnl._unit_registry.ms),
            (1, pnl._unit_registry.ms, 1 * pnl._unit_registry.ms),
            ('1', pnl._unit_registry.ms, 1 * pnl._unit_registry.ms),
            (1 * pnl._unit_registry.ms, pnl._unit_registry.ns, 1 * pnl._unit_registry.ms),
            (1000 * pnl._unit_registry.ms, None, 1000 * pnl._unit_registry.ms),
        ]
    )
    def test_TimeInterval_time_specs(self, repeat, unit, expected_repeat):
        if unit is None:
            c = pnl.TimeInterval(repeat=repeat)
        else:
            c = pnl.TimeInterval(repeat=repeat, unit=unit)

        assert c.repeat == expected_repeat

    @pytest.mark.parametrize(
        'repeat, inclusive, last_time',
        [
            (10, True, 10 * pnl._unit_registry.ms),
            (10, False, 11 * pnl._unit_registry.ms),
        ]
    )
    def test_TimeTermination(
        self,
        three_node_linear_composition,
        repeat,
        inclusive,
        last_time
    ):
        _, comp = three_node_linear_composition

        comp.scheduler.termination_conds = {
            pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.TimeTermination(repeat, inclusive)
        }
        list(comp.scheduler.run())

        assert comp.scheduler.get_clock(comp.scheduler.default_execution_id).time.absolute == last_time
