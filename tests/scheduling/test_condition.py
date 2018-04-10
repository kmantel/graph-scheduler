import logging

import pytest

from psyneulink.components.functions.function import Linear
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.composition import Composition
from psyneulink.scheduling.condition import AfterCall, AfterNCalls, AfterNCallsCombined, AfterNPasses, AfterNTrials, AfterPass, AfterTrial, All, AllHaveRun, Always, Any, AtPass, AtTrial, BeforeNCalls, BeforePass, BeforeTrial, Condition, ConditionError, ConditionSet, EveryNCalls, EveryNPasses, NWhen, Not, WhenFinished, WhenFinishedAll, WhenFinishedAny, WhileNot
from psyneulink.scheduling.scheduler import Scheduler
from psyneulink.scheduling.time import TimeScale


logger = logging.getLogger(__name__)


class TestCondition:

    def test_invalid_input_WhenFinished(self):
        with pytest.raises(ConditionError):
            WhenFinished(None).is_satisfied()

    def test_invalid_input_WhenFinishedAny_1(self):
        with pytest.raises(ConditionError):
            WhenFinished(None).is_satisfied()

    def test_invalid_input_WhenFinishedAny_2(self):
        with pytest.raises(ConditionError):
            WhenFinished({None}).is_satisfied()

    def test_invalid_input_WhenFinishedAll_1(self):
        with pytest.raises(ConditionError):
            WhenFinished(None).is_satisfied()

    def test_invalid_input_WhenFinishedAll_2(self):
        with pytest.raises(ConditionError):
            WhenFinished({None}).is_satisfied()

    def test_additional_args(self):
        class OneSatisfied(Condition):
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
        class OneSatisfied(Condition):
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
            comp = Composition()
            A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
            comp.add_mechanism(A)

            sched = Scheduler(composition=comp)
            sched.add_condition(A, WhileNot(lambda sched: sched.clock.get_total_times_relative(TimeScale.PASS, TimeScale.TRIAL) == 0, sched))

            termination_conds = {}
            termination_conds[TimeScale.RUN] = AfterNTrials(1)
            termination_conds[TimeScale.TRIAL] = AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [set(), A, A, A, A]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_WhileNot_AtPass_in_middle(self):
            comp = Composition()
            A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
            comp.add_mechanism(A)

            sched = Scheduler(composition=comp)
            sched.add_condition(A, WhileNot(lambda sched: sched.clock.get_total_times_relative(TimeScale.PASS, TimeScale.TRIAL) == 2, sched))

            termination_conds = {}
            termination_conds[TimeScale.RUN] = AfterNTrials(1)
            termination_conds[TimeScale.TRIAL] = AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [A, A, set(), A, A]
            assert output == pytest.helpers.setify_expected_output(expected_output)

    class TestRelative:

        def test_Any_end_before_one_finished(self):
            comp = Composition()
            A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
            for m in [A]:
                comp.add_mechanism(m)
            sched = Scheduler(composition=comp)

            sched.add_condition(A, EveryNPasses(1))

            termination_conds = {}
            termination_conds[TimeScale.RUN] = AfterNTrials(1)
            termination_conds[TimeScale.TRIAL] = Any(AfterNCalls(A, 10), AtPass(5))
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [A for _ in range(5)]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_All_end_after_one_finished(self):
            comp = Composition()
            A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
            for m in [A]:
                comp.add_mechanism(m)
            sched = Scheduler(composition=comp)

            sched.add_condition(A, EveryNPasses(1))

            termination_conds = {}
            termination_conds[TimeScale.RUN] = AfterNTrials(1)
            termination_conds[TimeScale.TRIAL] = Any(AfterNCalls(A, 5), AtPass(10))
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [A for _ in range(5)]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_Not_AtPass(self):
            comp = Composition()
            A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
            comp.add_mechanism(A)

            sched = Scheduler(composition=comp)
            sched.add_condition(A, Not(AtPass(0)))

            termination_conds = {}
            termination_conds[TimeScale.RUN] = AfterNTrials(1)
            termination_conds[TimeScale.TRIAL] = AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [set(), A, A, A, A]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_Not_AtPass_in_middle(self):
            comp = Composition()
            A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
            comp.add_mechanism(A)

            sched = Scheduler(composition=comp)
            sched.add_condition(A, Not(AtPass(2)))

            termination_conds = {}
            termination_conds[TimeScale.RUN] = AfterNTrials(1)
            termination_conds[TimeScale.TRIAL] = AtPass(5)
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
            comp = Composition()
            A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
            B = TransferMechanism(function=Linear(intercept=4.0), name='B')
            for m in [A, B]:
                comp.add_mechanism(m)
            comp.add_projection(A, MappingProjection(), B)

            sched = Scheduler(composition=comp)
            sched.add_condition(A, Always())
            sched.add_condition(B, NWhen(AfterNCalls(A, 3), n))

            termination_conds = {}
            termination_conds[TimeScale.RUN] = AfterNTrials(1)
            termination_conds[TimeScale.TRIAL] = AfterNCalls(A, 6)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [A if x == 'A' else B for x in expected_output]

            assert output == pytest.helpers.setify_expected_output(expected_output)

    class TestTime:

        def test_BeforePass(self):
            comp = Composition()
            A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
            comp.add_mechanism(A)

            sched = Scheduler(composition=comp)
            sched.add_condition(A, BeforePass(2))

            termination_conds = {}
            termination_conds[TimeScale.RUN] = AfterNTrials(1)
            termination_conds[TimeScale.TRIAL] = AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [A, A, set(), set(), set()]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_AtPass(self):
            comp = Composition()
            A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
            comp.add_mechanism(A)

            sched = Scheduler(composition=comp)
            sched.add_condition(A, AtPass(0))

            termination_conds = {}
            termination_conds[TimeScale.RUN] = AfterNTrials(1)
            termination_conds[TimeScale.TRIAL] = AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [A, set(), set(), set(), set()]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_AtPass_underconstrained(self):
            comp = Composition()
            A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
            B = TransferMechanism(function=Linear(intercept=4.0), name='B')
            C = TransferMechanism(function=Linear(intercept=1.5), name='C')
            for m in [A, B, C]:
                comp.add_mechanism(m)
            comp.add_projection(A, MappingProjection(), B)
            comp.add_projection(B, MappingProjection(), C)

            sched = Scheduler(composition=comp)
            sched.add_condition(A, AtPass(0))

            termination_conds = {}
            termination_conds[TimeScale.RUN] = AfterNTrials(1)
            termination_conds[TimeScale.TRIAL] = AfterNCalls(C, 2)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [A, B, C, B, C]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_AtPass_in_middle(self):
            comp = Composition()
            A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
            comp.add_mechanism(A)

            sched = Scheduler(composition=comp)
            sched.add_condition(A, AtPass(2))

            termination_conds = {}
            termination_conds[TimeScale.RUN] = AfterNTrials(1)
            termination_conds[TimeScale.TRIAL] = AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [set(), set(), A, set(), set()]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_AtPass_at_end(self):
            comp = Composition()
            A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
            comp.add_mechanism(A)

            sched = Scheduler(composition=comp)
            sched.add_condition(A, AtPass(5))

            termination_conds = {}
            termination_conds[TimeScale.RUN] = AfterNTrials(1)
            termination_conds[TimeScale.TRIAL] = AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [set(), set(), set(), set(), set()]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_AtPass_after_end(self):
            comp = Composition()
            A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
            comp.add_mechanism(A)

            sched = Scheduler(composition=comp)
            sched.add_condition(A, AtPass(6))

            termination_conds = {}
            termination_conds[TimeScale.RUN] = AfterNTrials(1)
            termination_conds[TimeScale.TRIAL] = AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [set(), set(), set(), set(), set()]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_AfterPass(self):
            comp = Composition()
            A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
            comp.add_mechanism(A)

            sched = Scheduler(composition=comp)
            sched.add_condition(A, AfterPass(0))

            termination_conds = {}
            termination_conds[TimeScale.RUN] = AfterNTrials(1)
            termination_conds[TimeScale.TRIAL] = AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [set(), A, A, A, A]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_AfterNPasses(self):
            comp = Composition()
            A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
            comp.add_mechanism(A)

            sched = Scheduler(composition=comp)
            sched.add_condition(A, AfterNPasses(1))

            termination_conds = {}
            termination_conds[TimeScale.RUN] = AfterNTrials(1)
            termination_conds[TimeScale.TRIAL] = AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [set(), A, A, A, A]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_BeforeTrial(self):
            comp = Composition()
            A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
            comp.add_mechanism(A)

            sched = Scheduler(composition=comp)
            sched.add_condition(A, BeforeTrial(4))

            termination_conds = {}
            termination_conds[TimeScale.RUN] = AfterNTrials(5)
            termination_conds[TimeScale.TRIAL] = AtPass(1)
            comp.run(
                inputs={A: range(6)},
                scheduler_processing=sched,
                termination_processing=termination_conds
            )
            output = sched.execution_list[comp._execution_id]

            expected_output = [A, A, A, A, set()]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_AtTrial(self):
            comp = Composition()
            A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
            comp.add_mechanism(A)

            sched = Scheduler(composition=comp)
            sched.add_condition(A, Always())

            termination_conds = {}
            termination_conds[TimeScale.RUN] = AtTrial(4)
            termination_conds[TimeScale.TRIAL] = AtPass(1)
            comp.run(
                inputs={A: range(6)},
                scheduler_processing=sched,
                termination_processing=termination_conds
            )
            output = sched.execution_list[comp._execution_id]

            expected_output = [A, A, A, A]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_AfterTrial(self):
            comp = Composition()
            A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
            comp.add_mechanism(A)

            sched = Scheduler(composition=comp)
            sched.add_condition(A, Always())

            termination_conds = {}
            termination_conds[TimeScale.RUN] = AfterTrial(4)
            termination_conds[TimeScale.TRIAL] = AtPass(1)
            comp.run(
                inputs={A: range(6)},
                scheduler_processing=sched,
                termination_processing=termination_conds
            )
            output = sched.execution_list[comp._execution_id]

            expected_output = [A, A, A, A, A]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_AfterNTrials(self):
            comp = Composition()
            A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
            comp.add_mechanism(A)

            sched = Scheduler(composition=comp)
            sched.add_condition(A, AfterNPasses(1))

            termination_conds = {}
            termination_conds[TimeScale.RUN] = AfterNTrials(1)
            termination_conds[TimeScale.TRIAL] = AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [set(), A, A, A, A]
            assert output == pytest.helpers.setify_expected_output(expected_output)

    class TestComponentBased:

        def test_BeforeNCalls(self):
            comp = Composition()
            A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
            comp.add_mechanism(A)

            sched = Scheduler(composition=comp)
            sched.add_condition(A, BeforeNCalls(A, 3))

            termination_conds = {}
            termination_conds[TimeScale.RUN] = AfterNTrials(1)
            termination_conds[TimeScale.TRIAL] = AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [A, A, A, set(), set()]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        # NOTE:
        # The behavior is not desired (i.e. depending on the order mechanisms are checked, B running AtCall(A, x))
        # may run on both the xth and x+1st call of A; if A and B are not parent-child
        # A fix could invalidate key assumptions and affect many other conditions
        # Since this condition is unlikely to be used, it's best to leave it for now
        # def test_AtCall(self):
        #     comp = Composition()
        #     A = TransferMechanism(function = Linear(slope=5.0, intercept = 2.0), name = 'A')
        #     B = TransferMechanism(function = Linear(intercept = 4.0), name = 'B')
        #     C = TransferMechanism(function = Linear(intercept = 1.5), name = 'C')
        #     for m in [A,B]:
        #         comp.add_mechanism(m)

        #     sched = Scheduler(composition=comp)
        #     sched.add_condition(A, Always())
        #     sched.add_condition(B, AtCall(A, 3))

        #     termination_conds = {}
        #     termination_conds[TimeScale.RUN] = AfterNTrials(1)
        #     termination_conds[TimeScale.TRIAL] = AtPass(5)
        #     output = list(sched.run(termination_conds=termination_conds))

        #     expected_output = [A, A, set([A, B]), A, A]
        #     assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_AfterCall(self):
            comp = Composition()
            A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
            B = TransferMechanism(function=Linear(intercept=4.0), name='B')
            for m in [A, B]:
                comp.add_mechanism(m)

            sched = Scheduler(composition=comp)
            sched.add_condition(B, AfterCall(A, 3))

            termination_conds = {}
            termination_conds[TimeScale.RUN] = AfterNTrials(1)
            termination_conds[TimeScale.TRIAL] = AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [A, A, A, set([A, B]), set([A, B])]
            assert output == pytest.helpers.setify_expected_output(expected_output)

        def test_AfterNCalls(self):
            comp = Composition()
            A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
            B = TransferMechanism(function=Linear(intercept=4.0), name='B')
            for m in [A, B]:
                comp.add_mechanism(m)

            sched = Scheduler(composition=comp)
            sched.add_condition(A, Always())
            sched.add_condition(B, AfterNCalls(A, 3))

            termination_conds = {}
            termination_conds[TimeScale.RUN] = AfterNTrials(1)
            termination_conds[TimeScale.TRIAL] = AtPass(5)
            output = list(sched.run(termination_conds=termination_conds))

            expected_output = [A, A, set([A, B]), set([A, B]), set([A, B])]
            assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_composite_condition_multi(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='C')
        for m in [A, B, C]:
            comp.add_mechanism(m)
        comp.add_projection(A, MappingProjection(), B)
        comp.add_projection(B, MappingProjection(), C)
        sched = Scheduler(composition=comp)

        sched.add_condition(A, EveryNPasses(1))
        sched.add_condition(B, EveryNCalls(A, 2))
        sched.add_condition(C, All(
            Any(
                AfterPass(6),
                AfterNCalls(B, 2)
            ),
            Any(
                AfterPass(2),
                AfterNCalls(B, 3)
            )
        )
        )

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = AfterNCalls(C, 3)
        output = list(sched.run(termination_conds=termination_conds))
        expected_output = [
            A, A, B, A, A, B, C, A, C, A, B, C
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_WhenFinishedAny_1(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
        A.is_finished = True
        B = TransferMechanism(function=Linear(intercept=4.0), name='B')
        B.is_finished = True
        C = TransferMechanism(function=Linear(intercept=1.5), name='C')
        for m in [A, B, C]:
            comp.add_mechanism(m)
        comp.add_projection(A, MappingProjection(), C)
        comp.add_projection(B, MappingProjection(), C)
        sched = Scheduler(composition=comp)

        sched.add_condition(A, EveryNPasses(1))
        sched.add_condition(B, EveryNPasses(1))
        sched.add_condition(C, WhenFinishedAny(A, B))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = AfterNCalls(C, 1)
        output = list(sched.run(termination_conds=termination_conds))
        expected_output = [
            set([A, B]), C
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_WhenFinishedAny_2(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
        A.is_finished = False
        B = TransferMechanism(function=Linear(intercept=4.0), name='B')
        B.is_finished = True
        C = TransferMechanism(function=Linear(intercept=1.5), name='C')
        for m in [A, B, C]:
            comp.add_mechanism(m)
        comp.add_projection(A, MappingProjection(), C)
        comp.add_projection(B, MappingProjection(), C)
        sched = Scheduler(composition=comp)

        sched.add_condition(A, EveryNPasses(1))
        sched.add_condition(B, EveryNPasses(1))
        sched.add_condition(C, WhenFinishedAny(A, B))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = AfterNCalls(A, 5)
        output = list(sched.run(termination_conds=termination_conds))
        expected_output = [
            set([A, B]), C, set([A, B]), C, set([A, B]), C, set([A, B]), C, set([A, B])
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_WhenFinishedAny_noargs(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='C')
        for m in [A, B, C]:
            comp.add_mechanism(m)
        comp.add_projection(A, MappingProjection(), C)
        comp.add_projection(B, MappingProjection(), C)
        sched = Scheduler(composition=comp)

        sched.add_condition(A, Always())
        sched.add_condition(B, Always())
        sched.add_condition(C, Always())

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = WhenFinishedAny()
        output = []
        i = 0
        for step in sched.run(termination_conds=termination_conds):
            if i == 3:
                A.is_finished = True
                B.is_finished = True
            if i == 4:
                C.is_finished = True
            output.append(step)
            i += 1
        expected_output = [
            set([A, B]), C, set([A, B]), C,
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_WhenFinishedAll_1(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
        A.is_finished = True
        B = TransferMechanism(function=Linear(intercept=4.0), name='B')
        B.is_finished = True
        C = TransferMechanism(function=Linear(intercept=1.5), name='C')
        for m in [A, B, C]:
            comp.add_mechanism(m)
        comp.add_projection(A, MappingProjection(), C)
        comp.add_projection(B, MappingProjection(), C)
        sched = Scheduler(composition=comp)

        sched.add_condition(A, EveryNPasses(1))
        sched.add_condition(B, EveryNPasses(1))
        sched.add_condition(C, WhenFinishedAll(A, B))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = AfterNCalls(C, 1)
        output = list(sched.run(termination_conds=termination_conds))
        expected_output = [
            set([A, B]), C
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_WhenFinishedAll_2(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
        A.is_finished = False
        B = TransferMechanism(function=Linear(intercept=4.0), name='B')
        B.is_finished = True
        C = TransferMechanism(function=Linear(intercept=1.5), name='C')
        for m in [A, B, C]:
            comp.add_mechanism(m)
        comp.add_projection(A, MappingProjection(), C)
        comp.add_projection(B, MappingProjection(), C)
        sched = Scheduler(composition=comp)

        sched.add_condition(A, EveryNPasses(1))
        sched.add_condition(B, EveryNPasses(1))
        sched.add_condition(C, WhenFinishedAll(A, B))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = AfterNCalls(A, 5)
        output = list(sched.run(termination_conds=termination_conds))
        expected_output = [
            set([A, B]), set([A, B]), set([A, B]), set([A, B]), set([A, B]),
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_WhenFinishedAll_noargs(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='C')
        for m in [A, B, C]:
            comp.add_mechanism(m)
        comp.add_projection(A, MappingProjection(), C)
        comp.add_projection(B, MappingProjection(), C)
        sched = Scheduler(composition=comp)

        sched.add_condition(A, Always())
        sched.add_condition(B, Always())
        sched.add_condition(C, Always())

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = WhenFinishedAll()
        output = []
        i = 0
        for step in sched.run(termination_conds=termination_conds):
            if i == 3:
                A.is_finished = True
                B.is_finished = True
            if i == 4:
                C.is_finished = True
            output.append(step)
            i += 1
        expected_output = [
            set([A, B]), C, set([A, B]), C, set([A, B]),
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_AfterNCallsCombined(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='C')
        for m in [A, B, C]:
            comp.add_mechanism(m)
        comp.add_projection(A, MappingProjection(), B)
        comp.add_projection(B, MappingProjection(), C)
        sched = Scheduler(composition=comp)

        sched.add_condition(A, EveryNPasses(1))
        sched.add_condition(B, EveryNCalls(A, 2))
        sched.add_condition(C, EveryNCalls(B, 2))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = AfterNCallsCombined(B, C, n=4)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            A, A, B, A, A, B, C, A, A, B
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_AllHaveRun(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='C')
        for m in [A, B, C]:
            comp.add_mechanism(m)
        comp.add_projection(A, MappingProjection(), B)
        comp.add_projection(B, MappingProjection(), C)
        sched = Scheduler(composition=comp)

        sched.add_condition(A, EveryNPasses(1))
        sched.add_condition(B, EveryNCalls(A, 2))
        sched.add_condition(C, EveryNCalls(B, 2))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = AllHaveRun(A, B, C)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            A, A, B, A, A, B, C
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_AllHaveRun_2(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='C')
        for m in [A, B, C]:
            comp.add_mechanism(m)
        comp.add_projection(A, MappingProjection(), B)
        comp.add_projection(B, MappingProjection(), C)
        sched = Scheduler(composition=comp)

        sched.add_condition(A, EveryNPasses(1))
        sched.add_condition(B, EveryNCalls(A, 2))
        sched.add_condition(C, EveryNCalls(B, 2))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = AllHaveRun()
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            A, A, B, A, A, B, C
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)
