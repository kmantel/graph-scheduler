import fractions
import logging
import networkx as nx
import numpy as np
import psyneulink as pnl
import pytest
import types

import graph_scheduler as gs


logger = logging.getLogger(__name__)


class TestScheduler:
    stroop_paths = [
        ['Color_Input', 'Color_Hidden', 'Output', 'Decision'],
        ['Word_Input', 'Word_Hidden', 'Output', 'Decision'],
        ['Reward']
    ]

    stroop_consideration_queue = [
        {'Color_Input', 'Word_Input', 'Reward'},
        {'Color_Hidden', 'Word_Hidden'},
        {'Output'},
        {'Decision'}
    ]

    @pytest.mark.parametrize(
        'graph, expected_consideration_queue',
        [
            (
                pytest.helpers.create_graph_from_pathways(*stroop_paths),
                stroop_consideration_queue
            ),
            (
                nx.DiGraph(pytest.helpers.create_graph_from_pathways(*stroop_paths)).reverse(),
                stroop_consideration_queue
            )
        ]
    )
    def test_construction(self, graph, expected_consideration_queue):
        sched = gs.Scheduler(graph)
        assert sched.consideration_queue == expected_consideration_queue

    def test_copy(self):
        pass

    def test_deepcopy(self):
        pass

    def test_create_multiple_contexts(self):
        graph = {'A': set()}
        scheduler = gs.Scheduler(graph)

        scheduler.get_clock(scheduler.default_execution_id)._increment_time(pnl.TimeScale.ENVIRONMENT_STATE_UPDATE)

        eid = 'eid'
        eid1 = 'eid1'
        scheduler._init_counts(execution_id=eid)

        assert scheduler.clocks[eid].time.environment_state_update == 0

        scheduler.get_clock(scheduler.default_execution_id)._increment_time(pnl.TimeScale.ENVIRONMENT_STATE_UPDATE)

        assert scheduler.clocks[eid].time.environment_state_update == 0

        scheduler._init_counts(execution_id=eid1, base_execution_id=scheduler.default_execution_id)

        assert scheduler.clocks[eid1].time.environment_state_update == 2

    @pytest.mark.psyneulink
    def test_two_compositions_one_scheduler(self):
        comp1 = pnl.Composition()
        comp2 = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        comp1.add_node(A)
        comp2.add_node(A)

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp1))

        sched.add_condition(A, pnl.BeforeNCalls(A, 5, time_scale=pnl.TimeScale.LIFE))

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(6)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AfterNPasses(1)
        comp1.run(
            inputs={A: [[0], [1], [2], [3], [4], [5]]},
            scheduler=sched,
            termination_processing=termination_conds
        )
        output = sched.execution_list[comp1.default_execution_id]

        expected_output = [
            A, A, A, A, A, set()
        ]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

        comp2.run(
            inputs={A: [[0], [1], [2], [3], [4], [5]]},
            scheduler=sched,
            termination_processing=termination_conds
        )
        output = sched.execution_list[comp2.default_execution_id]

        expected_output = [
            A, A, A, A, A, set()
        ]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

    @pytest.mark.psyneulink
    def test_one_composition_two_contexts(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        comp.add_node(A)

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(A, pnl.BeforeNCalls(A, 5, time_scale=pnl.TimeScale.LIFE))

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(6)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AfterNPasses(1)
        eid = 'eid'
        comp.run(
            inputs={A: [[0], [1], [2], [3], [4], [5]]},
            scheduler=sched,
            termination_processing=termination_conds,
            context=eid,
        )
        output = sched.execution_list[eid]

        expected_output = [
            A, A, A, A, A, set()
        ]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

        comp.run(
            inputs={A: [[0], [1], [2], [3], [4], [5]]},
            scheduler=sched,
            termination_processing=termination_conds,
            context=eid,
        )
        output = sched.execution_list[eid]

        expected_output = [
            A, A, A, A, A, set(), set(), set(), set(), set(), set(), set()
        ]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

        eid = 'eid1'
        comp.run(
            inputs={A: [[0], [1], [2], [3], [4], [5]]},
            scheduler=sched,
            termination_processing=termination_conds,
            context=eid,
        )
        output = sched.execution_list[eid]

        expected_output = [
            A, A, A, A, A, set()
        ]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

    @pytest.mark.psyneulink
    def test_change_termination_condition(self):
        D = pnl.DDM(
            function=pnl.DriftDiffusionIntegrator(
                threshold=10, time_step_size=1.0
            ),
            execute_until_finished=False,
            reset_stateful_function_when=pnl.Never()
        )
        C = pnl.Composition(pathways=[D])

        D.set_log_conditions(pnl.VALUE)

        def change_termination_processing():
            if C.termination_processing is None:
                C.scheduler.termination_conds = {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.WhenFinished(D)}
                C.termination_processing = {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.WhenFinished(D)}
            elif isinstance(C.termination_processing[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE], pnl.AllHaveRun):
                C.scheduler.termination_conds = {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.WhenFinished(D)}
                C.termination_processing = {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.WhenFinished(D)}
            else:
                C.scheduler.termination_conds = {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.AllHaveRun()}
                C.termination_processing = {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.AllHaveRun()}

        change_termination_processing()
        C.run(inputs={D: [[1.0], [2.0]]},
              # termination_processing={pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.WhenFinished(D)},
              call_after_trial=change_termination_processing,
              reset_stateful_functions_when=pnl.AtConsiderationSetExecution(0),
              num_trials=4)
        # EnvironmentStateUpdate 0:
        # input = 1.0, termination condition = pnl.WhenFinished
        # 10 passes (value = 1.0, 2.0 ... 9.0, 10.0)
        # EnvironmentStateUpdate 1:
        # input = 2.0, termination condition = pnl.AllHaveRun
        # 1 pass (value = 2.0)
        expected_results = [[np.array([10.]), np.array([10.])],
                            [np.array([2.]), np.array([1.])],
                            [np.array([10.]), np.array([10.])],
                            [np.array([2.]), np.array([1.])]]
        assert np.allclose(expected_results, np.asfarray(C.results))

    @pytest.mark.psyneulink
    def test_default_condition_1(self):
        A = pnl.TransferMechanism(name='A')
        B = pnl.TransferMechanism(name='B')
        C = pnl.TransferMechanism(name='C')

        comp = pnl.Composition(pathways=[[A, C], [A, B, C]])
        comp.scheduler.add_condition(A, pnl.AtPass(1))
        comp.scheduler.add_condition(B, pnl.Always())

        output = list(comp.scheduler.run())
        expected_output = [B, A, B, C]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    @pytest.mark.psyneulink
    def test_default_condition_2(self):
        A = pnl.TransferMechanism(name='A')
        B = pnl.TransferMechanism(name='B')
        C = pnl.TransferMechanism(name='C')

        comp = pnl.Composition(pathways=[[A, B], [C]])
        comp.scheduler.add_condition(C, pnl.AtPass(1))

        output = list(comp.scheduler.run())
        expected_output = [A, B, {C, A}]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_exact_time_mode(self):
        sched = gs.Scheduler(
            {'A': set(), 'B': {'A'}},
            mode=gs.SchedulingMode.EXACT_TIME
        )

        # these cannot run at same execution set unless in EXACT_TIME
        sched.add_condition('A', pnl.TimeInterval(start=1))
        sched.add_condition('B', pnl.TimeInterval(start=1))

        list(sched.run())

        assert sched.mode == gs.SchedulingMode.EXACT_TIME
        assert sched.execution_list[sched.default_execution_id] == [{'A', 'B'}]
        assert sched.execution_timestamps[sched.default_execution_id][0].absolute == 1 * gs._unit_registry.ms

    def test_run_with_new_execution_id(self):
        sched = gs.Scheduler({'A': set()})
        sched.add_condition('A', gs.AtPass(1))

        output = list(sched.run(execution_id='eid'))

        assert output == [set(), {'A'}]
        assert 'eid' in sched.execution_list
        assert sched.execution_list['eid'] == output

        assert sched.get_clock('eid') == sched.get_clock(types.SimpleNamespace(default_execution_id='eid'))

    def test_delete_counts(self):
        sched = gs.Scheduler(
            {
                'A': set(),
                'B': {'A'},
                'C': {'A'},
                'D': {'C', 'B'}
            }
        )

        sched.add_condition_set(
            {
                'A': gs.EveryNPasses(2),
                'B': gs.EveryNCalls('A', 2),
                'C': gs.EveryNCalls('A', 3),
                'D': gs.AfterNCallsCombined('B', 'C', n=6)
            }
        )

        eid_delete = 'eid'
        eid_repeat = 'eid2'

        del_run_1 = list(sched.run(execution_id=eid_delete))
        repeat_run_1 = list(sched.run(execution_id=eid_repeat))

        sched._delete_counts(eid_delete)

        del_run_2 = list(sched.run(execution_id=eid_delete))
        repeat_run_2 = list(sched.run(execution_id=eid_repeat))

        assert del_run_1 == del_run_2
        assert repeat_run_1 == repeat_run_2

        assert sched.execution_list[eid_delete] == del_run_1
        assert sched.execution_list[eid_repeat] == repeat_run_2 + repeat_run_2


@pytest.mark.psyneulink
class TestLinear:

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

    def test_no_termination_conds(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='scheduler-pytests-B')
        C = pnl.TransferMechanism(function=pnl.Linear(intercept=1.5), name='scheduler-pytests-C')
        for m in [A, B, C]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, B)
        comp.add_projection(pnl.MappingProjection(), B, C)

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(A, pnl.EveryNPasses(1))
        sched.add_condition(B, pnl.EveryNCalls(A, 2))
        sched.add_condition(C, pnl.EveryNCalls(B, 3))

        output = list(sched.run())

        expected_output = [
            A, A, B, A, A, B, A, A, B, C,
        ]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

    # tests below are copied from old scheduler, need renaming
    def test_1(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='scheduler-pytests-B')
        C = pnl.TransferMechanism(function=pnl.Linear(intercept=1.5), name='scheduler-pytests-C')
        for m in [A, B, C]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, B)
        comp.add_projection(pnl.MappingProjection(), B, C)

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(A, pnl.EveryNPasses(1))
        sched.add_condition(B, pnl.EveryNCalls(A, 2))
        sched.add_condition(C, pnl.EveryNCalls(B, 3))

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AfterNCalls(C, 4, time_scale=pnl.TimeScale.ENVIRONMENT_STATE_UPDATE)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            A, A, B, A, A, B, A, A, B, C,
            A, A, B, A, A, B, A, A, B, C,
            A, A, B, A, A, B, A, A, B, C,
            A, A, B, A, A, B, A, A, B, C,
        ]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_1b(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='scheduler-pytests-B')
        C = pnl.TransferMechanism(function=pnl.Linear(intercept=1.5), name='scheduler-pytests-C')
        for m in [A, B, C]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, B)
        comp.add_projection(pnl.MappingProjection(), B, C)

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(A, pnl.EveryNPasses(1))
        sched.add_condition(B, pnl.Any(pnl.EveryNCalls(A, 2), pnl.AfterPass(1)))
        sched.add_condition(C, pnl.EveryNCalls(B, 3))

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AfterNCalls(C, 4, time_scale=pnl.TimeScale.ENVIRONMENT_STATE_UPDATE)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            A, A, B, A, B, A, B, C,
            A, B, A, B, A, B, C,
            A, B, A, B, A, B, C,
            A, B, A, B, A, B, C,
        ]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_2(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='scheduler-pytests-B')
        C = pnl.TransferMechanism(function=pnl.Linear(intercept=1.5), name='scheduler-pytests-C')
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
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AfterNCalls(C, 1, time_scale=pnl.TimeScale.ENVIRONMENT_STATE_UPDATE)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A, A, B, A, A, B, C]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_3(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='scheduler-pytests-B')
        C = pnl.TransferMechanism(function=pnl.Linear(intercept=1.5), name='scheduler-pytests-C')
        for m in [A, B, C]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, B)
        comp.add_projection(pnl.MappingProjection(), B, C)

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(A, pnl.EveryNPasses(1))
        sched.add_condition(B, pnl.EveryNCalls(A, 2))
        sched.add_condition(C, pnl.All(pnl.AfterNCalls(B, 2), pnl.EveryNCalls(B, 1)))

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AfterNCalls(C, 4, time_scale=pnl.TimeScale.ENVIRONMENT_STATE_UPDATE)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            A, A, B, A, A, B, C, A, A, B, C, A, A, B, C, A, A, B, C
        ]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_6(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='scheduler-pytests-B')
        C = pnl.TransferMechanism(function=pnl.Linear(intercept=1.5), name='scheduler-pytests-C')
        for m in [A, B, C]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, B)
        comp.add_projection(pnl.MappingProjection(), B, C)

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(A, pnl.BeforePass(5))
        sched.add_condition(B, pnl.AfterNCalls(A, 5))
        sched.add_condition(C, pnl.AfterNCalls(B, 1))

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AfterNCalls(C, 3)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            A, A, A, A, A, B, C, B, C, B, C
        ]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_6_two_environment_state_updates(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='scheduler-pytests-B')
        C = pnl.TransferMechanism(function=pnl.Linear(intercept=1.5), name='scheduler-pytests-C')
        for m in [A, B, C]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, B)
        comp.add_projection(pnl.MappingProjection(), B, C)

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(A, pnl.BeforePass(5))
        sched.add_condition(B, pnl.AfterNCalls(A, 5))
        sched.add_condition(C, pnl.AfterNCalls(B, 1))

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(2)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AfterNCalls(C, 3)
        comp.run(
            inputs={A: [[0], [1], [2], [3], [4], [5]]},
            scheduler=sched,
            termination_processing=termination_conds
        )
        output = sched.execution_list[comp.default_execution_id]

        expected_output = [
            A, A, A, A, A, B, C, B, C, B, C,
            A, A, A, A, A, B, C, B, C, B, C
        ]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_7(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='scheduler-pytests-B')
        for m in [A, B]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, B)

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(A, pnl.EveryNPasses(1))
        sched.add_condition(B, pnl.EveryNCalls(A, 2))

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.Any(pnl.AfterNCalls(A, 1), pnl.AfterNCalls(B, 1))
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_8(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='scheduler-pytests-B')
        for m in [A, B]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, B)

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(A, pnl.EveryNPasses(1))
        sched.add_condition(B, pnl.EveryNCalls(A, 2))

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.All(pnl.AfterNCalls(A, 1), pnl.AfterNCalls(B, 1))
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A, A, B]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_9(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='scheduler-pytests-B')
        for m in [A, B]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, B)

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(A, pnl.EveryNPasses(1))
        sched.add_condition(B, pnl.WhenFinished(A))

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AfterNCalls(B, 2)

        output = []
        i = 0
        A.is_finished_flag = False
        for step in sched.run(termination_conds=termination_conds):
            if i == 3:
                A.is_finished_flag = True
            output.append(step)
            i += 1

        expected_output = [A, A, A, A, B, A, B]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_9b(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        A.is_finished_flag = False
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='scheduler-pytests-B')
        for m in [A, B]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, B)

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(A, pnl.EveryNPasses(1))
        sched.add_condition(B, pnl.WhenFinished(A))

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AtPass(5)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A, A, A, A, A]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_10(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        A.is_finished_flag = True
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='scheduler-pytests-B')

        for m in [A, B]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, B)

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(A, pnl.EveryNPasses(1))
        sched.add_condition(B, pnl.Any(pnl.WhenFinished(A), pnl.AfterNCalls(A, 3)))

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AfterNCalls(B, 5)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A, B, A, B, A, B, A, B, A, B]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_10b(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        A.is_finished_flag = False
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='scheduler-pytests-B')

        for m in [A, B]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, B)

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(A, pnl.EveryNPasses(1))
        sched.add_condition(B, pnl.Any(pnl.WhenFinished(A), pnl.AfterNCalls(A, 3)))

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AfterNCalls(B, 4)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A, A, A, B, A, B, A, B, A, B]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_10c(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        A.is_finished_flag = True
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='scheduler-pytests-B')

        for m in [A, B]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, B)

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(A, pnl.EveryNPasses(1))
        sched.add_condition(B, pnl.All(pnl.WhenFinished(A), pnl.AfterNCalls(A, 3)))

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AfterNCalls(B, 4)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A, A, A, B, A, B, A, B, A, B]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_10d(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        A.is_finished_flag = False
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='scheduler-pytests-B')

        for m in [A, B]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, B)

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(A, pnl.EveryNPasses(1))
        sched.add_condition(B, pnl.All(pnl.WhenFinished(A), pnl.AfterNCalls(A, 3)))

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AtPass(10)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A, A, A, A, A, A, A, A, A, A]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    ########################################
    # tests with linear compositions
    ########################################
    def test_linear_AAB(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='scheduler-pytests-B')
        for m in [A, B]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, B)

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(A, pnl.EveryNPasses(1))
        sched.add_condition(B, pnl.EveryNCalls(A, 2))

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNCalls(B, 2, time_scale=pnl.TimeScale.ENVIRONMENT_SEQUENCE)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AfterNCalls(B, 2, time_scale=pnl.TimeScale.ENVIRONMENT_STATE_UPDATE)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A, A, B, A, A, B]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_linear_ABB(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='scheduler-pytests-B')
        for m in [A, B]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, B)

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(A, pnl.Any(pnl.AtPass(0), pnl.EveryNCalls(B, 2)))
        sched.add_condition(B, pnl.Any(pnl.EveryNCalls(A, 1), pnl.EveryNCalls(B, 1)))

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AfterNCalls(B, 8, time_scale=pnl.TimeScale.ENVIRONMENT_STATE_UPDATE)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A, B, B, A, B, B, A, B, B, A, B, B]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_linear_ABBCC(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='scheduler-pytests-B')
        C = pnl.TransferMechanism(function=pnl.Linear(intercept=1.5), name='scheduler-pytests-C')
        for m in [A, B, C]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, B)
        comp.add_projection(pnl.MappingProjection(), B, C)

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(A, pnl.Any(pnl.AtPass(0), pnl.EveryNCalls(C, 2)))
        sched.add_condition(B, pnl.Any(pnl.JustRan(A), pnl.JustRan(B)))
        sched.add_condition(C, pnl.Any(pnl.EveryNCalls(B, 2), pnl.JustRan(C)))

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AfterNCalls(C, 4, time_scale=pnl.TimeScale.ENVIRONMENT_STATE_UPDATE)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A, B, B, C, C, A, B, B, C, C]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_linear_ABCBC(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='scheduler-pytests-B')
        C = pnl.TransferMechanism(function=pnl.Linear(intercept=1.5), name='scheduler-pytests-C')
        for m in [A, B, C]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, B)
        comp.add_projection(pnl.MappingProjection(), B, C)

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(A, pnl.Any(pnl.AtPass(0), pnl.EveryNCalls(C, 2)))
        sched.add_condition(B, pnl.Any(pnl.EveryNCalls(A, 1), pnl.EveryNCalls(C, 1)))
        sched.add_condition(C, pnl.EveryNCalls(B, 1))

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AfterNCalls(C, 4, time_scale=pnl.TimeScale.ENVIRONMENT_STATE_UPDATE)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A, B, C, B, C, A, B, C, B, C]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    ########################################
    # tests with small branching compositions
    ########################################


@pytest.mark.psyneulink
class TestBranching:
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

    #   triangle:         A
    #                    / \
    #                   B   C

    def test_triangle_1(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='scheduler-pytests-B')
        C = pnl.TransferMechanism(function=pnl.Linear(intercept=1.5), name='scheduler-pytests-C')
        for m in [A, B, C]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, B)
        comp.add_projection(pnl.MappingProjection(), A, C)

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(A, pnl.EveryNPasses(1))
        sched.add_condition(B, pnl.EveryNCalls(A, 1))
        sched.add_condition(C, pnl.EveryNCalls(A, 1))

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AfterNCalls(C, 3, time_scale=pnl.TimeScale.ENVIRONMENT_STATE_UPDATE)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            A, set([B, C]),
            A, set([B, C]),
            A, set([B, C]),
        ]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_triangle_2(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='scheduler-pytests-B')
        C = pnl.TransferMechanism(function=pnl.Linear(intercept=1.5), name='scheduler-pytests-C')
        for m in [A, B, C]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, B)
        comp.add_projection(pnl.MappingProjection(), A, C)

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(A, pnl.EveryNPasses(1))
        sched.add_condition(B, pnl.EveryNCalls(A, 1))
        sched.add_condition(C, pnl.EveryNCalls(A, 2))

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AfterNCalls(C, 3, time_scale=pnl.TimeScale.ENVIRONMENT_STATE_UPDATE)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            A, B,
            A, set([B, C]),
            A, B,
            A, set([B, C]),
            A, B,
            A, set([B, C]),
        ]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_triangle_3(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='scheduler-pytests-B')
        C = pnl.TransferMechanism(function=pnl.Linear(intercept=1.5), name='scheduler-pytests-C')
        for m in [A, B, C]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, B)
        comp.add_projection(pnl.MappingProjection(), A, C)

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(A, pnl.EveryNPasses(1))
        sched.add_condition(B, pnl.EveryNCalls(A, 2))
        sched.add_condition(C, pnl.EveryNCalls(A, 3))

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AfterNCalls(C, 2, time_scale=pnl.TimeScale.ENVIRONMENT_STATE_UPDATE)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            A, A, B, A, C, A, B, A, A, set([B, C])
        ]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

    # this is test 11 of original constraint_scheduler.py
    def test_triangle_4(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='scheduler-pytests-B')
        C = pnl.TransferMechanism(function=pnl.Linear(intercept=1.5), name='scheduler-pytests-C')

        for m in [A, B, C]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, B)
        comp.add_projection(pnl.MappingProjection(), A, C)

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(A, pnl.EveryNPasses(1))
        sched.add_condition(B, pnl.EveryNCalls(A, 2))
        sched.add_condition(C, pnl.All(pnl.WhenFinished(A), pnl.AfterNCalls(B, 3)))

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AfterNCalls(C, 1)
        output = []
        i = 0
        A.is_finished_flag = False
        for step in sched.run(termination_conds=termination_conds):
            if i == 3:
                A.is_finished_flag = True
            output.append(step)
            i += 1

        expected_output = [A, A, B, A, A, B, A, A, set([B, C])]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_triangle_4b(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='scheduler-pytests-B')
        C = pnl.TransferMechanism(function=pnl.Linear(intercept=1.5), name='scheduler-pytests-C')

        for m in [A, B, C]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, B)
        comp.add_projection(pnl.MappingProjection(), A, C)

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(A, pnl.EveryNPasses(1))
        sched.add_condition(B, pnl.EveryNCalls(A, 2))
        sched.add_condition(C, pnl.All(pnl.WhenFinished(A), pnl.AfterNCalls(B, 3)))

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AfterNCalls(C, 1)
        output = []
        i = 0
        A.is_finished_flag = False
        for step in sched.run(termination_conds=termination_conds):
            if i == 10:
                A.is_finished_flag = True
            output.append(step)
            i += 1

        expected_output = [A, A, B, A, A, B, A, A, B, A, A, set([B, C])]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

    #   inverted triangle:           A   B
    #                                 \ /
    #                                  C

    # this is test 4 of original constraint_scheduler.py
    # this test has an implicit priority set of A<B !
    def test_invtriangle_1(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='scheduler-pytests-B')
        C = pnl.TransferMechanism(function=pnl.Linear(intercept=1.5), name='scheduler-pytests-C')
        for m in [A, B, C]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, C)
        comp.add_projection(pnl.MappingProjection(), B, C)

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(A, pnl.EveryNPasses(1))
        sched.add_condition(B, pnl.EveryNCalls(A, 2))
        sched.add_condition(C, pnl.Any(pnl.AfterNCalls(A, 3), pnl.AfterNCalls(B, 3)))

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AfterNCalls(C, 4, time_scale=pnl.TimeScale.ENVIRONMENT_STATE_UPDATE)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            A, set([A, B]), A, C, set([A, B]), C, A, C, set([A, B]), C
        ]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

    # this is test 5 of original constraint_scheduler.py
    # this test has an implicit priority set of A<B !
    def test_invtriangle_2(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='scheduler-pytests-B')
        C = pnl.TransferMechanism(function=pnl.Linear(intercept=1.5), name='scheduler-pytests-C')
        for m in [A, B, C]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, C)
        comp.add_projection(pnl.MappingProjection(), B, C)

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(A, pnl.EveryNPasses(1))
        sched.add_condition(B, pnl.EveryNCalls(A, 2))
        sched.add_condition(C, pnl.All(pnl.AfterNCalls(A, 3), pnl.AfterNCalls(B, 3)))

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AfterNCalls(C, 2, time_scale=pnl.TimeScale.ENVIRONMENT_STATE_UPDATE)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            A, set([A, B]), A, set([A, B]), A, set([A, B]), C, A, C
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    #   checkmark:                   A
    #                                 \
    #                                  B   C
    #                                   \ /
    #                                    D

    # testing toposort
    def test_checkmark_1(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='scheduler-pytests-B')
        C = pnl.TransferMechanism(function=pnl.Linear(intercept=1.5), name='scheduler-pytests-C')
        D = pnl.TransferMechanism(function=pnl.Linear(intercept=.5), name='scheduler-pytests-D')
        for m in [A, B, C, D]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, B)
        comp.add_projection(pnl.MappingProjection(), B, D)
        comp.add_projection(pnl.MappingProjection(), C, D)

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(A, pnl.Always())
        sched.add_condition(B, pnl.Always())
        sched.add_condition(C, pnl.Always())
        sched.add_condition(D, pnl.Always())

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AfterNCalls(D, 1, time_scale=pnl.TimeScale.ENVIRONMENT_STATE_UPDATE)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            set([A, C]), B, D
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_checkmark_2(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='scheduler-pytests-B')
        C = pnl.TransferMechanism(function=pnl.Linear(intercept=1.5), name='scheduler-pytests-C')
        D = pnl.TransferMechanism(function=pnl.Linear(intercept=.5), name='scheduler-pytests-D')
        for m in [A, B, C, D]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, B)
        comp.add_projection(pnl.MappingProjection(), B, D)
        comp.add_projection(pnl.MappingProjection(), C, D)

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(A, pnl.EveryNPasses(1))
        sched.add_condition(B, pnl.EveryNCalls(A, 2))
        sched.add_condition(C, pnl.EveryNCalls(A, 2))
        sched.add_condition(D, pnl.All(pnl.EveryNCalls(B, 2), pnl.EveryNCalls(C, 2)))

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AfterNCalls(D, 1, time_scale=pnl.TimeScale.ENVIRONMENT_STATE_UPDATE)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            A, set([A, C]), B, A, set([A, C]), B, D
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_checkmark2_1(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='scheduler-pytests-B')
        C = pnl.TransferMechanism(function=pnl.Linear(intercept=1.5), name='scheduler-pytests-C')
        D = pnl.TransferMechanism(function=pnl.Linear(intercept=.5), name='scheduler-pytests-D')
        for m in [A, B, C, D]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, B)
        comp.add_projection(pnl.MappingProjection(), A, D)
        comp.add_projection(pnl.MappingProjection(), B, D)
        comp.add_projection(pnl.MappingProjection(), C, D)

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(A, pnl.EveryNPasses(1))
        sched.add_condition(B, pnl.EveryNCalls(A, 2))
        sched.add_condition(C, pnl.EveryNCalls(A, 2))
        sched.add_condition(D, pnl.All(pnl.EveryNCalls(B, 2), pnl.EveryNCalls(C, 2)))

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AfterNCalls(D, 1, time_scale=pnl.TimeScale.ENVIRONMENT_STATE_UPDATE)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            A, set([A, C]), B, A, set([A, C]), B, D
        ]

        assert output == pytest.helpers.setify_expected_output(expected_output)

    #   multi source:                   A1    A2
    #                                   / \  / \
    #                                  B1  B2  B3
    #                                   \ /  \ /
    #                                    C1   C2
    def test_multisource_1(self):
        comp = pnl.Composition()
        A1 = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A1')
        A2 = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A2')
        B1 = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='B1')
        B2 = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='B2')
        B3 = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='B3')
        C1 = pnl.TransferMechanism(function=pnl.Linear(intercept=1.5), name='C1')
        C2 = pnl.TransferMechanism(function=pnl.Linear(intercept=.5), name='C2')
        for m in [A1, A2, B1, B2, B3, C1, C2]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A1, B1)
        comp.add_projection(pnl.MappingProjection(), A1, B2)
        comp.add_projection(pnl.MappingProjection(), A2, B1)
        comp.add_projection(pnl.MappingProjection(), A2, B2)
        comp.add_projection(pnl.MappingProjection(), A2, B3)
        comp.add_projection(pnl.MappingProjection(), B1, C1)
        comp.add_projection(pnl.MappingProjection(), B2, C1)
        comp.add_projection(pnl.MappingProjection(), B1, C2)
        comp.add_projection(pnl.MappingProjection(), B3, C2)

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        for m in comp.nodes:
            sched.add_condition(m, pnl.Always())

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.All(pnl.AfterNCalls(C1, 1), pnl.AfterNCalls(C2, 1))
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            set([A1, A2]), set([B1, B2, B3]), set([C1, C2])
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_multisource_2(self):
        comp = pnl.Composition()
        A1 = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A1')
        A2 = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A2')
        B1 = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='B1')
        B2 = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='B2')
        B3 = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='B3')
        C1 = pnl.TransferMechanism(function=pnl.Linear(intercept=1.5), name='C1')
        C2 = pnl.TransferMechanism(function=pnl.Linear(intercept=.5), name='C2')
        for m in [A1, A2, B1, B2, B3, C1, C2]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A1, B1)
        comp.add_projection(pnl.MappingProjection(), A1, B2)
        comp.add_projection(pnl.MappingProjection(), A2, B1)
        comp.add_projection(pnl.MappingProjection(), A2, B2)
        comp.add_projection(pnl.MappingProjection(), A2, B3)
        comp.add_projection(pnl.MappingProjection(), B1, C1)
        comp.add_projection(pnl.MappingProjection(), B2, C1)
        comp.add_projection(pnl.MappingProjection(), B1, C2)
        comp.add_projection(pnl.MappingProjection(), B3, C2)

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition_set({
            A1: pnl.Always(),
            A2: pnl.Always(),
            B1: pnl.EveryNCalls(A1, 2),
            B3: pnl.EveryNCalls(A2, 2),
            B2: pnl.All(pnl.EveryNCalls(A1, 4), pnl.EveryNCalls(A2, 4)),
            C1: pnl.Any(pnl.AfterNCalls(B1, 2), pnl.AfterNCalls(B2, 2)),
            C2: pnl.Any(pnl.AfterNCalls(B2, 2), pnl.AfterNCalls(B3, 2)),
        })

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.All(pnl.AfterNCalls(C1, 1), pnl.AfterNCalls(C2, 1))
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            set([A1, A2]), set([A1, A2]), set([B1, B3]), set([A1, A2]), set([A1, A2]), set([B1, B2, B3]), set([C1, C2])
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

@pytest.mark.psyneulink
class TestTermination:
    def test_termination_conditions_reset(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = pnl.TransferMechanism(function=pnl.Linear(intercept=4.0), name='scheduler-pytests-B')
        for m in [A, B]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, B)

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))

        sched.add_condition(B, pnl.EveryNCalls(A, 2))

        termination_conds = {}
        termination_conds[pnl.TimeScale.ENVIRONMENT_SEQUENCE] = pnl.AfterNEnvironmentStateUpdates(1)
        termination_conds[pnl.TimeScale.ENVIRONMENT_STATE_UPDATE] = pnl.AfterNCalls(B, 2)

        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A, A, B, A, A, B]
        assert output == pytest.helpers.setify_expected_output(expected_output)

        # reset the ENVIRONMENT_SEQUENCE because schedulers run ENVIRONMENT_STATE_UPDATEs
        sched.get_clock(sched.default_execution_id)._increment_time(pnl.TimeScale.ENVIRONMENT_SEQUENCE)
        sched._reset_counts_total(pnl.TimeScale.ENVIRONMENT_SEQUENCE, execution_id=sched.default_execution_id)

        output = list(sched.run())

        expected_output = [A, A, B]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_partial_override_scheduler(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(name='scheduler-pytests-A')
        B = pnl.TransferMechanism(name='scheduler-pytests-B')
        for m in [A, B]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, B)

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(comp))
        sched.add_condition(B, pnl.EveryNCalls(A, 2))
        termination_conds = {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.AfterNCalls(B, 2)}

        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A, A, B, A, A, B]
        assert output == pytest.helpers.setify_expected_output(expected_output)


@pytest.mark.psyneulink
class TestAbsoluteTime:

    @pytest.mark.parametrize(
        'conditions, interval',
        [
            ({'A': pnl.TimeInterval(repeat=8), 'B': pnl.TimeInterval(repeat=4)}, fractions.Fraction(4, 3) * pnl._unit_registry.ms),
            ({'A': pnl.TimeInterval(repeat=1), 'B': pnl.TimeInterval(repeat=3)}, fractions.Fraction(1, 3) * pnl._unit_registry.ms),
            ({'A': pnl.Any(pnl.TimeInterval(repeat=2), pnl.TimeInterval(repeat=3))}, fractions.Fraction(1, 3) * pnl._unit_registry.ms),
            ({'A': pnl.TimeInterval(repeat=6), 'B': pnl.TimeInterval(repeat=3)}, 1 * pnl._unit_registry.ms),
            ({'A': pnl.TimeInterval(repeat=100 * pnl._unit_registry.us), 'B': pnl.TimeInterval(repeat=2)}, fractions.Fraction(100, 3) * pnl._unit_registry.us),
            ({'A': pnl.Any(pnl.TimeInterval(repeat=1000 * pnl._unit_registry.us), pnl.TimeInterval(repeat=2))}, fractions.Fraction(1, 3) * pnl._unit_registry.ms),
            ({'A': pnl.TimeInterval(repeat=1000 * pnl._unit_registry.us), 'B': pnl.TimeInterval(repeat=2)}, fractions.Fraction(1, 3) * pnl._unit_registry.ms),
            ({'A': pnl.Any(pnl.TimeInterval(repeat=1000), pnl.TimeInterval(repeat=1500)), 'B': pnl.TimeInterval(repeat=2000)}, fractions.Fraction(500, 3) * pnl._unit_registry.ms),
        ]
    )
    def test_absolute_interval_linear(self, three_node_linear_composition, conditions, interval):
        [A, B, C], comp = three_node_linear_composition

        for node in conditions:
            comp.scheduler.add_condition(eval(node), conditions[node])

        assert comp.scheduler._get_absolute_consideration_set_execution_unit() == interval
