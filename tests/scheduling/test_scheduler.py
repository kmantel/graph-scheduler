import fractions
import logging
import types

import networkx as nx
import pytest

import graph_scheduler as gs

logger = logging.getLogger(__name__)

SimpleTestNode = pytest.helpers.get_test_node()


test_graphs = {
    'three_node_linear': pytest.helpers.create_graph_from_pathways(['A', 'B', 'C']),
    'four_node_split': pytest.helpers.create_graph_from_pathways(['A', 'B', 'D'], ['C', 'D'])
}


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

        scheduler.get_clock(scheduler.default_execution_id)._increment_time(gs.TimeScale.ENVIRONMENT_STATE_UPDATE)

        eid = 'eid'
        eid1 = 'eid1'
        scheduler._init_counts(execution_id=eid)

        assert scheduler.clocks[eid].time.environment_state_update == 0

        scheduler.get_clock(scheduler.default_execution_id)._increment_time(gs.TimeScale.ENVIRONMENT_STATE_UPDATE)

        assert scheduler.clocks[eid].time.environment_state_update == 0

        scheduler._init_counts(execution_id=eid1, base_execution_id=scheduler.default_execution_id)

        assert scheduler.clocks[eid1].time.environment_state_update == 2

    def test_default_condition_1(self):
        A = SimpleTestNode('A')
        B = SimpleTestNode('B')
        C = SimpleTestNode('C')

        sched = gs.Scheduler(
            graph=pytest.helpers.create_graph_from_pathways([A, C], [A, B, C])
        )
        sched.add_condition(A, gs.AtPass(1))
        sched.add_condition(B, gs.Always())

        output = list(sched.run())
        expected_output = [B, A, B, C]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_default_condition_2(self):
        A = SimpleTestNode('A')
        B = SimpleTestNode('B')
        C = SimpleTestNode('C')

        sched = gs.Scheduler(
            graph=pytest.helpers.create_graph_from_pathways([A, B], [C])
        )
        sched.add_condition(C, gs.AtPass(1))

        output = list(sched.run())
        expected_output = [A, B, {C, A}]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_exact_time_mode(self):
        sched = gs.Scheduler(
            {'A': set(), 'B': {'A'}},
            mode=gs.SchedulingMode.EXACT_TIME
        )

        # these cannot run at same execution set unless in EXACT_TIME
        sched.add_condition('A', gs.TimeInterval(start=1))
        sched.add_condition('B', gs.TimeInterval(start=1))

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

    @pytest.mark.parametrize('add_method', ['add_graph_edge', 'add_condition_AddEdgeTo'])
    @pytest.mark.parametrize('remove_method', ['remove_graph_edge', 'add_condition_RemoveEdgeFrom'])
    def test_add_graph_structure_conditions(self, add_method, remove_method):
        def add_condition(owner, condition):
            if isinstance(condition, gs.AddEdgeTo) and add_method == 'add_graph_edge':
                return scheduler.add_graph_edge(owner, condition.node)
            elif isinstance(condition, gs.RemoveEdgeFrom) and remove_method == 'remove_graph_edge':
                return scheduler.remove_graph_edge(condition.node, owner)
            else:
                scheduler.add_condition(owner, condition)
                return condition

        initial_graph = pytest.helpers.create_graph_from_pathways(['A', 'B', 'C', 'D', 'E'])
        initial_conds = {'A': gs.AddEdgeTo('C')}
        scheduler = gs.Scheduler(initial_graph, initial_conds)

        assert scheduler.dependency_dict == {
            **initial_graph,
            **{'C': {'A', 'B'}},
        }
        assert len(scheduler._graphs) == 2
        assert scheduler._graphs[0] == initial_graph

        addl_conditions = [
            ('B', gs.AddEdgeTo('D')),
            ('B', gs.AddEdgeTo('E')),
            ('C', gs.AddEdgeTo('E')),
            ('E', gs.RemoveEdgeFrom('B')),
            ('D', gs.RemoveEdgeFrom('B')),
        ]

        for i, (owner, cond) in enumerate(addl_conditions):
            added_cond = add_condition(owner, cond)
            addl_conditions[i] = (owner, added_cond)

        assert scheduler.dependency_dict == {
            'A': set(),
            'B': {'A'},
            'C': {'A', 'B'},
            'D': {'C'},
            'E': {'C', 'D'},
        }
        assert scheduler._last_handled_structural_condition_order == (
            [initial_conds['A']] + [c[1] for c in addl_conditions]
        )

        # take only the first three elements in addl_conditions
        addl_conds_sub_idx = 3
        scheduler.conditions = gs.ConditionSet({
            **{
                k: [
                    addl_conditions[i][1] for i in range(addl_conds_sub_idx)
                    if addl_conditions[i][0] == k
                ]
                for k in initial_graph
            },
            'A': initial_conds['A'],
        })
        assert scheduler.dependency_dict == {
            'A': set(),
            'B': {'A'},
            'C': {'A', 'B'},
            'D': {'B', 'C'},
            'E': {'B', 'C', 'D'},
        }
        assert scheduler._last_handled_structural_condition_order == (
            [initial_conds['A']] + [c[1] for c in addl_conditions[:addl_conds_sub_idx]]
        )

    @pytest.mark.parametrize(
        'graph_name, conditions, expected_output',
        [
            ('three_node_linear', {'C': gs.BeforeNode('A')}, [{'C'}, {'A'}, {'B'}]),
            ('three_node_linear', {'B': gs.AfterNodes('C')}, [{'A'}, {'C'}, {'B'}]),
            ('four_node_split', {'D': gs.BeforeNodes('A', 'C')}, [{'D'}, {'A', 'C'}, {'B'}]),
        ]
    )
    def test_run_graph_structure_conditions(self, graph_name, conditions, expected_output):
        scheduler = gs.Scheduler(test_graphs[graph_name], conditions)
        output = list(scheduler.run())

        assert output == expected_output

    def test_gsc_creates_cyclic_graph(self):
        scheduler = gs.Scheduler(
            pytest.helpers.create_graph_from_pathways(['A', 'B', 'C'])
        )
        scheduler.add_condition('B', gs.EveryNCalls('A', 1))
        scheduler.add_condition('B', gs.AfterNode('C'))
        with pytest.warns(UserWarning, match='for B creates a cycle:'):
            scheduler.add_condition('B', gs.BeforeNode('A', prune_cycles=False))

        # If _build_consideration_queue failure not explicitly detected
        # and handled while adding BeforeNode('A') for 'B', the new
        # modified cyclic graph is pushed but the condition is not
        # added, resulting in incorrect state of scheduler._graphs.
        # Assert this doesn't happen.
        assert len(scheduler._graphs) == 3
        assert len(scheduler.conditions.structural_condition_order) == 2

        with pytest.raises(gs.SchedulerError, match='contains a cycle'):
            list(scheduler.run())

    def test_gsc_exact_time_warning(self):
        scheduler = gs.Scheduler(
            {'A': set(), 'B': set()}, mode=gs.SchedulingMode.EXACT_TIME
        )
        scheduler.add_condition('A', gs.AfterNode('B'))

        with pytest.warns(
            UserWarning,
            match='In exact time mode, graph structure conditions will have no effect'
        ):
            list(scheduler.run())


class TestLinear:
    def test_no_termination_conds(self):
        A = SimpleTestNode('A')
        B = SimpleTestNode('B')
        C = SimpleTestNode('C')
        sched = gs.Scheduler(graph=pytest.helpers.create_graph_from_pathways([A, B, C]))

        sched.add_condition(A, gs.EveryNPasses(1))
        sched.add_condition(B, gs.EveryNCalls(A, 2))
        sched.add_condition(C, gs.EveryNCalls(B, 3))

        output = list(sched.run())

        expected_output = [
            A, A, B, A, A, B, A, A, B, C,
        ]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

    # tests below are copied from old scheduler, need renaming
    def test_1(self):
        A = SimpleTestNode('A')
        B = SimpleTestNode('B')
        C = SimpleTestNode('C')
        sched = gs.Scheduler(graph=pytest.helpers.create_graph_from_pathways([A, B, C]))

        sched.add_condition(A, gs.EveryNPasses(1))
        sched.add_condition(B, gs.EveryNCalls(A, 2))
        sched.add_condition(C, gs.EveryNCalls(B, 3))

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AfterNCalls(C, 4, time_scale=gs.TimeScale.ENVIRONMENT_STATE_UPDATE)
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
        A = SimpleTestNode('A')
        B = SimpleTestNode('B')
        C = SimpleTestNode('C')
        sched = gs.Scheduler(graph=pytest.helpers.create_graph_from_pathways([A, B, C]))

        sched.add_condition(A, gs.EveryNPasses(1))
        sched.add_condition(B, gs.Any(gs.EveryNCalls(A, 2), gs.AfterPass(1)))
        sched.add_condition(C, gs.EveryNCalls(B, 3))

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AfterNCalls(C, 4, time_scale=gs.TimeScale.ENVIRONMENT_STATE_UPDATE)
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
        A = SimpleTestNode('A')
        B = SimpleTestNode('B')
        C = SimpleTestNode('C')
        sched = gs.Scheduler(graph=pytest.helpers.create_graph_from_pathways([A, B, C]))

        sched.add_condition(A, gs.EveryNPasses(1))
        sched.add_condition(B, gs.EveryNCalls(A, 2))
        sched.add_condition(C, gs.EveryNCalls(B, 2))

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AfterNCalls(C, 1, time_scale=gs.TimeScale.ENVIRONMENT_STATE_UPDATE)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A, A, B, A, A, B, C]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_3(self):
        A = SimpleTestNode('A')
        B = SimpleTestNode('B')
        C = SimpleTestNode('C')
        sched = gs.Scheduler(graph=pytest.helpers.create_graph_from_pathways([A, B, C]))

        sched.add_condition(A, gs.EveryNPasses(1))
        sched.add_condition(B, gs.EveryNCalls(A, 2))
        sched.add_condition(C, gs.All(gs.AfterNCalls(B, 2), gs.EveryNCalls(B, 1)))

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AfterNCalls(C, 4, time_scale=gs.TimeScale.ENVIRONMENT_STATE_UPDATE)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            A, A, B, A, A, B, C, A, A, B, C, A, A, B, C, A, A, B, C
        ]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_6(self):
        A = SimpleTestNode('A')
        B = SimpleTestNode('B')
        C = SimpleTestNode('C')
        sched = gs.Scheduler(graph=pytest.helpers.create_graph_from_pathways([A, B, C]))

        sched.add_condition(A, gs.BeforePass(5))
        sched.add_condition(B, gs.AfterNCalls(A, 5))
        sched.add_condition(C, gs.AfterNCalls(B, 1))

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AfterNCalls(C, 3)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            A, A, A, A, A, B, C, B, C, B, C
        ]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_6_two_environment_state_updates(self):
        A = SimpleTestNode('A')
        B = SimpleTestNode('B')
        C = SimpleTestNode('C')
        sched = gs.Scheduler(graph=pytest.helpers.create_graph_from_pathways([A, B, C]))

        sched.add_condition(A, gs.BeforePass(5))
        sched.add_condition(B, gs.AfterNCalls(A, 5))
        sched.add_condition(C, gs.AfterNCalls(B, 1))

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(2)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AfterNCalls(C, 3)
        for _ in range(5):
            list(sched.run(termination_conds))

        output = sched.execution_list[None]

        expected_output = [
            A, A, A, A, A, B, C, B, C, B, C,
            A, A, A, A, A, B, C, B, C, B, C
        ]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_7(self):
        A = SimpleTestNode('A')
        B = SimpleTestNode('B')
        sched = gs.Scheduler(graph={A: set(), B: {A}})

        sched.add_condition(A, gs.EveryNPasses(1))
        sched.add_condition(B, gs.EveryNCalls(A, 2))

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.Any(gs.AfterNCalls(A, 1), gs.AfterNCalls(B, 1))
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_8(self):
        A = SimpleTestNode('A')
        B = SimpleTestNode('B')
        sched = gs.Scheduler(graph={A: set(), B: {A}})

        sched.add_condition(A, gs.EveryNPasses(1))
        sched.add_condition(B, gs.EveryNCalls(A, 2))

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.All(gs.AfterNCalls(A, 1), gs.AfterNCalls(B, 1))
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A, A, B]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_9(self):
        A = SimpleTestNode('A')
        B = SimpleTestNode('B')
        sched = gs.Scheduler(graph={A: set(), B: {A}})

        sched.add_condition(A, gs.EveryNPasses(1))
        sched.add_condition(B, gs.WhenFinished(A))

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AfterNCalls(B, 2)

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
        A = SimpleTestNode('A', False)
        B = SimpleTestNode('B')

        sched = gs.Scheduler(graph={A: set(), B: {A}})

        sched.add_condition(A, gs.EveryNPasses(1))
        sched.add_condition(B, gs.WhenFinished(A))

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AtPass(5)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A, A, A, A, A]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_10(self):
        A = SimpleTestNode('A')
        B = SimpleTestNode('B')

        sched = gs.Scheduler(graph={A: set(), B: {A}})

        sched.add_condition(A, gs.EveryNPasses(1))
        sched.add_condition(B, gs.Any(gs.WhenFinished(A), gs.AfterNCalls(A, 3)))

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AfterNCalls(B, 5)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A, B, A, B, A, B, A, B, A, B]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_10b(self):
        A = SimpleTestNode('A', False)
        B = SimpleTestNode('B')

        sched = gs.Scheduler(graph={A: set(), B: {A}})

        sched.add_condition(A, gs.EveryNPasses(1))
        sched.add_condition(B, gs.Any(gs.WhenFinished(A), gs.AfterNCalls(A, 3)))

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AfterNCalls(B, 4)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A, A, A, B, A, B, A, B, A, B]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_10c(self):
        A = SimpleTestNode('A')
        B = SimpleTestNode('B')

        sched = gs.Scheduler(graph={A: set(), B: {A}})

        sched.add_condition(A, gs.EveryNPasses(1))
        sched.add_condition(B, gs.All(gs.WhenFinished(A), gs.AfterNCalls(A, 3)))

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AfterNCalls(B, 4)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A, A, A, B, A, B, A, B, A, B]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_10d(self):
        A = SimpleTestNode('A', False)
        B = SimpleTestNode('B')

        sched = gs.Scheduler(graph={A: set(), B: {A}})

        sched.add_condition(A, gs.EveryNPasses(1))
        sched.add_condition(B, gs.All(gs.WhenFinished(A), gs.AfterNCalls(A, 3)))

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AtPass(10)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A, A, A, A, A, A, A, A, A, A]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    ########################################
    # tests with linear schedositions
    ########################################
    def test_linear_AAB(self):
        A = SimpleTestNode('A')
        B = SimpleTestNode('B')
        sched = gs.Scheduler(graph={A: set(), B: {A}})

        sched.add_condition(A, gs.EveryNPasses(1))
        sched.add_condition(B, gs.EveryNCalls(A, 2))

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNCalls(B, 2, time_scale=gs.TimeScale.ENVIRONMENT_SEQUENCE)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AfterNCalls(B, 2, time_scale=gs.TimeScale.ENVIRONMENT_STATE_UPDATE)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A, A, B, A, A, B]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_linear_ABB(self):
        A = SimpleTestNode('A')
        B = SimpleTestNode('B')
        sched = gs.Scheduler(graph={A: set(), B: {A}})

        sched.add_condition(A, gs.Any(gs.AtPass(0), gs.EveryNCalls(B, 2)))
        sched.add_condition(B, gs.Any(gs.EveryNCalls(A, 1), gs.EveryNCalls(B, 1)))

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AfterNCalls(B, 8, time_scale=gs.TimeScale.ENVIRONMENT_STATE_UPDATE)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A, B, B, A, B, B, A, B, B, A, B, B]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_linear_ABBCC(self):
        A = SimpleTestNode('A')
        B = SimpleTestNode('B')
        C = SimpleTestNode('C')
        sched = gs.Scheduler(graph=pytest.helpers.create_graph_from_pathways([A, B, C]))

        sched.add_condition(A, gs.Any(gs.AtPass(0), gs.EveryNCalls(C, 2)))
        sched.add_condition(B, gs.Any(gs.JustRan(A), gs.JustRan(B)))
        sched.add_condition(C, gs.Any(gs.EveryNCalls(B, 2), gs.JustRan(C)))

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AfterNCalls(C, 4, time_scale=gs.TimeScale.ENVIRONMENT_STATE_UPDATE)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A, B, B, C, C, A, B, B, C, C]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_linear_ABCBC(self):
        A = SimpleTestNode('A')
        B = SimpleTestNode('B')
        C = SimpleTestNode('C')
        sched = gs.Scheduler(graph=pytest.helpers.create_graph_from_pathways([A, B, C]))

        sched.add_condition(A, gs.Any(gs.AtPass(0), gs.EveryNCalls(C, 2)))
        sched.add_condition(B, gs.Any(gs.EveryNCalls(A, 1), gs.EveryNCalls(C, 1)))
        sched.add_condition(C, gs.EveryNCalls(B, 1))

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AfterNCalls(C, 4, time_scale=gs.TimeScale.ENVIRONMENT_STATE_UPDATE)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A, B, C, B, C, A, B, C, B, C]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    ########################################
    # tests with small branching schedositions
    ########################################


class TestBranching:
    #   triangle:         A
    #                    / \
    #                   B   C

    def test_triangle_1(self):
        A = SimpleTestNode('A')
        B = SimpleTestNode('B')
        C = SimpleTestNode('C')
        sched = gs.Scheduler(graph={A: set(), B: {A}, C: {A}})

        sched.add_condition(A, gs.EveryNPasses(1))
        sched.add_condition(B, gs.EveryNCalls(A, 1))
        sched.add_condition(C, gs.EveryNCalls(A, 1))

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AfterNCalls(C, 3, time_scale=gs.TimeScale.ENVIRONMENT_STATE_UPDATE)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            A, set([B, C]),
            A, set([B, C]),
            A, set([B, C]),
        ]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_triangle_2(self):
        A = SimpleTestNode('A')
        B = SimpleTestNode('B')
        C = SimpleTestNode('C')
        sched = gs.Scheduler(graph={A: set(), B: {A}, C: {A}})

        sched.add_condition(A, gs.EveryNPasses(1))
        sched.add_condition(B, gs.EveryNCalls(A, 1))
        sched.add_condition(C, gs.EveryNCalls(A, 2))

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AfterNCalls(C, 3, time_scale=gs.TimeScale.ENVIRONMENT_STATE_UPDATE)
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
        A = SimpleTestNode('A')
        B = SimpleTestNode('B')
        C = SimpleTestNode('C')
        sched = gs.Scheduler(graph={A: set(), B: {A}, C: {A}})

        sched.add_condition(A, gs.EveryNPasses(1))
        sched.add_condition(B, gs.EveryNCalls(A, 2))
        sched.add_condition(C, gs.EveryNCalls(A, 3))

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AfterNCalls(C, 2, time_scale=gs.TimeScale.ENVIRONMENT_STATE_UPDATE)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            A, A, B, A, C, A, B, A, A, set([B, C])
        ]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

    # this is test 11 of original constraint_scheduler.py
    def test_triangle_4(self):
        A = SimpleTestNode('A')
        B = SimpleTestNode('B')
        C = SimpleTestNode('C')

        sched = gs.Scheduler(graph={A: set(), B: {A}, C: {A}})

        sched.add_condition(A, gs.EveryNPasses(1))
        sched.add_condition(B, gs.EveryNCalls(A, 2))
        sched.add_condition(C, gs.All(gs.WhenFinished(A), gs.AfterNCalls(B, 3)))

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AfterNCalls(C, 1)
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
        A = SimpleTestNode('A')
        B = SimpleTestNode('B')
        C = SimpleTestNode('C')

        sched = gs.Scheduler(graph={A: set(), B: {A}, C: {A}})

        sched.add_condition(A, gs.EveryNPasses(1))
        sched.add_condition(B, gs.EveryNCalls(A, 2))
        sched.add_condition(C, gs.All(gs.WhenFinished(A), gs.AfterNCalls(B, 3)))

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AfterNCalls(C, 1)
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
        A = SimpleTestNode('A')
        B = SimpleTestNode('B')
        C = SimpleTestNode('C')

        sched = gs.Scheduler(graph={A: set(), B: set(), C: {A, B}})

        sched.add_condition(A, gs.EveryNPasses(1))
        sched.add_condition(B, gs.EveryNCalls(A, 2))
        sched.add_condition(C, gs.Any(gs.AfterNCalls(A, 3), gs.AfterNCalls(B, 3)))

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AfterNCalls(C, 4, time_scale=gs.TimeScale.ENVIRONMENT_STATE_UPDATE)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            A, set([A, B]), A, C, set([A, B]), C, A, C, set([A, B]), C
        ]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

    # this is test 5 of original constraint_scheduler.py
    # this test has an implicit priority set of A<B !
    def test_invtriangle_2(self):
        A = SimpleTestNode('A')
        B = SimpleTestNode('B')
        C = SimpleTestNode('C')

        sched = gs.Scheduler(graph={A: set(), B: set(), C: {A, B}})

        sched.add_condition(A, gs.EveryNPasses(1))
        sched.add_condition(B, gs.EveryNCalls(A, 2))
        sched.add_condition(C, gs.All(gs.AfterNCalls(A, 3), gs.AfterNCalls(B, 3)))

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AfterNCalls(C, 2, time_scale=gs.TimeScale.ENVIRONMENT_STATE_UPDATE)
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
        A = SimpleTestNode('A')
        B = SimpleTestNode('B')
        C = SimpleTestNode('C')
        D = SimpleTestNode('D')

        sched = gs.Scheduler(graph={A: set(), B: {A}, C: set(), D: {B, C}})

        sched.add_condition(A, gs.Always())
        sched.add_condition(B, gs.Always())
        sched.add_condition(C, gs.Always())
        sched.add_condition(D, gs.Always())

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AfterNCalls(D, 1, time_scale=gs.TimeScale.ENVIRONMENT_STATE_UPDATE)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            set([A, C]), B, D
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_checkmark_2(self):
        A = SimpleTestNode('A')
        B = SimpleTestNode('B')
        C = SimpleTestNode('C')
        D = SimpleTestNode('D')

        sched = gs.Scheduler(graph={A: set(), B: {A}, C: set(), D: {B, C}})

        sched.add_condition(A, gs.EveryNPasses(1))
        sched.add_condition(B, gs.EveryNCalls(A, 2))
        sched.add_condition(C, gs.EveryNCalls(A, 2))
        sched.add_condition(D, gs.All(gs.EveryNCalls(B, 2), gs.EveryNCalls(C, 2)))

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AfterNCalls(D, 1, time_scale=gs.TimeScale.ENVIRONMENT_STATE_UPDATE)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            A, set([A, C]), B, A, set([A, C]), B, D
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_checkmark2_1(self):
        A = SimpleTestNode('A')
        B = SimpleTestNode('B')
        C = SimpleTestNode('C')
        D = SimpleTestNode('D')

        sched = gs.Scheduler(graph={A: set(), B: {A}, C: set(), D: {A, B, C}})

        sched.add_condition(A, gs.EveryNPasses(1))
        sched.add_condition(B, gs.EveryNCalls(A, 2))
        sched.add_condition(C, gs.EveryNCalls(A, 2))
        sched.add_condition(D, gs.All(gs.EveryNCalls(B, 2), gs.EveryNCalls(C, 2)))

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AfterNCalls(D, 1, time_scale=gs.TimeScale.ENVIRONMENT_STATE_UPDATE)
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
        A1 = SimpleTestNode(name='A1')
        A2 = SimpleTestNode(name='A2')
        B1 = SimpleTestNode(name='B1')
        B2 = SimpleTestNode(name='B2')
        B3 = SimpleTestNode(name='B3')
        C1 = SimpleTestNode(name='C1')
        C2 = SimpleTestNode(name='C2')

        sched = gs.Scheduler(
            graph={
                A1: set(), A2: set(),
                B1: {A1}, B2: {A1, A2}, B3: {A2},
                C1: {B1, B2}, C2: {B2, B3}
            }
        )

        for m in sched.nodes:
            sched.add_condition(m, gs.Always())

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.All(gs.AfterNCalls(C1, 1), gs.AfterNCalls(C2, 1))
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            set([A1, A2]), set([B1, B2, B3]), set([C1, C2])
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_multisource_2(self):
        A1 = SimpleTestNode(name='A1')
        A2 = SimpleTestNode(name='A2')
        B1 = SimpleTestNode(name='B1')
        B2 = SimpleTestNode(name='B2')
        B3 = SimpleTestNode(name='B3')
        C1 = SimpleTestNode(name='C1')
        C2 = SimpleTestNode(name='C2')

        sched = gs.Scheduler(
            graph={
                A1: set(), A2: set(),
                B1: {A1}, B2: {A1, A2}, B3: {A2},
                C1: {B1, B2}, C2: {B2, B3}
            }
        )

        sched.add_condition_set({
            A1: gs.Always(),
            A2: gs.Always(),
            B1: gs.EveryNCalls(A1, 2),
            B3: gs.EveryNCalls(A2, 2),
            B2: gs.All(gs.EveryNCalls(A1, 4), gs.EveryNCalls(A2, 4)),
            C1: gs.Any(gs.AfterNCalls(B1, 2), gs.AfterNCalls(B2, 2)),
            C2: gs.Any(gs.AfterNCalls(B2, 2), gs.AfterNCalls(B3, 2)),
        })

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.All(gs.AfterNCalls(C1, 1), gs.AfterNCalls(C2, 1))
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            set([A1, A2]), set([A1, A2]), set([B1, B3]), set([A1, A2]), set([A1, A2]), set([B1, B2, B3]), set([C1, C2])
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

class TestTermination:
    def test_termination_conditions_reset(self):
        A = SimpleTestNode('A')
        B = SimpleTestNode('B')
        sched = gs.Scheduler(graph={A: set(), B: {A}})

        sched.add_condition(B, gs.EveryNCalls(A, 2))

        termination_conds = {}
        termination_conds[gs.TimeScale.ENVIRONMENT_SEQUENCE] = gs.AfterNEnvironmentStateUpdates(1)
        termination_conds[gs.TimeScale.ENVIRONMENT_STATE_UPDATE] = gs.AfterNCalls(B, 2)

        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A, A, B, A, A, B]
        assert output == pytest.helpers.setify_expected_output(expected_output)

        # reset the ENVIRONMENT_SEQUENCE because schedulers run ENVIRONMENT_STATE_UPDATEs
        sched.get_clock(sched.default_execution_id)._increment_time(gs.TimeScale.ENVIRONMENT_SEQUENCE)
        sched._reset_counts_total(gs.TimeScale.ENVIRONMENT_SEQUENCE, execution_id=sched.default_execution_id)

        output = list(sched.run())

        expected_output = [A, A, B]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_partial_override_scheduler(self):
        A = SimpleTestNode('A')
        B = SimpleTestNode('B')
        sched = gs.Scheduler(graph={A: set(), B: {A}})
        sched.add_condition(B, gs.EveryNCalls(A, 2))
        termination_conds = {gs.TimeScale.ENVIRONMENT_STATE_UPDATE: gs.AfterNCalls(B, 2)}

        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A, A, B, A, A, B]
        assert output == pytest.helpers.setify_expected_output(expected_output)


class TestAbsoluteTime:

    @pytest.mark.parametrize(
        'conditions, interval',
        [
            ({'A': gs.TimeInterval(repeat=8), 'B': gs.TimeInterval(repeat=4)}, fractions.Fraction(4, 3) * gs._unit_registry.ms),
            ({'A': gs.TimeInterval(repeat=1), 'B': gs.TimeInterval(repeat=3)}, fractions.Fraction(1, 3) * gs._unit_registry.ms),
            ({'A': gs.Any(gs.TimeInterval(repeat=2), gs.TimeInterval(repeat=3))}, fractions.Fraction(1, 3) * gs._unit_registry.ms),
            ({'A': gs.TimeInterval(repeat=6), 'B': gs.TimeInterval(repeat=3)}, 1 * gs._unit_registry.ms),
            ({'A': gs.TimeInterval(repeat=100 * gs._unit_registry.us), 'B': gs.TimeInterval(repeat=2)}, fractions.Fraction(100, 3) * gs._unit_registry.us),
            ({'A': gs.Any(gs.TimeInterval(repeat=1000 * gs._unit_registry.us), gs.TimeInterval(repeat=2))}, fractions.Fraction(1, 3) * gs._unit_registry.ms),
            ({'A': gs.TimeInterval(repeat=1000 * gs._unit_registry.us), 'B': gs.TimeInterval(repeat=2)}, fractions.Fraction(1, 3) * gs._unit_registry.ms),
            ({'A': gs.Any(gs.TimeInterval(repeat=1000), gs.TimeInterval(repeat=1500)), 'B': gs.TimeInterval(repeat=2000)}, fractions.Fraction(500, 3) * gs._unit_registry.ms),
        ]
    )
    def test_absolute_interval_linear(self, three_node_linear_scheduler, conditions, interval):
        [A, B, C], sched = three_node_linear_scheduler

        for node in conditions:
            sched.add_condition(eval(node), conditions[node])

        assert sched._get_absolute_consideration_set_execution_unit() == interval
