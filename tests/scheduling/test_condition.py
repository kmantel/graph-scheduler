import inspect
import logging
import warnings
from typing import Any, Dict, List, Union

import numpy as np
import pytest

import graph_scheduler as gs

logger = logging.getLogger(__name__)

SimpleTestNode = pytest.helpers.get_test_node()


test_graphs = {
    'five_node_hub': {'A': set(), 'B': set(), 'C': {'A', 'B'}, 'D': {'C'}, 'E': {'C'}},
    'nine_node_multi': {
        'A': set(),
        'B': set(),
        'C': {'A', 'B'},
        'D': {'A', 'C'},
        'E': {'C', 'H'},
        'F': {'D', 'E'},
        'G': set(),
        'H': {'G'},
        'I': {'H'}
    },
}


def all_single_pairs_parametrization(graph_name):
    graph = test_graphs[graph_name]
    return [(graph_name, a, b) for b in graph for a in graph if a != b]


def verify_operation(operation, test_neighbors, source_neighbors, subject_neighbors):
    print(operation)
    if operation is gs.Operation.KEEP:
        assert test_neighbors == source_neighbors
    elif operation is gs.Operation.INTERSECTION:
        assert test_neighbors == subject_neighbors.intersection(source_neighbors)
    elif operation is gs.Operation.MERGE:
        assert test_neighbors == subject_neighbors.union(source_neighbors)
    elif operation is gs.Operation.REPLACE:
        assert test_neighbors == subject_neighbors
    elif operation is gs.Operation.DISCARD:
        assert test_neighbors == set()
    elif operation == gs.Operation.DIFFERENCE:
        assert test_neighbors == source_neighbors.difference(subject_neighbors)
    elif operation == gs.Operation.SYMMETRIC_DIFFERENCE:
        assert test_neighbors == source_neighbors.symmetric_difference(subject_neighbors)
    elif operation == gs.Operation.INVERSE_DIFFERENCE:
        assert test_neighbors == subject_neighbors.difference(source_neighbors)
    else:
        assert False, f"Unrecognized operation {operation}"


def flatten_list(obj: List[Union[Any, List]]) -> List:
    new_list = []
    for o in obj:
        try:
            new_list.extend(o)
        except TypeError:
            new_list.append(o)
    return new_list


def gen_expected_conditions(
    indices: Union[List[int], Dict[Any, List[int]]],
    conditions: List[Union[gs.Condition, List[gs.Condition]]]
) -> Union[List[gs.Condition], Dict[Any, Union[gs.Condition, List[gs.Condition]]]]:
    """
    Used in tests to simplify parametrization notation. See
    TestConditionSet.test_add_condition for example usage.

    Args:
        indices: list of the indices of the Conditions (or lists of
            Conditions) in **conditions**, optionally as a dict mapping
            items to index lists

        conditions: a list of Conditions (or lists of Conditions)

    Returns:
        A list or dict (depending on the type of **indices**) containing
        the Conditions specified
    """
    def get_cond_elem(e):
        if isinstance(e, list):
            val = [conditions[i] for i in e]
        else:
            val = conditions[e]
        return val

    if isinstance(indices, list):
        return [get_cond_elem(i) for i in indices]
    else:
        return {owner: get_cond_elem(i) for owner, i in indices.items()}


def basic_Condition(*args):
    return gs.Condition(lambda: None, *args)


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

    # _indices parameters match indices of conditions parameter
    @pytest.mark.parametrize(
        'owners, conditions, expected_conditions_indices, expected_basic_indices, expected_structural_indices, expected_structural_order_indices',
        [
            pytest.param(['A'], [gs.condition._GSCWithOperations('B')], {'A': 0}, {}, {'A': [0]}, [0], id='single_gsc'),
            pytest.param(
                ['A', 'A'],
                [
                    basic_Condition('B'),
                    basic_Condition('C'),
                ],
                {'A': 1},
                {'A': 1},
                {},
                [],
                id='double_basic'
            ),
            pytest.param(
                ['A', 'A'],
                [
                    gs.condition._GSCWithOperations('B'),
                    gs.condition._GSCWithOperations('C'),
                ],
                {'A': [0, 1]},
                {},
                {'A': [0, 1]},
                [0, 1],
                id='double_gsc'
            ),
            pytest.param(
                ['A', 'B', 'A'],
                [
                    gs.condition._GSCWithOperations('B'),
                    gs.condition._GSCWithOperations('C'),
                    gs.condition._GSCWithOperations('C'),
                ],
                {'A': [0, 2], 'B': 1},
                {},
                {'A': [0, 2], 'B': [1]},
                [0, 1, 2],
                id='double_gsc_mixed_owners'
            ),
            pytest.param(
                ['A', 'B', 'A'],
                [
                    basic_Condition('B'),
                    gs.condition._GSCWithOperations('C'),
                    gs.condition._GSCWithOperations('B'),
                ],
                {'A': [0, 2], 'B': 1},
                {'A': 0},
                {'A': [2], 'B': [1]},
                [1, 2],
                id='single_basic_double_gsc_mixed_owners'
            ),
            pytest.param(
                ['A', 'B', 'A', 'A', 'A'],
                [
                    basic_Condition('B'),
                    gs.condition._GSCWithOperations('C'),
                    basic_Condition('C'),
                    gs.condition._GSCWithOperations('B'),
                    gs.condition._GSCWithOperations('C'),
                ],
                {'A': [2, 3, 4], 'B': 1},
                {'A': 2},
                {'A': [3, 4], 'B': [1]},
                [1, 3, 4],
                id='double_basic_triple_gsc_mixed_owners'
            ),
            pytest.param(
                ['A', 'B', 'A'],
                [
                    basic_Condition('A'),
                    gs.condition._GSCWithOperations('C'),
                    [
                        basic_Condition('B'),
                        gs.condition._GSCWithOperations('B'),
                        gs.condition._GSCWithOperations('C'),
                    ]
                ],
                {'A': [2, 3, 4], 'B': 1},
                {'A': 2},
                {'A': [3, 4], 'B': [1]},
                [1, 3, 4],
                id='double_basic_triple_gsc_mixed_owners_in_list'
            ),
        ]
    )
    @pytest.mark.parametrize(
        'constructor', [
            pytest.param(True, id='in_constructor'),
            pytest.param(False, id='after_construction')
        ]
    )
    def test_add_condition(
        self,
        owners,
        conditions,
        expected_conditions_indices,
        expected_basic_indices,
        expected_structural_indices,
        expected_structural_order_indices,
        constructor,
    ):
        n_conds = len(conditions)
        assert n_conds == len(owners)

        if constructor:
            condition_set = gs.ConditionSet(
                *[{owners[i]: conditions[i]} for i in range(n_conds)]
            )
        else:
            condition_set = gs.ConditionSet()
            for i in range(n_conds):
                if isinstance(conditions[i], list):
                    for c in conditions[i]:
                        condition_set.add_condition(owners[i], c)
                else:
                    condition_set.add_condition(owners[i], conditions[i])

        conditions = flatten_list(conditions)

        expected_conditions = gen_expected_conditions(expected_conditions_indices, conditions)
        expected_basic = gen_expected_conditions(expected_basic_indices, conditions)
        expected_structural = gen_expected_conditions(expected_structural_indices, conditions)
        expected_structural_order = gen_expected_conditions(expected_structural_order_indices, conditions)

        assert condition_set.conditions == expected_conditions
        assert condition_set.conditions_basic == expected_basic
        assert condition_set.conditions_structural == expected_structural
        assert condition_set.structural_condition_order == expected_structural_order

    @pytest.mark.parametrize(
        'constructor_args, additional_conditions',
        [
            ([{'A': basic_Condition('B')}, {'A': basic_Condition('B')}], []),
            ([{'A': basic_Condition('B')}], [{'A': basic_Condition('B')}]),
            ([], [{'A': basic_Condition('B')}, {'A': basic_Condition('B')}]),
        ]
    )
    def test_add_condition_warning(self, constructor_args, additional_conditions):
        with pytest.warns(UserWarning, match=r'Replacing basic condition for'):
            condition_set = gs.ConditionSet(*constructor_args)
            for addl_cset in additional_conditions:
                for owner, condition in addl_cset.items():
                    condition_set.add_condition(owner, condition)

    @pytest.mark.parametrize(
        'to_remove, expected_cond_owners',
        [
            ([], ['A', 'B']),
            (['A'], ['B']),
            (['B'], ['A']),
            (['A', 'B'], []),
            (['B', 'A'], []),
        ]
    )
    @pytest.mark.parametrize(
        'pass_condition', [
            pytest.param(True, id='pass_condition'),
            pytest.param(False, id='no_pass_condition')
        ]
    )
    def test_remove_condition_basic(self, to_remove, expected_cond_owners, pass_condition):
        conds = {
            'A': basic_Condition('A'),
            'B': basic_Condition('B'),
        }
        condition_set = gs.ConditionSet(conds)

        for owner in to_remove:
            if pass_condition:
                condition_set.remove_condition(conds[owner])
            else:
                condition_set.remove_condition(owner)

        assert condition_set.conditions == {owner: conds[owner] for owner in expected_cond_owners}

    def test_remove_condition_structural(self):
        gscA0 = gs.condition._GSCWithOperations('B')
        gscA1 = gs.condition._GSCWithOperations('C')

        gscB0 = gs.condition._GSCWithOperations('A')
        gscB1 = gs.condition._GSCWithOperations('C')

        conds = {
            'A': [gscA0, gscA1],
            'B': [gscB0, gscB1],
        }
        condition_set = gs.ConditionSet(conds)

        condition_set.remove_condition(gscB0)
        assert condition_set.conditions_structural == {'A': [gscA0, gscA1], 'B': [gscB1]}
        assert condition_set.structural_condition_order == [gscA0, gscA1, gscB1]

        condition_set.remove_condition(gscA1)
        assert condition_set.conditions_structural == {'A': [gscA0], 'B': [gscB1]}
        assert condition_set.structural_condition_order == [gscA0, gscB1]

        condition_set.remove_condition(gscA0)
        assert condition_set.conditions_structural == {'B': [gscB1]}
        assert condition_set.structural_condition_order == [gscB1]

        condition_set.remove_condition(gscB1)
        assert condition_set.conditions_structural == {}
        assert condition_set.structural_condition_order == []

    @pytest.mark.parametrize(
        'owners, conditions, removal_order_indices, expected_conditions_step_indices, expected_structural_order_step_indices',
        [
            (
                ['A', 'A', 'A'],
                [
                    gs.condition._GSCWithOperations('B'),
                    basic_Condition('A'),
                    gs.condition._GSCWithOperations('C')
                ],
                [1, 0, 2],
                [{'A': [0, 2]}, {'A': 2}, {}],
                [[0, 2], [2], []],
            ),
            (
                ['A', 'A', 'A'],
                [
                    gs.condition._GSCWithOperations('B'),
                    basic_Condition('A'),
                    gs.condition._GSCWithOperations('C')
                ],
                [0, 2, 1],
                [{'A': [1, 2]}, {'A': 1}, {}],  # basic conditions are always first
                [[2], [], []],
            ),
        ]
    )
    def test_remove_condition_mixed(
        self,
        owners,
        conditions,
        removal_order_indices,
        expected_conditions_step_indices,
        expected_structural_order_step_indices,
    ):
        n_conds = len(conditions)
        n_removals = len(removal_order_indices)

        assert n_conds == len(owners)
        assert n_removals == len(expected_conditions_step_indices)
        assert n_removals == len(expected_structural_order_step_indices)

        condition_set = gs.ConditionSet()
        for i in range(n_conds):
            condition_set.add_condition(owners[i], conditions[i])

        for i in range(n_removals):
            condition_set.remove_condition(conditions[removal_order_indices[i]])
            assert condition_set.conditions == gen_expected_conditions(
                expected_conditions_step_indices[i], conditions
            )
            assert condition_set.structural_condition_order == gen_expected_conditions(
                expected_structural_order_step_indices[i], conditions
            )

    def test_remove_condition_error_has_no_owner(self):
        condition_set = gs.ConditionSet()
        with pytest.raises(
            gs.ConditionError, match=r'Condition must have an owner to remove'
        ):
            condition_set.remove_condition(basic_Condition('A'))

    @pytest.mark.parametrize(
        'conditions',
        [
            ([basic_Condition('A'), gs.condition._GSCWithOperations('B')]),
            (
                [
                    gs.condition._GSCWithOperations('B'),
                    gs.condition._GSCWithOperations('C')
                ]
            ),
        ]
    )
    def test_remove_condition_error_multiple_possible(self, conditions):
        condition_set = gs.ConditionSet({'A': conditions})
        with pytest.raises(
            gs.ConditionError, match=r'Multiple possible conditions for'
        ):
            condition_set.remove_condition('A')

    def test_remove_condition_warning_condition_not_added(self):
        cond = basic_Condition('A')
        cond.owner = 'A'

        condition_set = gs.ConditionSet({'A': basic_Condition('A')})
        with pytest.warns(
            UserWarning, match=r'Condition .* not found for owner'
        ):
            condition_set.remove_condition(cond)

    def test_remove_condition_warning_owner_has_no_conditions(self):
        condition_set = gs.ConditionSet()
        with pytest.warns(
            UserWarning, match=r'Condition .* not found for owner'
        ):
            condition_set.remove_condition('B')


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


class TestGraphStructureCondition:
    class TestBaseManipulateGraph:
        @pytest.mark.parametrize(
            'graph_name, owner, subjects, owner_senders, owner_receivers, subject_senders, subject_receivers, expected_graph',
            [
                (
                    'nine_node_multi',
                    'E',
                    ['D'],
                    gs.Operation.DIFFERENCE,
                    gs.Operation.REPLACE,
                    gs.Operation.INTERSECTION,
                    gs.Operation.SYMMETRIC_DIFFERENCE,
                    {
                        'A': set(),
                        'B': set(),
                        'C': {'A', 'B'},
                        'D': {'C'},
                        'E': {'H'},
                        'F': {'E'},
                        'G': set(),
                        'H': {'G'},
                        'I': {'H'}
                    },
                ),
                (
                    'nine_node_multi',
                    'E',
                    ['D'],
                    gs.Operation.INTERSECTION,
                    gs.Operation.DISCARD,
                    gs.Operation.MERGE,
                    gs.Operation.INTERSECTION,
                    {
                        'A': set(),
                        'B': set(),
                        'C': {'A', 'B'},
                        'D': {'A', 'C', 'H'},
                        'E': {'C'},
                        'F': {'D'},
                        'G': set(),
                        'H': {'G'},
                        'I': {'H'},
                    },
                ),
                (
                    'nine_node_multi',
                    'C',
                    ['H', 'I'],
                    gs.Operation.KEEP,
                    gs.Operation.INTERSECTION,
                    {'H': gs.Operation.SYMMETRIC_DIFFERENCE, 'I': gs.Operation.MERGE},
                    gs.Operation.MERGE,
                    {
                        'A': set(),
                        'B': set(),
                        'C': {'A', 'B'},
                        'D': {'A', 'H', 'I'},
                        'E': {'C', 'H', 'I'},
                        'F': {'D', 'E'},
                        'G': set(),
                        'H': {'A', 'B', 'G'},
                        'I': {'A', 'B', 'H'},
                    },
                ),
                (
                    'nine_node_multi',
                    'C',
                    ['H', 'I'],
                    gs.Operation.INTERSECTION,
                    gs.Operation.DISCARD,
                    {'H': gs.Operation.MERGE, 'I': gs.Operation.REPLACE},
                    {'H': gs.Operation.INTERSECTION, 'I': gs.Operation.INVERSE_DIFFERENCE},
                    {
                        'A': set(),
                        'B': set(),
                        'C': set(),
                        'D': {'A', 'I'},
                        'E': {'H', 'I'},
                        'F': {'D', 'E'},
                        'G': set(),
                        'H': {'A', 'B', 'G'},
                        'I': {'A', 'B'},
                    },
                ),
            ],
        )
        def test_single_operation(
            self,
            graph_name,
            owner,
            subjects,
            owner_senders,
            owner_receivers,
            subject_senders,
            subject_receivers,
            expected_graph,
        ):
            cond = gs.condition._GSCWithOperations(
                *subjects,
                owner_senders=owner_senders,
                owner_receivers=owner_receivers,
                subject_senders=subject_senders,
                subject_receivers=subject_receivers,
                reconnect_non_subject_receivers=False,
                remove_new_self_referential_edges=False,
                debug=True
            )
            cond.owner = owner

            graph = test_graphs[graph_name]
            new_graph = cond.modify_graph(graph)
            old_receivers = gs.get_receivers(graph)
            new_receivers = gs.get_receivers(new_graph)

            combined_old_subject_receivers = set().union(*[old_receivers[s] for s in subjects])
            combined_old_subject_senders = set().union(*[graph[s] for s in subjects])

            for s in subjects:
                subject_senders_s = cond._get_subject_operation(subject_senders, s)
                subject_receivers_s = cond._get_subject_operation(subject_receivers, s)

                verify_operation(owner_senders, new_graph[owner], graph[owner], combined_old_subject_senders)
                verify_operation(owner_receivers, new_receivers[owner], old_receivers[owner], combined_old_subject_receivers)
                verify_operation(subject_senders_s, new_graph[s], graph[s], graph[owner])
                verify_operation(subject_receivers_s, new_receivers[s], old_receivers[s], old_receivers[owner])

            assert new_graph == expected_graph

        @pytest.mark.parametrize(
            'graph_name, owner, subjects, operation_args, err_msg_patterns',
            [
                (
                    'five_node_hub',
                    'C',
                    ['B'],
                    {'owner_senders': gs.Operation.KEEP, 'owner_receivers': gs.Operation.REPLACE},
                    [
                        r'owner_receivers\W+\(Operation.REPLACE\)\W+applied on C with D,E against C makes C a receiver of C',
                        r'owner_senders\W+\(Operation.KEEP\)\W+applied on C with A,B against {} does not make C a sender of C',
                    ]
                ),
                (
                    'five_node_hub',
                    'C',
                    ['B'],
                    {'owner_receivers': gs.Operation.REPLACE},
                    [
                        r'owner_receivers\W+\(Operation.REPLACE\)\W+applied on C with D,E against C makes C a receiver of C',
                        r'owner_senders\W+\(Operation.KEEP\)\W+applied on C with A,B against {} does not make C a sender of C',
                    ]
                ),
                (
                    'five_node_hub',
                    'C',
                    ['B'],
                    {'owner_senders': gs.Operation.KEEP, 'subject_receivers': gs.Operation.REPLACE},
                    [
                        r'owner_senders\W+\(Operation.KEEP\)\W+applied on C with A,B against {} makes B a sender of C',
                        r'subject_receivers\W+\(Operation.REPLACE\)\W+applied on B with C against D,E does not make C a receiver of B',
                    ]
                ),
                (
                    'five_node_hub',
                    'C',
                    ['E'],
                    {'subject_senders': gs.Operation.MERGE, 'owner_receivers': gs.Operation.REPLACE},
                    [
                        r'subject_senders\W+\(Operation.MERGE\)\W+applied on E with C against A,B makes C a sender of E',
                        r'owner_receivers\W+\(Operation.REPLACE\)\W+applied on C with D,E against {} does not make E a receiver of C',
                    ]
                ),
                (
                    'five_node_hub',
                    'C',
                    ['E', 'D'],
                    {'subject_senders': gs.Operation.DISCARD, 'owner_receivers': gs.Operation.MERGE},
                    [
                        r'owner_receivers\W+\(Operation.MERGE\)\W+applied on C with D,E against {} makes D a receiver of C',
                        r'subject_senders\W+\(Operation.DISCARD\)\W+applied on D with C against A,B does not make C a sender of D',
                        r'owner_receivers\W+\(Operation.MERGE\)\W+applied on C with D,E against {} makes E a receiver of C',
                        r'subject_senders\W+\(Operation.DISCARD\)\W+applied on E with C against A,B does not make C a sender of E',
                    ]
                ),
                (
                    'nine_node_multi',
                    'C',
                    ['H', 'I'],
                    {
                        'owner_senders': gs.Operation.INTERSECTION,
                        'owner_receivers': gs.Operation.MERGE,
                        'subject_receivers': {
                            'H': gs.Operation.SYMMETRIC_DIFFERENCE,
                            'I': gs.Operation.MERGE,
                        },
                    },
                    [
                        r'owner_receivers\W+\(Operation.MERGE\)\W+applied on C with D,E against E,I makes I a receiver of C',
                        r'subject_senders\W+\(Operation.KEEP\)\W+applied on I with H against A,B does not make C a sender of I',
                    ]
                ),
                (
                    'nine_node_multi',
                    'C',
                    ['H', 'I'],
                    {
                        'owner_senders': gs.Operation.MERGE,
                        'owner_receivers': gs.Operation.REPLACE,
                        'subject_senders': {
                            'H': gs.Operation.SYMMETRIC_DIFFERENCE,
                            'I': gs.Operation.MERGE,
                        },
                        'subject_receivers': gs.Operation.INTERSECTION,
                    },
                    [
                        r'owner_senders\W+\(Operation.MERGE\)\W+applied on C with A,B against G,H makes H a sender of C',
                        r'subject_receivers\W+\(Operation.INTERSECTION\)\W+applied on H with E,I against D,E does not make C a receiver of H',
                        r'subject_senders\W+\(Operation.MERGE\)\W+applied on I with H against A,B makes H a sender of I',
                        r'subject_receivers\W+\(Operation.INTERSECTION\)\W+applied on H with E,I against D,E does not make I a receiver of H',
                        r'owner_receivers\W+\(Operation.REPLACE\)\W+applied on C with D,E against E,I makes I a receiver of C',
                        r'subject_senders\W+\(Operation.MERGE\)\W+applied on I with H against A,B does not make C a sender of I',
                    ]
                ),
            ]
        )
        def test_conflicting_operations(
            self,
            graph_name,
            owner,
            subjects,
            operation_args,
            err_msg_patterns
        ):
            cond = gs.condition._GSCWithOperations(
                *subjects,
                **operation_args,
                reconnect_non_subject_receivers=False,
                remove_new_self_referential_edges=False,
                debug=True
            )
            cond.owner = owner

            graph = test_graphs[graph_name]

            err_msg = r'Conflict between operations:\W+{0}$'.format(
                r'\W+'.join(err_msg_patterns)
            )
            with pytest.raises(gs.ConditionError, match=err_msg):
                cond.modify_graph(graph)

        @pytest.mark.parametrize(
            'graph_name, owner, subjects, operation_args, expected_graph',
            [
                (
                    'five_node_hub',
                    'C',
                    ['D'],
                    {
                        'owner_receivers': gs.Operation.DISCARD,
                        'subject_senders': gs.Operation.DISCARD,
                    },
                    {'A': set(), 'B': set(), 'C': {'A', 'B'}, 'D': set(), 'E': {'A', 'B'}},
                ),
                (
                    'nine_node_multi',
                    'C',
                    ['E'],
                    {'owner_senders': gs.Operation.DISCARD},
                    {
                        'A': set(),
                        'B': set(),
                        'C': set(),
                        'D': {'A', 'B', 'C'},
                        'E': {'C', 'H'},
                        'F': {'D', 'E'},
                        'G': set(),
                        'H': {'G'},
                        'I': {'H'}
                    },
                ),
                (
                    'nine_node_multi',
                    'C',
                    ['D'],
                    {'owner_senders': gs.Operation.DISCARD},
                    {
                        'A': set(),
                        'B': set(),
                        'C': set(),
                        'D': {'A', 'C'},
                        'E': {'A', 'B', 'C', 'H'},
                        'F': {'D', 'E'},
                        'G': set(),
                        'H': {'G'},
                        'I': {'H'}
                    },
                )
            ]
        )
        def test_reconnect_non_subject_receivers(
            self, graph_name, owner, subjects, operation_args, expected_graph
        ):
            cond = gs.condition._GSCWithOperations(
                *subjects,
                **operation_args,
                reconnect_non_subject_receivers=True,
                remove_new_self_referential_edges=False,
                prune_cycles=False,
            )
            cond.owner = owner

            graph = test_graphs[graph_name]
            assert cond.modify_graph(graph) == expected_graph

    @pytest.mark.parametrize(
        'operation_arg, expected_arg_value',
        [
            ('KEEP', gs.Operation.KEEP),
            ('INTERSECTION', gs.Operation.INTERSECTION),
            ('UNION', gs.Operation.UNION),
            ('MERGE', gs.Operation.MERGE),
            ('REPLACE', gs.Operation.REPLACE),
            ('DISCARD', gs.Operation.DISCARD),
            ('DIFFERENCE', gs.Operation.DIFFERENCE),
            ('SYMMETRIC_DIFFERENCE', gs.Operation.SYMMETRIC_DIFFERENCE),
            ('INVERSE_DIFFERENCE', gs.Operation.INVERSE_DIFFERENCE),
        ]
    )
    @pytest.mark.parametrize(
        'operation',
        ['owner_senders', 'owner_receivers', 'subject_senders', 'subject_receivers']
    )
    def test_operation_argument_string_parsing(self, operation, operation_arg, expected_arg_value):
        n = 'A'
        cond = gs.condition._GSCWithOperations(n, **{operation: operation_arg})
        assert getattr(cond, operation) == expected_arg_value

        cond_lower = gs.condition._GSCWithOperations(n, **{operation: str.lower(operation_arg)})
        assert getattr(cond_lower, operation) == expected_arg_value

    @pytest.mark.parametrize(
        'operation_arg, expected_arg_value',
        [
            ('KEEP', gs.Operation.KEEP),
            ('keep', gs.Operation.KEEP),
            (gs.Operation.KEEP, gs.Operation.KEEP),
        ]
    )
    @pytest.mark.parametrize(
        'operation',
        ['subject_senders', 'subject_receivers']
    )
    def test_operation_argument_dict_parsing(self, operation, operation_arg, expected_arg_value):
        n = 'A'
        cond = gs.condition._GSCWithOperations(n, **{operation: {n: operation_arg}})
        operation_attr = getattr(cond, operation)
        assert operation_attr == {n: expected_arg_value}
        assert cond._get_subject_operation(operation_attr, n) == expected_arg_value

    def _get_graph_structure_condition_from_generic_class(self, cls_, graph):
        nodes = list(graph.keys())

        try:
            gsc = cls_(nodes[0])
        except TypeError as e:
            if "Can't instantiate abstract class" in str(e):
                pytest.skip('Class is abstract')
            else:
                raise
        except gs.ConditionError as e:
            if 'must be a callable' in str(e):
                gsc = cls_(lambda d: d)
            else:
                raise

        gsc.owner = nodes[1]
        return gsc

    @pytest.mark.parametrize(
        'cls_',
        [
            c for c in gs.condition.__dict__.values()
            if (
                inspect.isclass(c)
                and issubclass(
                    c, gs.condition.GraphStructureCondition
                )
            )
        ]
    )
    @pytest.mark.parametrize('graph', [test_graphs['five_node_hub']])
    def test_modify_graph_returns_clone(self, cls_, graph):
        gsc = self._get_graph_structure_condition_from_generic_class(cls_, graph)
        result_graph = gsc.modify_graph(graph)

        assert result_graph is not graph

        for k, v in result_graph.items():
            assert graph[k] is not v

    # not a foolproof test, but should catch some problem cases
    @pytest.mark.parametrize(
        'cls_',
        [
            c for c in gs.condition.__dict__.values()
            if (
                inspect.isclass(c)
                and issubclass(
                    c, gs.condition.GraphStructureCondition
                )
            )
        ]
    )
    @pytest.mark.parametrize('graph', [test_graphs['five_node_hub']])
    def test_postprocess_no_modification(self, cls_, graph):
        gsc = self._get_graph_structure_condition_from_generic_class(cls_, graph)
        orig_graph = gs.clone_graph(graph)
        gsc._postprocess({}, graph)

        assert orig_graph == graph

    def custom_gsc_func_1(graph):
        graph['B'].add('A')
        return graph

    @pytest.mark.parametrize(
        'func, nodes, graph, expected_graph',
        [
            (
                custom_gsc_func_1,
                ['B'],
                {'A': set(), 'B': set()},
                {'A': set(), 'B': {'A'}}
            ),
            (
                lambda self, graph: {k: graph[k] if k in self.nodes else {'C'} for k in graph},
                ['C'],
                {'A': 'B', 'B': set(), 'C': {'A'}},
                {'A': {'C'}, 'B': {'C'}, 'C': {'A'}},
            ),
        ]
    )
    def test_CustomGraphStructureCondition(self, func, nodes, graph, expected_graph):
        cond = gs.CustomGraphStructureCondition(func, nodes=nodes)
        cond.owner = 'A'
        assert cond.modify_graph(graph) == expected_graph

    def test_CustomGraphStructureCondition_wrong_self_pos(self):
        def func(a, self):
            pass
        cond = gs.CustomGraphStructureCondition(func)
        cond.owner = 'A'
        with pytest.raises(
            gs.ConditionError, match=r"If using 'self'.*must be the first argument"
        ):
            cond.modify_graph({'A': set(), 'B': set()})

    @staticmethod
    def _single_condition_test_helper(
        condition_type, graph_name, owner, nodes, warning_pat, expected_graph
    ):
        condition = condition_type(*nodes)
        condition.owner = owner

        if warning_pat is None:
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                try:
                    assert condition.modify_graph(test_graphs[graph_name]) == expected_graph
                except UserWarning as e:
                    if type(condition).__name__ in str(e) and 'already' in str(e):
                        raise
        else:
            with pytest.warns(UserWarning, match=warning_pat):
                assert condition.modify_graph(test_graphs[graph_name]) == expected_graph

    beforenodes_parametrizations = [
        (
            'five_node_hub', 'D', ['C'], None,
            {'A': set(), 'B': set(), 'C': {'A', 'B', 'D'}, 'D': {'A', 'B'}, 'E': {'C'}},
        ),
        (
            'five_node_hub', 'D', ['A'], None,
            {'A': {'D'}, 'B': set(), 'C': {'A', 'B'}, 'D': set(), 'E': {'C'}}
        ),
        (
            'five_node_hub', 'C', ['B'], None,
            {'A': set(), 'B': {'C'}, 'C': {'A'}, 'D': {'B', 'C'}, 'E': {'B', 'C'}},
        ),
        (
            'five_node_hub', 'C', ['D'], r'.*C is already before D.*Condition is ignored.',
            {'A': set(), 'B': set(), 'C': {'A', 'B'}, 'D': {'C'}, 'E': {'C'}},
        ),
        pytest.param(
            'five_node_hub', 'C', ['B', 'D'], r'.*C is already before D.*(?<!ignored.)$',
            {'A': set(), 'B': {'C'}, 'C': {'A'}, 'D': {'C'}, 'E': {'B', 'C'}},
            id='five_node_hub-C-B_D-Condition_not_ignored'
        ),
        (
            'five_node_hub', 'C', ['D', 'E'], r'.*C is already before D,E.*Condition is ignored.',
            {'A': set(), 'B': set(), 'C': {'A', 'B'}, 'D': {'C'}, 'E': {'C'}},
        ),
        (
            'nine_node_multi', 'E', ['C', 'G'], None,
            {
                'A': set(),
                'B': set(),
                'C': {'A', 'B', 'E'},
                'D': {'A', 'C'},
                'E': {'A', 'B'},
                'F': {'C', 'D', 'E', 'H'},
                'G': {'E'},
                'H': {'G'},
                'I': {'H'}
            },
        ),
        (
            'nine_node_multi', 'C', ['G'], None,
            {
                'A': set(),
                'B': set(),
                'C': {'A', 'B'},
                'D': {'A', 'C'},
                'E': {'C', 'H'},
                'F': {'D', 'E'},
                'G': {'C'},
                'H': {'G'},
                'I': {'H'}
            },
        ),
    ]

    @pytest.mark.parametrize(
        'graph_name, owner, nodes, warning_pat, expected_graph',
        beforenodes_parametrizations
    )
    def test_BeforeNodes(self, graph_name, owner, nodes, warning_pat, expected_graph):
        TestGraphStructureCondition._single_condition_test_helper(
            gs.BeforeNodes, graph_name, owner, nodes, warning_pat, expected_graph
        )

    @pytest.mark.parametrize(
        'graph_name, owner, nodes, warning_pat, expected_graph',
        filter(lambda p: len(p[2]) == 1, beforenodes_parametrizations),
    )
    def test_BeforeNode(self, graph_name, owner, nodes, warning_pat, expected_graph):
        TestGraphStructureCondition._single_condition_test_helper(
            gs.BeforeNode, graph_name, owner, nodes, warning_pat, expected_graph
        )

    @pytest.mark.parametrize(
        'graph_name, owner, nodes, warning_pat, expected_graph',
        [
            (
                'five_node_hub', 'A', ['C'], None,
                {'A': {'B'}, 'B': set(), 'C': {'B'}, 'D': {'C'}, 'E': {'C'}},
            ),
            (
                'five_node_hub', 'A', ['D'], None,
                {'A': {'C'}, 'B': set(), 'C': {'B'}, 'D': {'C'}, 'E': {'C'}}
            ),
            (
                'five_node_hub', 'D', ['E'], r'.*D is already with E.*Condition is ignored.',
                {'A': set(), 'B': set(), 'C': {'A', 'B'}, 'D': {'C'}, 'E': {'C'}},
            ),
            (
                'nine_node_multi', 'C', ['E'], None,
                {
                    'A': set(),
                    'B': set(),
                    'C': {'A', 'B', 'H'},
                    'D': {'A', 'C'},
                    'E': {'A', 'B', 'H'},
                    'F': {'D', 'E'},
                    'G': set(),
                    'H': {'G'},
                    'I': {'H'}
                },
            ),
            (
                'nine_node_multi', 'E', ['H'], None,
                {
                    'A': set(),
                    'B': set(),
                    'C': {'A', 'B'},
                    'D': {'A', 'C'},
                    'E': {'C', 'G'},
                    'F': {'D', 'E', 'H'},
                    'G': set(),
                    'H': {'C', 'G'},
                    'I': {'H'}
                },
            ),
        ]
    )
    def test_WithNode(self, graph_name, owner, nodes, warning_pat, expected_graph):
        TestGraphStructureCondition._single_condition_test_helper(
            gs.WithNode, graph_name, owner, nodes, warning_pat, expected_graph
        )

    afternodes_parametrizations = [
        (
            'five_node_hub', 'A', ['C'], None,
            {'A': {'C'}, 'B': set(), 'C': {'B'}, 'D': {'A', 'C'}, 'E': {'A', 'C'}},
        ),
        (
            'five_node_hub', 'A', ['D'], None,
            {'A': {'D'}, 'B': set(), 'C': {'B'}, 'D': {'C'}, 'E': {'C'}}
        ),
        (
            'five_node_hub', 'C', ['D'], None,
            {'A': set(), 'B': set(), 'C': {'A', 'B', 'D'}, 'D': {'A', 'B'}, 'E': {'C'}}
        ),
        (
            'five_node_hub', 'D', ['A'], None,
            {'A': set(), 'B': set(), 'C': {'A', 'B', 'D'}, 'D': {'A'}, 'E': {'C'}}
        ),
        (
            'five_node_hub', 'D', ['C'], r'.*D is already after C.*Condition is ignored.',
            {'A': set(), 'B': set(), 'C': {'A', 'B'}, 'D': {'C'}, 'E': {'C'}},
        ),
        # A->B comes from subject_senders MERGE
        pytest.param(
            'five_node_hub', 'C', ['B', 'D'], r'.*C is already after B.*(?<!ignored.)$',
            {'A': set(), 'B': {'A'}, 'C': {'A', 'B', 'D'}, 'D': {'A'}, 'E': {'C'}},
            id='five_node_hub-C-B_D-Condition_not_ignored'
        ),
        (
            'five_node_hub', 'C', ['A', 'B'], r'.*C is already after A,B.*Condition is ignored.',
            {'A': set(), 'B': set(), 'C': {'A', 'B'}, 'D': {'C'}, 'E': {'C'}},
        ),
        (
            'nine_node_multi', 'C', ['E', 'G'], None,
            {
                'A': set(),
                'B': set(),
                'C': {'A', 'B', 'E', 'G'},
                'D': {'A', 'C'},
                'E': {'A', 'B', 'H'},
                'F': {'C', 'D', 'E'},
                'G': {'A', 'B'},
                'H': {'G'},
                'I': {'H'}
            },
        ),
        (
            'nine_node_multi', 'C', ['E'], None,
            {
                'A': set(),
                'B': set(),
                'C': {'A', 'B', 'E'},
                'D': {'A', 'C'},
                'E': {'A', 'B', 'H'},
                'F': {'C', 'D', 'E'},
                'G': set(),
                'H': {'G'},
                'I': {'H'}
            },
        ),
        (
            'nine_node_multi', 'C', ['F'], None,
            {
                'A': set(),
                'B': set(),
                'C': {'A', 'B', 'F'},
                'D': {'A', 'B'},
                'E': {'A', 'B', 'H'},
                'F': {'A', 'B', 'D', 'E'},
                'G': set(),
                'H': {'G'},
                'I': {'H'}
            },
        ),
    ]

    @pytest.mark.parametrize(
        'graph_name, owner, nodes, warning_pat, expected_graph',
        afternodes_parametrizations
    )
    def test_AfterNodes(self, graph_name, owner, nodes, warning_pat, expected_graph):
        TestGraphStructureCondition._single_condition_test_helper(
            gs.AfterNodes, graph_name, owner, nodes, warning_pat, expected_graph
        )

    @pytest.mark.parametrize(
        'graph_name, owner, nodes, warning_pat, expected_graph',
        filter(lambda p: len(p[2]) == 1, afternodes_parametrizations),
    )
    def test_AfterNode(self, graph_name, owner, nodes, warning_pat, expected_graph):
        TestGraphStructureCondition._single_condition_test_helper(
            gs.AfterNode, graph_name, owner, nodes, warning_pat, expected_graph
        )
