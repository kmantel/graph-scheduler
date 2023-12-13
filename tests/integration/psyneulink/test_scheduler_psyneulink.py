import numpy as np
import psyneulink as pnl
import pytest


@pytest.mark.psyneulink
class TestScheduler:
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
class TestLinear:
    def test_one_run_twice(self):
        A = pnl.IntegratorMechanism(
            name='A',
            default_variable=[0],
            function=pnl.SimpleIntegrator(
                rate=.5,
            )
        )

        c = pnl.Composition(pathways=[A])

        term_conds = {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.AfterNCalls(A, 2)}
        stim_list = {A: [[1]]}

        c.run(
            inputs=stim_list,
            termination_processing=term_conds
        )

        terminal_mech = A
        expected_output = [
            np.array([1.]),
        ]

        for i in range(len(expected_output)):
            np.testing.assert_allclose(expected_output[i], terminal_mech.get_output_values(c)[i])

    def test_two_AAB(self):
        A = pnl.IntegratorMechanism(
            name='A',
            default_variable=[0],
            function=pnl.SimpleIntegrator(
                rate=.5
            )
        )

        B = pnl.TransferMechanism(
            name='B',
            default_variable=[0],
            function=pnl.Linear(slope=2.0),
        )

        c = pnl.Composition(pathways=[A, B])

        term_conds = {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.AfterNCalls(B, 1)}
        stim_list = {A: [[1]]}

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(c))
        sched.add_condition(B, pnl.EveryNCalls(A, 2))
        c.scheduler = sched

        c.run(
            inputs=stim_list,
            termination_processing=term_conds
        )

        terminal_mech = B
        expected_output = [
            np.array([2.]),
        ]

        for i in range(len(expected_output)):
            np.testing.assert_allclose(expected_output[i], terminal_mech.get_output_values(c)[i])

    def test_two_ABB(self):
        A = pnl.TransferMechanism(
            name='A',
            default_variable=[0],
            function=pnl.Linear(slope=2.0),
        )

        B = pnl.IntegratorMechanism(
            name='B',
            default_variable=[0],
            function=pnl.SimpleIntegrator(
                rate=.5
            )
        )

        c = pnl.Composition(pathways=[A, B])

        term_conds = {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.AfterNCalls(B, 2)}
        stim_list = {A: [[1]]}

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(c))
        sched.add_condition(A, pnl.Any(pnl.AtPass(0), pnl.AfterNCalls(B, 2)))
        sched.add_condition(B, pnl.Any(pnl.JustRan(A), pnl.JustRan(B)))
        c.scheduler = sched

        c.run(
            inputs=stim_list,
            termination_processing=term_conds
        )

        terminal_mech = B
        expected_output = [
            np.array([2.]),
        ]

        for i in range(len(expected_output)):
            np.testing.assert_allclose(expected_output[i], terminal_mech.get_output_values(c)[i])


@pytest.mark.psyneulink
class TestBranching:
    def test_three_ABAC(self):
        A = pnl.IntegratorMechanism(
            name='A',
            default_variable=[0],
            function=pnl.SimpleIntegrator(
                rate=.5
            )
        )

        B = pnl.TransferMechanism(
            name='B',
            default_variable=[0],
            function=pnl.Linear(slope=2.0),
        )
        C = pnl.TransferMechanism(
            name='C',
            default_variable=[0],
            function=pnl.Linear(slope=2.0),
        )

        c = pnl.Composition(pathways=[[A, B], [A, C]])

        term_conds = {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.AfterNCalls(C, 1)}
        stim_list = {A: [[1]]}

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(c))
        sched.add_condition(B, pnl.Any(pnl.AtNCalls(A, 1), pnl.EveryNCalls(A, 2)))
        sched.add_condition(C, pnl.EveryNCalls(A, 2))
        c.scheduler = sched

        c.run(
            inputs=stim_list,
            termination_processing=term_conds
        )

        terminal_mechs = [B, C]
        expected_output = [
            [
                np.array([1.]),
            ],
            [
                np.array([2.]),
            ],
        ]

        for m in range(len(terminal_mechs)):
            for i in range(len(expected_output[m])):
                np.testing.assert_allclose(expected_output[m][i], terminal_mechs[m].get_output_values(c)[i])

    def test_three_ABAC_convenience(self):
        A = pnl.IntegratorMechanism(
            name='A',
            default_variable=[0],
            function=pnl.SimpleIntegrator(
                rate=.5
            )
        )

        B = pnl.TransferMechanism(
            name='B',
            default_variable=[0],
            function=pnl.Linear(slope=2.0),
        )
        C = pnl.TransferMechanism(
            name='C',
            default_variable=[0],
            function=pnl.Linear(slope=2.0),
        )

        c = pnl.Composition(pathways=[[A, B], [A, C]])

        term_conds = {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.AfterNCalls(C, 1)}
        stim_list = {A: [[1]]}

        c.scheduler.add_condition(B, pnl.Any(pnl.AtNCalls(A, 1), pnl.EveryNCalls(A, 2)))
        c.scheduler.add_condition(C, pnl.EveryNCalls(A, 2))

        c.run(
            inputs=stim_list,
            termination_processing=term_conds
        )

        terminal_mechs = [B, C]
        expected_output = [
            [
                np.array([1.]),
            ],
            [
                np.array([2.]),
            ],
        ]

        for m in range(len(terminal_mechs)):
            for i in range(len(expected_output[m])):
                np.testing.assert_allclose(expected_output[m][i], terminal_mechs[m].get_output_values(c)[i])

    def test_three_ABACx2(self):
        A = pnl.IntegratorMechanism(
            name='A',
            default_variable=[0],
            function=pnl.SimpleIntegrator(
                rate=.5
            )
        )

        B = pnl.TransferMechanism(
            name='B',
            default_variable=[0],
            function=pnl.Linear(slope=2.0),
        )
        C = pnl.TransferMechanism(
            name='C',
            default_variable=[0],
            function=pnl.Linear(slope=2.0),
        )

        c = pnl.Composition(pathways=[[A, B], [A, C]])

        term_conds = {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.AfterNCalls(C, 2)}
        stim_list = {A: [[1]]}

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(c))
        sched.add_condition(B, pnl.Any(pnl.AtNCalls(A, 1), pnl.EveryNCalls(A, 2)))
        sched.add_condition(C, pnl.EveryNCalls(A, 2))
        c.scheduler = sched

        c.run(
            inputs=stim_list,
            termination_processing=term_conds
        )

        terminal_mechs = [B, C]
        expected_output = [
            [
                np.array([3.]),
            ],
            [
                np.array([4.]),
            ],
        ]

        for m in range(len(terminal_mechs)):
            for i in range(len(expected_output[m])):
                np.testing.assert_allclose(expected_output[m][i], terminal_mechs[m].get_output_values(c)[i])

    def test_three_2_ABC(self):
        A = pnl.IntegratorMechanism(
            name='A',
            default_variable=[0],
            function=pnl.SimpleIntegrator(
                rate=.5
            )
        )

        B = pnl.IntegratorMechanism(
            name='B',
            default_variable=[0],
            function=pnl.SimpleIntegrator(
                rate=1
            )
        )

        C = pnl.TransferMechanism(
            name='C',
            default_variable=[0],
            function=pnl.Linear(slope=2.0),
        )

        c = pnl.Composition(pathways=[[A, C], [B, C]])

        term_conds = {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.AfterNCalls(C, 1)}
        stim_list = {A: [[1]], B: [[2]]}

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(c))
        sched.add_condition(C, pnl.All(pnl.EveryNCalls(A, 1), pnl.EveryNCalls(B, 1)))
        c.scheduler = sched

        c.run(
            inputs=stim_list,
            termination_processing=term_conds
        )

        terminal_mechs = [C]
        expected_output = [
            [
                np.array([5.]),
            ],
        ]

        for m in range(len(terminal_mechs)):
            for i in range(len(expected_output[m])):
                np.testing.assert_allclose(expected_output[m][i], terminal_mechs[m].get_output_values(c)[i])

    def test_three_2_ABCx2(self):
        A = pnl.IntegratorMechanism(
            name='A',
            default_variable=[0],
            function=pnl.SimpleIntegrator(
                rate=.5
            )
        )

        B = pnl.IntegratorMechanism(
            name='B',
            default_variable=[0],
            function=pnl.SimpleIntegrator(
                rate=1
            )
        )

        C = pnl.TransferMechanism(
            name='C',
            default_variable=[0],
            function=pnl.Linear(slope=2.0),
        )

        c = pnl.Composition(pathways=[[A, C], [B, C]])

        term_conds = {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.AfterNCalls(C, 2)}
        stim_list = {A: [[1]], B: [[2]]}

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(c))
        sched.add_condition(C, pnl.All(pnl.EveryNCalls(A, 1), pnl.EveryNCalls(B, 1)))
        c.scheduler = sched

        c.run(
            inputs=stim_list,
            termination_processing=term_conds
        )

        terminal_mechs = [C]
        expected_output = [
            [
                np.array([10.]),
            ],
        ]

        for m in range(len(terminal_mechs)):
            for i in range(len(expected_output[m])):
                np.testing.assert_allclose(expected_output[m][i], terminal_mechs[m].get_output_values(c)[i])

    def test_three_integrators(self):
        A = pnl.IntegratorMechanism(
            name='A',
            default_variable=[0],
            function=pnl.SimpleIntegrator(
                rate=1
            )
        )

        B = pnl.IntegratorMechanism(
            name='B',
            default_variable=[0],
            function=pnl.SimpleIntegrator(
                rate=1
            )
        )

        C = pnl.IntegratorMechanism(
            name='C',
            default_variable=[0],
            function=pnl.SimpleIntegrator(
                rate=1
            )
        )

        c = pnl.Composition(pathways=[[A, C], [B, C]])

        term_conds = {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.AfterNCalls(C, 2)}
        stim_list = {A: [[1]], B: [[1]]}

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(c))
        sched.add_condition(B, pnl.EveryNCalls(A, 2))
        sched.add_condition(C, pnl.Any(pnl.EveryNCalls(A, 1), pnl.EveryNCalls(B, 1)))
        c.scheduler = sched

        c.run(
            inputs=stim_list,
            termination_processing=term_conds
        )

        mechs = [A, B, C]
        expected_output = [
            [
                np.array([2.]),
            ],
            [
                np.array([1.]),
            ],
            [
                np.array([4.]),
            ],
        ]

        for m in range(len(mechs)):
            for i in range(len(expected_output[m])):
                np.testing.assert_allclose(expected_output[m][i], mechs[m].get_output_values(c)[i])

    def test_four_ABBCD(self):
        A = pnl.TransferMechanism(
            name='A',
            default_variable=[0],
            function=pnl.Linear(slope=2.0),
        )

        B = pnl.IntegratorMechanism(
            name='B',
            default_variable=[0],
            function=pnl.SimpleIntegrator(
                rate=.5
            )
        )

        C = pnl.IntegratorMechanism(
            name='C',
            default_variable=[0],
            function=pnl.SimpleIntegrator(
                rate=.5
            )
        )

        D = pnl.TransferMechanism(
            name='D',
            default_variable=[0],
            function=pnl.Linear(slope=1.0),
        )

        c = pnl.Composition(pathways=[[A, B, D], [A, C, D]])

        term_conds = {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.AfterNCalls(D, 1)}
        stim_list = {A: [[1]]}

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(c))
        sched.add_condition(B, pnl.EveryNCalls(A, 1))
        sched.add_condition(C, pnl.EveryNCalls(A, 2))
        sched.add_condition(D, pnl.Any(pnl.EveryNCalls(B, 3), pnl.EveryNCalls(C, 3)))
        c.scheduler = sched

        c.run(
            inputs=stim_list,
            termination_processing=term_conds
        )

        terminal_mechs = [D]
        expected_output = [
            [
                np.array([4.]),
            ],
        ]

        for m in range(len(terminal_mechs)):
            for i in range(len(expected_output[m])):
                np.testing.assert_allclose(expected_output[m][i], terminal_mechs[m].get_output_values(c)[i])

    def test_four_integrators_mixed(self):
        A = pnl.IntegratorMechanism(
            name='A',
            default_variable=[0],
            function=pnl.SimpleIntegrator(
                rate=1
            )
        )

        B = pnl.IntegratorMechanism(
            name='B',
            default_variable=[0],
            function=pnl.SimpleIntegrator(
                rate=1
            )
        )

        C = pnl.IntegratorMechanism(
            name='C',
            default_variable=[0],
            function=pnl.SimpleIntegrator(
                rate=1
            )
        )

        D = pnl.IntegratorMechanism(
            name='D',
            default_variable=[0],
            function=pnl.SimpleIntegrator(
                rate=1
            )
        )

        c = pnl.Composition(pathways=[[A, C], [A, D], [B, C], [B, D]])

        term_conds = {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.All(pnl.AfterNCalls(C, 1), pnl.AfterNCalls(D, 1))}
        stim_list = {A: [[1]], B: [[1]]}

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(c))
        sched.add_condition(B, pnl.EveryNCalls(A, 2))
        sched.add_condition(C, pnl.EveryNCalls(A, 1))
        sched.add_condition(D, pnl.EveryNCalls(B, 1))
        c.scheduler = sched

        c.run(
            inputs=stim_list,
            termination_processing=term_conds
        )

        mechs = [A, B, C, D]
        expected_output = [
            [
                np.array([2.]),
            ],
            [
                np.array([1.]),
            ],
            [
                np.array([4.]),
            ],
            [
                np.array([3.]),
            ],
        ]

        for m in range(len(mechs)):
            for i in range(len(expected_output[m])):
                np.testing.assert_allclose(expected_output[m][i], mechs[m].get_output_values(c)[i])

    def test_five_ABABCDE(self):
        A = pnl.TransferMechanism(
            name='A',
            default_variable=[0],
            function=pnl.Linear(slope=2.0),
        )

        B = pnl.TransferMechanism(
            name='B',
            default_variable=[0],
            function=pnl.Linear(slope=2.0),
        )

        C = pnl.IntegratorMechanism(
            name='C',
            default_variable=[0],
            function=pnl.SimpleIntegrator(
                rate=.5
            )
        )

        D = pnl.TransferMechanism(
            name='D',
            default_variable=[0],
            function=pnl.Linear(slope=1.0),
        )

        E = pnl.TransferMechanism(
            name='E',
            default_variable=[0],
            function=pnl.Linear(slope=2.0),
        )

        c = pnl.Composition(pathways=[[A, C, D], [B, C, E]])

        term_conds = {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.AfterNCalls(E, 1)}
        stim_list = {A: [[1]], B: [[2]]}

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(c))
        sched.add_condition(C, pnl.Any(pnl.EveryNCalls(A, 1), pnl.EveryNCalls(B, 1)))
        sched.add_condition(D, pnl.EveryNCalls(C, 1))
        sched.add_condition(E, pnl.EveryNCalls(C, 1))
        c.scheduler = sched

        c.run(
            inputs=stim_list,
            termination_processing=term_conds
        )

        terminal_mechs = [D, E]
        expected_output = [
            [
                np.array([3.]),
            ],
            [
                np.array([6.]),
            ],
        ]

        for m in range(len(terminal_mechs)):
            for i in range(len(expected_output[m])):
                np.testing.assert_allclose(expected_output[m][i], terminal_mechs[m].get_output_values(c)[i])

    #
    #   A  B
    #   |\/|
    #   C  D
    #   |\/|
    #   E  F
    #
    def test_six_integrators_threelayer_mixed(self):
        A = pnl.IntegratorMechanism(
            name='A',
            default_variable=[0],
            function=pnl.SimpleIntegrator(
                rate=1
            )
        )

        B = pnl.IntegratorMechanism(
            name='B',
            default_variable=[0],
            function=pnl.SimpleIntegrator(
                rate=1
            )
        )

        C = pnl.IntegratorMechanism(
            name='C',
            default_variable=[0],
            function=pnl.SimpleIntegrator(
                rate=1
            )
        )

        D = pnl.IntegratorMechanism(
            name='D',
            default_variable=[0],
            function=pnl.SimpleIntegrator(
                rate=1
            )
        )

        E = pnl.IntegratorMechanism(
            name='E',
            default_variable=[0],
            function=pnl.SimpleIntegrator(
                rate=1
            )
        )

        F = pnl.IntegratorMechanism(
            name='F',
            default_variable=[0],
            function=pnl.SimpleIntegrator(
                rate=1
            )
        )

        c = pnl.Composition(pathways=[[A, C, E], [A, C, F], [A, D, E], [A, D, F], [B, C, E], [B, C, F], [B, D, E], [B, D, F]])

        term_conds = {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.All(pnl.AfterNCalls(E, 1), pnl.AfterNCalls(F, 1))}
        stim_list = {A: [[1]], B: [[1]]}

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(c))
        sched.add_condition(B, pnl.EveryNCalls(A, 2))
        sched.add_condition(C, pnl.EveryNCalls(A, 1))
        sched.add_condition(D, pnl.EveryNCalls(B, 1))
        sched.add_condition(E, pnl.EveryNCalls(C, 1))
        sched.add_condition(F, pnl.EveryNCalls(D, 2))
        c.scheduler = sched

        c.run(
            inputs=stim_list,
            termination_processing=term_conds
        )

        # Intermediate consideration set executions
        #
        #     0   1   2   3
        #
        # A   1   2   3   4
        # B       1       2
        # C   1   4   8   14
        # D       3       9
        # E   1   8   19  42
        # F               23
        #
        expected_output = {
            A: [
                np.array([4.]),
            ],
            B: [
                np.array([2.]),
            ],
            C: [
                np.array([14.]),
            ],
            D: [
                np.array([9.]),
            ],
            E: [
                np.array([42.]),
            ],
            F: [
                np.array([23.]),
            ],
        }

        for m in expected_output:
            for i in range(len(expected_output[m])):
                np.testing.assert_allclose(expected_output[m][i], m.get_output_values(c)[i])


@pytest.mark.psyneulink
class TestTermination:
    def test_partial_override_composition(self):
        comp = pnl.Composition()
        A = pnl.TransferMechanism(name='scheduler-pytests-A')
        B = pnl.IntegratorMechanism(name='scheduler-pytests-B')
        for m in [A, B]:
            comp.add_node(m)
        comp.add_projection(pnl.MappingProjection(), A, B)

        termination_conds = {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.AfterNCalls(B, 2)}

        output = comp.run(inputs={A: 1}, termination_processing=termination_conds)
        # two executions of B
        assert output == [.75]

    def test_termination_conditions_reset_execution(self):
        A = pnl.IntegratorMechanism(
            name='A',
            default_variable=[0],
            function=pnl.SimpleIntegrator(
                rate=.5
            )
        )

        B = pnl.TransferMechanism(
            name='B',
            default_variable=[0],
            function=pnl.Linear(slope=2.0),
        )

        c = pnl.Composition(pathways=[[A, B]])

        term_conds = {pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.AfterNCalls(B, 2)}
        stim_list = {A: [[1]]}

        sched = pnl.Scheduler(**pytest.helpers.composition_to_scheduler_args(c))
        sched.add_condition(B, pnl.EveryNCalls(A, 2))
        c.scheduler = sched

        c.run(
            inputs=stim_list,
            termination_processing=term_conds
        )

        # A should run four times
        terminal_mech = B
        expected_output = [
            np.array([4.]),
        ]

        for i in range(len(expected_output)):
            np.testing.assert_allclose(expected_output[i], terminal_mech.get_output_values(c)[i])

        c.run(
            inputs=stim_list,
        )

        # A should run an additional two times
        terminal_mech = B
        expected_output = [
            np.array([6.]),
        ]

        for i in range(len(expected_output)):
            np.testing.assert_allclose(expected_output[i], terminal_mech.get_output_values(c)[i])


@pytest.mark.psyneulink
class TestFeedback:
    @pytest.mark.parametrize(
        'timescale, expected',
        [
            (pnl.TimeScale.CONSIDERATION_SET_EXECUTION, [[0.5], [0.4375]]),
            (pnl.TimeScale.PASS, [[0.5], [0.4375]]),
            (pnl.TimeScale.ENVIRONMENT_STATE_UPDATE, [[1.5], [0.4375]]),
            (pnl.TimeScale.ENVIRONMENT_SEQUENCE, [[1.5], [0.4375]])],
        ids=lambda x: x if isinstance(x, pnl.TimeScale) else ""
    )
    # 'LLVM' mode is not supported, because synchronization of compiler and
    # python values during execution is not implemented.
    @pytest.mark.usefixtures("comp_mode_no_llvm")
    def test_time_termination_measures(self, comp_mode, timescale, expected):
        in_one_pass = timescale in {pnl.TimeScale.CONSIDERATION_SET_EXECUTION, pnl.TimeScale.PASS}
        attention = pnl.TransferMechanism(
            name='Attention',
            integrator_mode=True,
            termination_threshold=3,
            termination_measure=timescale,
            execute_until_finished=in_one_pass
        )
        counter = pnl.IntegratorMechanism(
            function=pnl.AdaptiveIntegrator(rate=0.0, offset=1.0))

        response = pnl.IntegratorMechanism(
            function=pnl.AdaptiveIntegrator(rate=0.5))

        comp = pnl.Composition()
        comp.add_linear_processing_pathway([counter, response])
        comp.add_node(attention)
        comp.scheduler.add_condition(response, pnl.WhenFinished(attention))
        comp.scheduler.add_condition(counter, pnl.Always())
        inputs = {attention: [[0.5]], counter: [[2.0]]}
        result = comp.run(inputs=inputs, execution_mode=comp_mode)
        if comp_mode is pnl.ExecutionMode.Python:
            assert attention.execution_count == 3
            assert counter.execution_count == 1 if in_one_pass else 3
            assert response.execution_count == 1
        assert np.allclose(result, expected)

    @pytest.mark.parametrize(
        "condition,scale,expected_result",
        [
            (pnl.BeforeNCalls, pnl.TimeScale.ENVIRONMENT_STATE_UPDATE, [[.05, .05]]),
            (pnl.BeforeNCalls, pnl.TimeScale.PASS, [[.05, .05]]),
            (pnl.EveryNCalls, None, [[0.05, .05]]),
            (pnl.AtNCalls, pnl.TimeScale.ENVIRONMENT_STATE_UPDATE, [[.25, .25]]),
            (pnl.AtNCalls, pnl.TimeScale.ENVIRONMENT_SEQUENCE, [[.25, .25]]),
            (pnl.AfterNCalls, pnl.TimeScale.ENVIRONMENT_STATE_UPDATE, [[.25, .25]]),
            (pnl.AfterNCalls, pnl.TimeScale.PASS, [[.05, .05]]),
            (pnl.WhenFinished, None, [[1.0, 1.0]]),
            (pnl.WhenFinishedAny, None, [[1.0, 1.0]]),
            (pnl.WhenFinishedAll, None, [[1.0, 1.0]]),
            (pnl.All, None, [[1.0, 1.0]]),
            (pnl.Any, None, [[1.0, 1.0]]),
            (pnl.Not, None, [[.05, .05]]),
            (pnl.AllHaveRun, None, [[.05, .05]]),
            (pnl.Always, None, [[0.05, 0.05]]),
            (pnl.AtPass, None, [[.3, .3]]),
            (pnl.AtTrial, None, [[0.05, 0.05]]),
            # (pnl.Never), #TODO: Find a good test case for this!
        ]
    )
    # 'LLVM' mode is not supported, because synchronization of compiler and
    # python values during execution is not implemented.
    @pytest.mark.usefixtures("comp_mode_no_llvm")
    def test_scheduler_conditions(self, comp_mode, condition, scale, expected_result):
        decisionMaker = pnl.DDM(
            function=pnl.DriftDiffusionIntegrator(starting_value=0,
                                                  threshold=1,
                                                  noise=0.0,
                                                  time_step_size=1.0),
            reset_stateful_function_when=pnl.AtTrialStart(),
            execute_until_finished=False,
            output_ports=[pnl.DECISION_VARIABLE, pnl.RESPONSE_TIME],
            name='pnl.DDM'
        )

        response = pnl.ProcessingMechanism(size=2, name="GATE")

        comp = pnl.Composition()
        comp.add_linear_processing_pathway([decisionMaker, response])

        if condition is pnl.BeforeNCalls:
            comp.scheduler.add_condition(response, condition(decisionMaker, 5,
                                                             time_scale=scale))
        elif condition is pnl.AtNCalls:
            comp.scheduler.add_condition(response, condition(decisionMaker, 5,
                                                             time_scale=scale))
        elif condition is pnl.AfterNCalls:
            # Mechanisms run only once per PASS unless they are in
            # 'run_until_finished' mode.
            c = 1 if scale is pnl.TimeScale.PASS else 5
            comp.scheduler.add_condition(response, condition(decisionMaker, c,
                                                             time_scale=scale))
        elif condition is pnl.EveryNCalls:
            comp.scheduler.add_condition(response, condition(decisionMaker, 1))
        elif condition is pnl.WhenFinished:
            comp.scheduler.add_condition(response, condition(decisionMaker))
        elif condition is pnl.WhenFinishedAny:
            comp.scheduler.add_condition(response, condition(decisionMaker))
        elif condition is pnl.WhenFinishedAll:
            comp.scheduler.add_condition(response, condition(decisionMaker))
        elif condition is pnl.All:
            comp.scheduler.add_condition(response, condition(pnl.WhenFinished(decisionMaker)))
        elif condition is pnl.Any:
            comp.scheduler.add_condition(response, condition(pnl.WhenFinished(decisionMaker)))
        elif condition is pnl.Not:
            comp.scheduler.add_condition(response, condition(pnl.WhenFinished(decisionMaker)))
        elif condition is pnl.AllHaveRun:
            comp.scheduler.add_condition(response, condition(decisionMaker))
        elif condition is pnl.Always:
            comp.scheduler.add_condition(response, condition())
        elif condition is pnl.AtPass:
            comp.scheduler.add_condition(response, condition(5))
        elif condition is pnl.AtTrial:
            comp.scheduler.add_condition(response, condition(0))

        result = comp.run([0.05], execution_mode=comp_mode)
        # HACK: The result is an object dtype in Python mode for some reason?
        if comp_mode is pnl.ExecutionMode.Python:
            result = np.asfarray(result[0])
        assert np.allclose(result, expected_result)

    @pytest.mark.parametrize(
        "mode", [
            pnl.ExecutionMode.Python,
            pytest.param(pnl.ExecutionMode.LLVMRun, marks=pytest.mark.llvm),
            pytest.param(pnl.ExecutionMode.PTXRun, marks=[pytest.mark.llvm, pytest.mark.cuda]),
        ]
    )
    @pytest.mark.parametrize(
        "condition,scale,expected_result",
        [
            (pnl.AtTrial, None, [[[1.0]], [[2.0]]]),
        ]
    )
    def test_run_term_conditions(self, mode, condition, scale, expected_result):
        incrementing_mechanism = pnl.ProcessingMechanism(
            function=pnl.SimpleIntegrator
        )
        comp = pnl.Composition(
            pathways=[incrementing_mechanism]
        )
        comp.scheduler.termination_conds = {
            pnl.TimeScale.ENVIRONMENT_SEQUENCE: condition(2)
        }
        r = comp.run(inputs=[1], num_trials=5, execution_mode=mode)
        assert np.allclose(r, expected_result[-1])
        assert np.allclose(comp.results, expected_result)
