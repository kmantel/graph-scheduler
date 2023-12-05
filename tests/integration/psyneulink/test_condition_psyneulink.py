import numpy as np
import psyneulink as pnl
import pytest


@pytest.mark.psyneulink
class TestCondition:
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
            pnl.TimeScale.TRIAL: pnl.Threshold(A, parameter, threshold, '>=', indices=indices)
        }

        comp.run(inputs={A: np.ones(A.defaults.variable.shape)})

        np.testing.assert_array_equal(comp.results, expected_results)
