import types

import psyneulink as pnl
import pytest


def pytest_assertrepr_compare(op, left, right):
    if isinstance(left, list) and isinstance(right, list) and op == '==':
        return [
            'Time Step output matching:',
            'Actual output:', str(left),
            'Expected output:', str(right)
        ]


@pytest.helpers.register
def setify_expected_output(expected_output):
    type_set = type(set())
    for i in range(len(expected_output)):
        if type(expected_output[i]) is not type_set:
            try:
                iter(expected_output[i])
                expected_output[i] = set(expected_output[i])
            except TypeError:
                expected_output[i] = set([expected_output[i]])
    return expected_output


@pytest.helpers.register
def create_graph_from_pathways(*pathways):
    dependency_dict = {}
    for p in pathways:
        for i in range(len(p)):
            if p[i] not in dependency_dict:
                dependency_dict[p[i]] = set()

            try:
                dependency_dict[p[i + 1]].add(p[i])
            except KeyError:
                dependency_dict[p[i + 1]] = {p[i]}
            except IndexError:
                pass

    return dependency_dict


@pytest.helpers.register
def create_node(function=lambda x: x):
    return types.SimpleNamespace(function=function)


@pytest.fixture
def three_node_linear_composition():
    A = pnl.TransferMechanism(name='A')
    B = pnl.TransferMechanism(name='B')
    C = pnl.TransferMechanism(name='C')

    comp = pnl.Composition()
    comp.add_linear_processing_pathway([A, B, C])

    return comp.nodes, comp
