import types

import pytest

import graph_scheduler as gs


class SimpleTestNode(types.SimpleNamespace):
    def __init__(self, name=None, is_finished_flag=True, value=0):
        super().__init__(name=name, is_finished_flag=is_finished_flag, value=value)

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return str(self.name)

    def is_finished(self, execution_id):
        return self.is_finished_flag

    def add(self, n=1):
        self.value += n
        return self.value


def pytest_assertrepr_compare(op, left, right):
    if isinstance(left, list) and isinstance(right, list) and op == '==':
        return [
            'Time Step output matching:',
            'Actual output:', str(left),
            'Expected output:', str(right)
        ]


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
def three_node_linear_scheduler():
    A = SimpleTestNode('A')
    B = SimpleTestNode('B')
    C = SimpleTestNode('C')

    sched = gs.Scheduler(graph=create_graph_from_pathways([A, B, C]))

    return sched.nodes, sched


@pytest.helpers.register
def get_test_node():
    return SimpleTestNode


@pytest.helpers.register
def run_scheduler(scheduler, func=lambda test_node: test_node, **run_kwargs):
    for execution_set in scheduler.run(**run_kwargs):
        for node in execution_set:
            func(node)

    return [node.value for node in scheduler.nodes]
