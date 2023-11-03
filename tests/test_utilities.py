import types

import pytest

import graph_scheduler as gs


class HashableTestObject(types.SimpleNamespace):
    def __hash__(self):
        return hash(id(self))


t1 = HashableTestObject()
t2 = HashableTestObject()
test_graphs = [
    {t1: {1, 'A', t2}, 1: set(), 'A': {1}, t2: set()},
]


@pytest.mark.parametrize('graph', test_graphs)
def test_clone_graph(graph):
    res = gs.clone_graph(graph)

    assert graph is not res
    assert graph.keys() == res.keys()

    for node in graph:
        assert graph[node] == res[node]
        assert graph[node] is not res[node]
