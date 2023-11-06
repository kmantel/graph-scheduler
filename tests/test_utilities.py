import inspect
import logging
import types

import networkx as nx
import pytest

import graph_scheduler as gs

root_logger = logging.getLogger()
gs_logger = logging.getLogger(gs.__name__)
gs_utils_logger = logging.getLogger(gs.utilities.__name__)

# root logger level may be modified elsewhere (currently is by psyneulink)
orig_root_logger_level = root_logger.level


class HashableTestObject(types.SimpleNamespace):
    def __hash__(self):
        return hash(id(self))


class LogFilterAll(logging.Filter):
    def filter(self, record):
        return False


t1 = HashableTestObject()
t2 = HashableTestObject()
test_graphs = [
    {t1: {1, 'A', t2}, 1: set(), 'A': {1}, t2: set()},
]
nx_digraph_types = [
    t for t in nx.__dict__.values()
    if inspect.isclass(t) and issubclass(t, nx.DiGraph)
]


def graph_as_nx_graph(graph, typ=nx.DiGraph):
    nx_graph = typ()

    nx_graph.add_nodes_from(graph.keys())
    for receiver, senders in graph.items():
        for s in senders:
            nx_graph.add_edge(s, receiver)

    return nx_graph


@pytest.mark.parametrize('graph', test_graphs)
def test_clone_graph(graph):
    res = gs.clone_graph(graph)

    assert graph is not res
    assert graph.keys() == res.keys()

    for node in graph:
        assert graph[node] == res[node]
        assert graph[node] is not res[node]


@pytest.fixture(scope='function')
def clean_logging(request):
    def _reset_loggers():
        root_logger.setLevel(orig_root_logger_level)
        gs_logger.setLevel(logging.NOTSET)

        for h in root_logger.handlers:
            root_logger.removeHandler(h)

        for h in gs_logger.handlers:
            gs_logger.removeHandler(h)

    filter_all = LogFilterAll()

    request.addfinalizer(_reset_loggers)

    gs_utils_logger.addFilter(filter_all)
    yield
    gs_utils_logger.removeFilter(filter_all)


@pytest.mark.parametrize('disable_level', logging._levelToName.values())
def test_debug_helpers(clean_logging, request, disable_level):
    def assert_initialized_gs_logger_properties():
        assert gs_logger.level == logging.NOTSET
        assert len(gs_logger.handlers) == 1
        assert gs_logger.handlers[0].level == logging.DEBUG

    disable_level = getattr(logging, disable_level)

    assert root_logger.level == orig_root_logger_level
    assert gs_logger.level == logging.NOTSET

    gs.enable_debug_logging()
    assert_initialized_gs_logger_properties()
    assert root_logger.level == logging.DEBUG

    # check new handler not created on second call
    gs.enable_debug_logging()
    assert_initialized_gs_logger_properties()

    gs.disable_debug_logging(disable_level)
    assert_initialized_gs_logger_properties()
    assert root_logger.level == disable_level

    gs.enable_debug_logging()
    assert_initialized_gs_logger_properties()
    assert root_logger.level == logging.DEBUG

    gs.disable_debug_logging()
    assert_initialized_gs_logger_properties()
    assert root_logger.level == logging.WARNING


@pytest.mark.parametrize('graph', test_graphs)
@pytest.mark.parametrize('nx_type', nx_digraph_types)
def test_convert_from_dependency_dict(graph, nx_type):
    res = gs.networkx_digraph_to_dependency_dict(
        gs.dependency_dict_to_networkx_digraph(graph, nx_type)
    )
    assert graph == res


@pytest.mark.parametrize('graph', test_graphs)
@pytest.mark.parametrize('nx_type', nx_digraph_types)
def test_convert_from_nx_graph(graph, nx_type):
    nx_graph = graph_as_nx_graph(graph, nx_type)

    res = gs.dependency_dict_to_networkx_digraph(
        gs.networkx_digraph_to_dependency_dict(nx_graph), nx_type,
    )
    assert nx_graph.nodes == nx_graph.nodes
    assert nx_graph.edges == res.edges


@pytest.mark.parametrize('graph', test_graphs)
@pytest.mark.parametrize('nx_type', nx_digraph_types)
@pytest.mark.parametrize('format', ['png', 'jpg', 'svg'])
def test_output_graph_image(graph, nx_type, format, tmp_path):
    fname = f'{tmp_path}_fig.{format}'
    nx_graph = graph_as_nx_graph(graph, nx_type)

    gs.output_graph_image(graph, fname, format)
    gs.output_graph_image(nx_graph, fname, format)


@pytest.mark.parametrize(
    'graph, expected_receivers',
    [
        (
            {'A': set(), 'B': {'A'}, 'C': {'A', 'B'}, 'D': {'E'}, 'E': {'D'}, 'F': set()},
            {'A': {'B', 'C'}, 'B': {'C'}, 'C': set(), 'D': {'E'}, 'E': {'D'}, 'F': set()},
        ),
    ]
)
def test_get_receivers(graph, expected_receivers):
    assert gs.get_receivers(graph) == expected_receivers
