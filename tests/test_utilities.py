import logging
import types

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
