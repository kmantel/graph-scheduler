import collections
import functools
import inspect
import logging
import weakref
from typing import Dict, Hashable, List, Set, Union

import networkx as nx

__all__ = [
    'clone_graph', 'dependency_dict_to_networkx_digraph',
    'disable_debug_logging', 'enable_debug_logging', 'get_ancestors',
    'get_descendants', 'get_receivers', 'get_simple_cycles',
    'networkx_digraph_to_dependency_dict', 'output_graph_image',
]


logger = logging.getLogger(__name__)

_unused_args_sig_cache = weakref.WeakKeyDictionary()

typing_graph_dependency_dict = Dict[Hashable, Set[Hashable]]


class HashableDict(dict):
    def __hash__(self):
        return hash(frozenset((k, tuple(v)) for k, v in self.items()))


def prune_unused_args(func, args=None, kwargs=None):
    """
        Arguments
        ---------
            func : function

            args : *args

            kwargs : **kwargs


        Returns
        -------
            a tuple such that the first item is the intersection of **args** and the
            positional arguments of **func**, and the second item is the intersection
            of **kwargs** and the keyword arguments of **func**

    """
    # use the func signature to filter out arguments that aren't compatible
    try:
        sig = _unused_args_sig_cache[func]
    except KeyError:
        sig = inspect.signature(func)
        _unused_args_sig_cache[func] = sig

    has_args_param = False
    has_kwargs_param = False
    count_positional = 0
    func_kwargs_names = set()

    for name, param in sig.parameters.items():
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            has_args_param = True
        elif param.kind is inspect.Parameter.VAR_KEYWORD:
            has_kwargs_param = True
        elif param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD or param.kind is inspect.Parameter.KEYWORD_ONLY:
            if param.default is inspect.Parameter.empty:
                count_positional += 1
            func_kwargs_names.add(name)

    if args is not None:
        try:
            args = list(args)
        except TypeError:
            args = [args]

        if not has_args_param:
            args = args[:count_positional]
    else:
        args = []

    if kwargs is not None:
        kwargs = dict(kwargs)

        if not has_kwargs_param:
            filtered = set()
            for kw in kwargs:
                if kw not in func_kwargs_names:
                    filtered.add(kw)
            for kw in filtered:
                del kwargs[kw]
    else:
        kwargs = {}

    return args, kwargs


def call_with_pruned_args(func, *args, **kwargs):
    """
        Calls **func** with only the **args** and **kwargs** that
        exist in its signature
    """
    args, kwargs = prune_unused_args(func, args, kwargs)
    return func(*args, **kwargs)


def clone_graph(graph: typing_graph_dependency_dict) -> typing_graph_dependency_dict:
    """
    Returns a copy of dependency-dict-formatted **graph** where the
    nodes within are copied as references
    """
    res = {}
    for k, v in graph.items():
        if isinstance(v, collections.abc.Iterable) and not isinstance(v, str):
            v = set(v)
        else:
            v = set([v])
        res[k] = v
    return res


def enable_debug_logging():
    """
    Enables display of debugging logs using python logging module

    Note: sets python root logger level to DEBUG
    """
    root_logger = logging.getLogger()
    gs_logger = logging.getLogger(__package__)
    gs_logger.propagate = False

    if len(gs_logger.handlers) == 0:
        gs_logging_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s'
        )
        gs_debug_log_handler = logging.StreamHandler()
        gs_debug_log_handler.setLevel(logging.DEBUG)
        gs_debug_log_handler.setFormatter(gs_logging_formatter)
        gs_logger.addHandler(gs_debug_log_handler)

    new_level = logging.DEBUG
    root_logger.setLevel(new_level)

    logger.info(
        f"Changing root logger level to {logging.getLevelName(new_level)}"
    )
    logger.debug('Debug mode on')


def disable_debug_logging(level: int = logging.WARNING):
    """
    Disables display of debugging logs using python logging module

    Args:
        level (int, optional): Level to reset root logger to. Defaults
            to logging.WARNING (root logger default).
    """
    root_logger = logging.getLogger()

    logger.debug('Debug mode off')
    logger.info(
        f'Changing root logger level to {logging.getLevelName(level)}'
    )

    root_logger.setLevel(level)


def dependency_dict_to_networkx_digraph(
    graph: typing_graph_dependency_dict, nx_type=nx.DiGraph
) -> nx.DiGraph:
    """
    Converts a graph in dependency dict form to a networkx DiGraph of type
    **nx_type**

    Args:
        graph: a graph in dependency dict form
        nx_type (optional): a subclass of networkx.DiGraph

    Returns:
        networkx.DiGraph
    """
    # networkx treats dict arguments as {sender: {receivers}}, so
    # reverse directed graphs
    return nx_type(graph).reverse()


def networkx_digraph_to_dependency_dict(
    graph: nx.DiGraph, add_missing=True
) -> typing_graph_dependency_dict:
    """
    Converts a networkx DiGraph to a graph in dependency dict form

    Args:
        graph: a networkx.DiGraph
        add_missing: if True, adds empty set of dependencies for nodes
        listed only in dependencies of other nodes

    Returns:
        a graph in dependency dict form
    """
    res_graph = {}
    for sender, receivers in graph.adj.items():
        if add_missing and sender not in res_graph:
            res_graph[sender] = set()
        for rec in receivers:
            if rec not in res_graph:
                res_graph[rec] = set()
            res_graph[rec].add(sender)
    return res_graph


def output_graph_image(
    graph: Union[typing_graph_dependency_dict, nx.Graph],
    filename: str = None,
    format: str = 'png',
):
    """
    Writes an image representation of **graph** to file **filename**.

    Args:
        graph: a graph in dependency dict form
        filename (str, optional): full path of image to write. Defaults
            to 'graph-scheduler-figure-<graph id>.<format>' in the current
            directory.
        format (str, optional): image format. Many common formats
            supported. Pass None to display supported formats. Defaults
            to png.

    Requires:
        - system graphviz: https://graphviz.org/download
        - Python pydot: pip install pydot
    """
    if filename is None:
        filename = f'graph-scheduler-figure-{id(graph)}.{format}'

    if not isinstance(graph, nx.Graph):
        graph = dependency_dict_to_networkx_digraph(graph)

    try:
        pd = nx.drawing.nx_pydot.to_pydot(graph)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            'Python pydot is required for output_graph_image.'
            ' Install it with: pip install pydot'
        ) from e

    try:
        pd.write(filename, format=format)
    except AssertionError as e:
        raise AssertionError(
            f"Format '{format}' not recognized. Supported formats:"
            f" {', '.join(pd.formats)}"
        ) from e
    except FileNotFoundError as e:
        if '"dot" not found in path' in str(e):
            raise FileNotFoundError(
                'System graphviz is required for output_graph_image.'
                ' Install it from https://graphviz.org/download'
            ) from e
        else:
            raise

    print(f'graph_scheduler.output_graph_image: wrote {format} to {filename}')


def cached_graph_function(func):
    """
    Decorator that can be applied to cache the results of a function
    that takes a graph dependency dict
    """
    @functools.lru_cache()
    def cached_orig_func(graph: HashableDict):
        return func(graph)

    @functools.wraps(func)
    def graph_function_wrapper(graph: typing_graph_dependency_dict):
        return cached_orig_func(HashableDict(graph))

    return graph_function_wrapper


@cached_graph_function
def get_ancestors(graph: typing_graph_dependency_dict) -> Dict[Hashable, Set[Hashable]]:
    """
    Returns a dict containing the ancestors of each node in dependency
    dictionary **graph**

    Args:
        graph: a graph in dependency dict form
    """
    nx_graph = dependency_dict_to_networkx_digraph(graph)
    return {node: nx.ancestors(nx_graph, node) for node in graph}


@cached_graph_function
def get_descendants(graph: typing_graph_dependency_dict) -> Dict[Hashable, Set[Hashable]]:
    """
    Returns a dict containing the descendants of each node in dependency
    dictionary **graph**

    Args:
        graph: a graph in dependency dict form
    """
    nx_graph = dependency_dict_to_networkx_digraph(graph)
    return {node: nx.descendants(nx_graph, node) for node in graph}


@cached_graph_function
def get_receivers(graph: typing_graph_dependency_dict) -> Dict[Hashable, Set[Hashable]]:
    """
    Returns a dict containing the receivers of each node in dependency
    dictionary **graph**

    Args:
        graph: a graph in dependency dict form
    """
    receivers = {node: set() for node in graph}
    for node, senders in graph.items():
        for s in senders:
            # sender may not be in graph
            if s not in receivers:
                receivers[s] = set()
            receivers[s].add(node)
    return receivers


@cached_graph_function
def get_simple_cycles(graph: typing_graph_dependency_dict) -> List[List[Hashable]]:
    """
    Returns a list containing the simple cycles of each node in
    dependency dictionary **graph**

    Args:
        graph: a graph in dependency dict form
    """
    nx_graph = dependency_dict_to_networkx_digraph(graph)
    return list(nx.simple_cycles(nx_graph))
