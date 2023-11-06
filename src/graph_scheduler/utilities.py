import collections
import inspect
import logging
import weakref
from typing import Dict, Hashable, Set

import networkx as nx

__all__ = [
    'clone_graph', 'dependency_dict_to_networkx_digraph',
    'disable_debug_logging', 'enable_debug_logging',
    'networkx_digraph_to_dependency_dict',
]


logger = logging.getLogger(__name__)

_unused_args_sig_cache = weakref.WeakKeyDictionary()

typing_graph_dependency_dict = Dict[Hashable, Set[Hashable]]


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
