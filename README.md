# Graph Scheduler

[![CI](https://github.com/kmantel/graph-scheduler/actions/workflows/ci.yml/badge.svg)](https://github.com/kmantel/graph-scheduler/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/github/kmantel/graph-scheduler/badge.svg)](https://coveralls.io/github/kmantel/graph-scheduler)

A graph scheduler generates the order in which the nodes of a directed
acyclic graph (DAG) are executed using the structure of the graph and
expressive
[conditions](https://kmantel.github.io/graph-scheduler/Condition.html).
Specifically, a scheduler uses a topological ordering of the nodes as a
base sequence of execution and further restricts execution based on
predefined or custom conditions provided by the user. Patterns of
execution are linked to abstract units of time and may optionally be
mapped to real time units using [pint](https://pint.readthedocs.io/).

Documentation is available on github-pages [for the current
release](https://kmantel.github.io/graph-scheduler/) and [for the
current main
branch](https://kmantel.github.io/graph-scheduler/branch/main). For
prior releases, go to
``https://kmantel.github.io/graph-scheduler/tag/<tag_name>``.

## Installation

Install from pypi:

```sh
pip install graph-scheduler
```

## Example

The graph is specified here in dependency dictionary format, but
[networkx](https://github.com/networkx/networkx) Digraphs are also
supported.

```python
>>> import graph_scheduler

>>> graph = {
    'A': set(),
    'B': {'A'},
    'C': {'A'},
    'D': {'B', 'C'},
}

>>> sched = graph_scheduler.Scheduler(graph=graph)
>>> sched.add_condition('C', graph_scheduler.EveryNCalls('A', 2))
>>> sched.add_condition('D', graph_scheduler.EveryNCalls('C', 2))

>>> print(list(sched.run()))
[{'A'}, {'B'}, {'A'}, {'C', 'B'}, {'A'}, {'B'}, {'A'}, {'C', 'B'}, {'D'}]
```
