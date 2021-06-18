import doctest

import pytest
import graph_scheduler


@pytest.mark.parametrize(
    "mod",
    [
        graph_scheduler.condition,
        graph_scheduler.scheduler,
        graph_scheduler.time,
    ]
)
def test_other_docs(mod, capsys):
    fail, total = doctest.testmod(mod, optionflags=doctest.REPORT_NDIFF)
    if fail > 0:
        captured = capsys.readouterr()
        pytest.fail("{} out of {} examples failed:\n{}\n{}".format(
            fail, total, captured.err, captured.out), pytrace=False)
