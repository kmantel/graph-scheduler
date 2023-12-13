import psyneulink as pnl
import pytest


@pytest.mark.psyneulink
class TestTime:
    def test_multiple_runs(self):
        t1 = pnl.TransferMechanism()
        t2 = pnl.TransferMechanism()

        C = pnl.Composition(pathways=[t1, t2])

        C.run(inputs={t1: [[1.0], [2.0], [3.0]]})
        assert C.scheduler.get_clock(C).time == pnl.Time(run=1, trial=0, pass_=0, time_step=0)

        C.run(inputs={t1: [[4.0], [5.0], [6.0]]})
        assert C.scheduler.get_clock(C).time == pnl.Time(run=2, trial=0, pass_=0, time_step=0)
