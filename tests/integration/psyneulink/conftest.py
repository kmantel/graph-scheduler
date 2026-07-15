import pytest
from psyneulink.core import llvm as pnlvm


def pytest_runtest_setup(item):
    if 'cuda' in item.keywords and not pnlvm.ptx_enabled:
        pytest.skip('PTX engine not enabled/available')


def pytest_generate_tests(metafunc):
    if "comp_mode_no_llvm" in metafunc.fixturenames:
        # in psyneulink<v0.16.0.0, mode is public. After, it's private
        try:
            mode_llvm = pnlvm.ExecutionMode.LLVM
        except AttributeError:
            mode_llvm = pnlvm.ExecutionMode._LLVM

        modes = [m for m in get_comp_execution_modes()
                 if m.values[0] is not mode_llvm]
        metafunc.parametrize("comp_mode", modes)

    elif "comp_mode" in metafunc.fixturenames:
        metafunc.parametrize("comp_mode", get_comp_execution_modes())


def pytest_runtest_teardown(item):
    pnlvm.cleanup()


@pytest.fixture
def comp_mode_no_llvm():
    # dummy fixture to allow 'comp_mode' filtering
    pass


@pytest.helpers.register
def get_comp_execution_modes():
    modes = [
        pytest.param(pnlvm.ExecutionMode.Python),
        pytest.param(pnlvm.ExecutionMode.LLVMRun, marks=pytest.mark.llvm),
        pytest.param(pnlvm.ExecutionMode.PTXRun, marks=[pytest.mark.llvm, pytest.mark.cuda])
    ]

    # in psyneulink<v0.16.0.0, these are public. After, they're private
    priv_pub_modes = ['LLVM', 'LLVMExec']
    for mode_llvm in priv_pub_modes:
        try:
            mode_llvm = getattr(pnlvm.ExecutionMode, mode_llvm)
        except AttributeError:
            mode_llvm = getattr(pnlvm.ExecutionMode, f'_{mode_llvm}')
        modes.append(pytest.param(mode_llvm, marks=pytest.mark.llvm))

    # in psyneulink>=v0.15.2.0, PTXExec is not present
    try:
        mode_llvm = pnlvm.ExecutionMode.PTXExec
    except AttributeError:
        pass
    else:
        pytest.param(mode_llvm, marks=[pytest.mark.llvm, pytest.mark.cuda])

    return modes


@pytest.helpers.register
def composition_to_scheduler_args(composition):
    return {
        'graph': composition.graph_processing.prune_feedback_edges()[0],
        'default_execution_id': composition.default_execution_id
    }
