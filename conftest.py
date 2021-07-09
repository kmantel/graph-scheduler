import doctest
import pytest

from psyneulink import clear_registry, primary_registries
from psyneulink.core import llvm as pnlvm


def pytest_runtest_setup(item):
    if 'cuda' in item.keywords and not pnlvm.ptx_enabled:
            pytest.skip('PTX engine not enabled/available')

    doctest.ELLIPSIS_MARKER = "[...]"

def pytest_generate_tests(metafunc):
    mech_and_func_modes = ['Python',
                           pytest.param('LLVM', marks=pytest.mark.llvm),
                           pytest.param('PTX', marks=[pytest.mark.llvm,
                                                      pytest.mark.cuda])
                          ]

    if "func_mode" in metafunc.fixturenames:
        metafunc.parametrize("func_mode", mech_and_func_modes)

    if "mech_mode" in metafunc.fixturenames:
        metafunc.parametrize("mech_mode", mech_and_func_modes)

    if "comp_mode_no_llvm" in metafunc.fixturenames:
        modes = [m for m in get_comp_execution_modes()
                 if m.values[0] is not pnlvm.ExecutionMode.LLVM]
        metafunc.parametrize("comp_mode", modes)

    elif "comp_mode" in metafunc.fixturenames:
        metafunc.parametrize("comp_mode", get_comp_execution_modes())

    if "autodiff_mode" in metafunc.fixturenames:
        auto_modes = [pnlvm.ExecutionMode.Python,
                      pytest.param(pnlvm.ExecutionMode.LLVMRun, marks=pytest.mark.llvm)]
        metafunc.parametrize("autodiff_mode", auto_modes)


def pytest_runtest_teardown(item):
    for registry in primary_registries:
        # Clear Registry to have a stable reference for indexed suffixes of default names
        clear_registry(registry)

    pnlvm.cleanup()

@pytest.fixture
def comp_mode_no_llvm():
    # dummy fixture to allow 'comp_mode' filtering
    pass

@pytest.helpers.register
def get_comp_execution_modes():
    return [pytest.param(pnlvm.ExecutionMode.Python),
            pytest.param(pnlvm.ExecutionMode.LLVM, marks=pytest.mark.llvm),
            pytest.param(pnlvm.ExecutionMode.LLVMExec, marks=pytest.mark.llvm),
            pytest.param(pnlvm.ExecutionMode.LLVMRun, marks=pytest.mark.llvm),
            pytest.param(pnlvm.ExecutionMode.PTXExec, marks=[pytest.mark.llvm, pytest.mark.cuda]),
            pytest.param(pnlvm.ExecutionMode.PTXRun, marks=[pytest.mark.llvm,  pytest.mark.cuda])
           ]


# TODO: remove this helper when tests no longer use psyneulink
@pytest.helpers.register
def composition_to_scheduler_args(composition):
    return {
        'graph': composition.graph_processing.prune_feedback_edges()[0],
        'default_execution_id': composition.default_execution_id
    }
