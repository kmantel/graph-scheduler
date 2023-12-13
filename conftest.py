import doctest

import pytest


def pytest_runtest_setup(item):
    doctest.ELLIPSIS_MARKER = "[...]"


@pytest.helpers.register
def setify_expected_output(expected_output):
    type_set = type(set())
    for i in range(len(expected_output)):
        if type(expected_output[i]) is not type_set:
            try:
                iter(expected_output[i])
                expected_output[i] = set(expected_output[i])
            except TypeError:
                expected_output[i] = set([expected_output[i]])
    return expected_output
