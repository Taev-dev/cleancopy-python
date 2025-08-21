import pytest

_TEST_PHASES: dict[None | str, int] = {
    None: 0,
    'integr8': 1,
    'e2e': 2,
}


def pytest_addoption(parser):
    parser.addoption(
        '--run-e2e',
        action='store_true', default=False, help='Run end-to-end tests')
    parser.addoption(
        '--run-integr8',
        action='store_true', default=False, help='Run integration tests')


collect_ignore_glob = []


def pytest_configure(config):
    if not config.getoption('--run-e2e'):
        collect_ignore_glob.append('*.e2e.test.py')
    if not config.getoption('--run-integr8'):
        collect_ignore_glob.append('*.integr8.test.py')


def pytest_collection_modifyitems(config, items):
    # We use this to re-order items inplace so that unittests are run first,
    # then integr8, then e2e
    items.sort(key=_sort_by_test_phase)


def _sort_by_test_phase(item: pytest.Item):
    """Use this as a sorting key divide the collected tests up into
    phases, based on _TEST_PHASES. The goal here is to run the tests
    starting with the fastest phase first, and then proceed onto the
    slower phases.
    """
    test_fs_path = item.path
    if test_fs_path is None:
        return _TEST_PHASES[None]

    suffixes = {suffix.lstrip('.') for suffix in test_fs_path.suffixes}
    maybe_phase_name = suffixes.intersection(_TEST_PHASES)

    if maybe_phase_name:
        try:
            phase_name, = maybe_phase_name
        except ValueError as exc:
            exc.add_note(
                'Apparently you have a test file with multiple phases?')
            raise exc

        return _TEST_PHASES[phase_name]

    else:
        return _TEST_PHASES[None]
