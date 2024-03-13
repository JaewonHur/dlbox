import pytest


def pytest_addoption(parser):
    parser.addoption('--baseline', action='store_true', default=False)