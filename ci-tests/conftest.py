import pytest

def pytest_addoption(parser):
    parser.addoption('--model', action='store', default='googlenet')
