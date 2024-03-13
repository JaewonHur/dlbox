import pytest


def pytest_addoption(parser):
    parser.addoption('--is_vm', action='store_true', default=False)
    parser.addoption('--baseline', action='store_true', default=False)
    parser.addoption('--dataset', action='store', default='cifar10')
    parser.addoption('--model', action='store', default='resnet')
    parser.addoption('--max_epochs', action='store', default=10)
