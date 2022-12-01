import pytest


def pytest_addoption(parser):
    parser.addoption('--dataset', action='store', default='cifar10')
    parser.addoption('--model', action='store', default='resnet')
