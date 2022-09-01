#
# Copyright (c) 2022
#

import pytest
import torch

import prime
from prime.proxy import _client

from tests.common import *

################################################################################
# Init server before starting tests                                            #
################################################################################

def test_initServer():
    reset_server()

    time.sleep(1)
    if not _client.check_server():
        raise Exception('Server  not running')

    export_f_output(_client)


@with_args
def test_queue(samples_d, labels_d, samples, labels):

    datapairs = [ (s,l) for s, l in zip(samples_d, labels_d) ]

    assert _client.SupplyData(datapairs) == len(samples)


################################################################################
# Kill server after all tests are completed                                    #
################################################################################

@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    def kill_server():
        try: prime.utils.kill_server()
        except: pass
    request.addfinalizer(kill_server)
