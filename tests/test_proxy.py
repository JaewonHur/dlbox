#
# Copyright (c) 2022
#

import pytest
import torch
from torch import Tensor

import prime
from prime.proxy import Proxy, _client
from tests.common import *

samples_d = Proxy('_SAMPLES')
labels_d = Proxy('_LABELS')

################################################################################
# Init server before starting tests                                            #
################################################################################

def test_initServer():
    time.sleep(1)
    if not _client.check_server():
        raise Exception('Server  not running')

    export_f_output(_client)


def test_Proxy():
    samples, labels = sample_init()

    # Check attribute access is correct
    for a in [ i for i in dir(samples) if not i.startswith('_') ]:
        try:
            attr = getattr(samples, a)
        except:
            with pytest.raises(RuntimeError):
                getattr(samples_d, a)
            continue

        if not hasattr(attr, '__name__'): continue

        # Normally, get __name__ of Proxy should be avoided
        assert read_val(_client, getattr(samples_d, a).__name__) == a



################################################################################
# Kill server after all tests are completed                                    #
################################################################################

@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    def kill_server():
        try: prime.utils.kill_server()
        except: pass
    request.addfinalizer(kill_server)
