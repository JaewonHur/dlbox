#
# Copyright (c) 2022
#

import pytest
import re
from random import randint

import torch
from torch import Tensor

import prime
from prime.proxy import Proxy, _client
from prime.exceptions import PrimeNotSupportedError
from tests.common import *

################################################################################
# Init server before starting tests                                            #
################################################################################

def test_initServer():
    time.sleep(1)
    if not _client.check_server():
        raise Exception('Server  not running')


@with_args
def test_Proxy(samples_d, labels_d, samples, labels):

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

    # hash, bool, len should raise exception
    i = randint(0, len(samples) - 1)
    with pytest.raises(PrimeNotSupportedError,
                       match=re.escape("'Proxy' does not support hash()")):

        print(f'Shoule not be printed: {hash(samples_d)}')

    with pytest.raises(PrimeNotSupportedError,
                       match=re.escape("'Proxy' does not support bool() conversion")):

        if samples_d[i] == samples_d[i]:
            print('Should not be printed', file=sys.stderr)

    with pytest.raises(PrimeNotSupportedError,
                       match=re.escape("'Proxy' does not support __index__()")):
        print(f'Should not be printed: {len(samples_d[0:5])}')

    # getitem is correct
    i = randint(0, len(samples) - 1)
    assert torch.equal(read_val(_client, samples_d[i]), samples[i])
    assert read_val(_client, labels_d[i]) == labels[i]

    # setitem is correct
    v = randint(0, 10)
    samples[i] = v
    samples_d[i] = v
    assert torch.equal(read_val(_client, samples_d[i]), samples[i])

    # iterator is correct
    for i, v in enumerate(samples_d):
        assert torch.equal(read_val(_client, v), samples[i])


################################################################################
# Kill server after all tests are completed                                    #
################################################################################

@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    def kill_server():
        try: prime.utils.kill_server()
        except: pass
    request.addfinalizer(kill_server)
