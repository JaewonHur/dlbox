#
# Copyright (c) 2022
#

import pytest
import time

from prime.proxy import Proxy, _client
from prime.profiler import profiles
from prime.utils import run_server, kill_server

################################################################################
# Init server before starting tests                                            #
################################################################################

def test_init_server():
    kill_server()

    run_server(ll='ERROR')
    time.sleep(1)

    if not _client.check_server():
        raise Exception('Server not running')

################################################################################

def test_micro():
    samples, labels = Proxy('_SAMPLES'), Proxy('_LABELS')

    x = samples[0]
    y = samples[1]
    
    for _ in range(10):
        x = x.matmul(y)

    count = profiles['count']
    rpc = profiles['rpc'] / count * 1e6
    serialize = profiles['serialize'] / count * 1e6
    op = profiles['op'] / count * 1e6
    taint = profiles['taint'] / count * 1e6

    print('rpc\tserialize\top\ttaint')
    print(f'{rpc}\t{serialize}\t{op}\t{taint}')


@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    def _kill_server():
        try: kill_server()
        except: pass
    request.addfinalizer(_kill_server)
