#
# Copyright (c) 2022
#

import pytest
import torch
import time


from prime.proxy import Proxy, _client
from prime.profiler import profiles, clear
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

def test_latency_single():
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


def test_latency_op():
    N = 10

    samples, _ = Proxy('_SAMPLES'), Proxy('_LABELS')

    clear()
    
    x: torch.Tensor = samples[0]
    y = torch.randint(0, 16, (16, 16))

    uops = ['dim', 'abs', 'mean', 'std', 
            'argmax', 'argmin', 'ceil', 'floor']

    bops = ['add', 'sub', 'mul', 'div', 
            'matmul', 'gcd', 'ge', 'le']

    ops = uops + bops
    latencies = {}

    for o in ops:
        if o in uops:
            for _ in range(N):
                eval(f'x.{o}()')
        else:
            for _ in range(N):
                eval(f'x.{o}(y)')

        count = profiles['count']
        rpc = profiles['rpc'] / N * 1e6
        serialize = profiles['serialize'] / N * 1e6
        op = profiles['op'] / N * 1e6
        taint = profiles['taint'] / N * 1e6

        latencies[o] = (rpc, serialize, op, taint)
        clear()

    for k,v in latencies.items():
        rpc, serialize, op, taint = v

        print(f'{k}\t{rpc:.0f}\t{serialize:.0f}\t{op:.0f}\t{taint:.0f}')


@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    def _kill_server():
        try: kill_server()
        except: pass
    request.addfinalizer(_kill_server)
