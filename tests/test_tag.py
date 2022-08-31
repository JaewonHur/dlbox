#
# Copyright (c) 2022
#

import pytest
import random
import re

import torch

import prime
from prime.proxy import Proxy, _client
from prime.exceptions import PrimeNotAllowedError

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
def test_torch_tag(samples_d, labels_d, samples, labels):

    safe_tag = 'tag\(0x[-0-9a-f]{3}\.\.[0-9a-f]{2},SAFE\)'
    undef_tag = 'tag\(0x[-0-9a-f]{{3}}\.\.[0-9a-f]{{2}},UNDEF\[{0}\]\)'
    danger_tag = 'tag\(0x[-0-9a-f]{3}\.\.[0-9a-f]{2},DANGER\)'

    # Invoking torch function on TagSack is prohibited
    with pytest.raises(PrimeNotAllowedError,
                       match=".*is not allowed on TagSack"):
        m = R('torch.max')(samples_d)

    # getitem fetch correct tag
    i = random.randint(0, 9)
    assert re.match(undef_tag.format(i), read_tag(_client, samples_d[i]))

    # Invoking getitem with unsafe index raises error
    x = R('torch.equal')(samples_d[0], samples[0])
    with pytest.raises(PrimeNotAllowedError,
                       match="__getitem__ cannot use unsafe tag"):
        x = samples_d[x]

    # setitem set correct tag
    tmp = samples_d[i]
    samples_d[i] = samples[i]

    assert re.match(safe_tag, read_tag(_client, samples_d[i]))

    samples_d[i] = tmp
    assert re.match(undef_tag.format(i), read_tag(_client, samples_d[i]))

    # len returns exact value if safe
    assert len(samples_d) == len(samples)

    # invoking method on TagSack returns value if safe
    assert torch.equal(samples_d.mean(), samples.mean())
    assert torch.equal(samples_d.mean(0), samples.mean(0))
    assert torch.equal(samples_d.mean((0, 1)), samples.mean((0, 1)))

    assert torch.equal(samples_d.sum(), samples.sum())

    # sum of samples are safe
    sum_d, sum_f = samples_d[0], samples[0]
    for i, j in zip(samples_d[1:], samples[1:]):
        sum_d = sum_d + i
        sum_f = sum_f + j

    assert torch.equal(sum_d, sum_f)

    # non-fair aggregation should give danger tag
    x = samples_d[0] * 2 + samples_d[1]
    assert re.match(danger_tag, read_tag(_client, x))

    # some fair operations are permitted
    sum_d, sum_f = samples_d[0].mul(2).add(1), samples[0].mul(2).add(1)
    for i, j in zip(samples_d[1:], samples[1:]):
        sum_d = sum_d + i.mul(2).add(1)
        sum_f = sum_f + j.mul(2).add(1)

    assert torch.equal(sum_d, sum_f)

    # tagsack should also work
    for i in range(len(samples_d)):
        samples_d[i] = samples_d[i] * 2 + 1
        samples[i] = samples[i] * 2 + 1

    assert torch.equal(samples_d.sum(), samples.sum())

    # set attribute should raise error
    with pytest.raises(PrimeNotAllowedError,
                       match='setattr on Tensor is prohibited'):
        samples_d[i].x = 'x'

    # iter, contains, getitem, setitem on tagged Tensor should raise error
    with pytest.raises(PrimeNotAllowedError,
                       match='iter on tagged Tensor is prohibited'):
        for i in samples_d[0]: print(i)

    with pytest.raises(PrimeNotAllowedError,
                       match='contains on tagged Tensor is prohibited'):
        x = (0 in samples_d[0])

    with pytest.raises(PrimeNotAllowedError,
                       match='__getitem__ on tagged Tensor is prohibited'):
        x = samples_d[0][0]

    with pytest.raises(PrimeNotAllowedError,
                       match='__setitem__ on tagged Tensor is prohibited'):
        samples_d[0][0] = samples[0][1]

    with pytest.raises(PrimeNotAllowedError,
                       match='in-place operation is not allowed'):
        samples_d[0].mul_(2)


################################################################################
# Kill server after all tests are completed                                    #
################################################################################

@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    def kill_server():
        try: prime.utils.kill_server()
        except: pass
    request.addfinalizer(kill_server)
