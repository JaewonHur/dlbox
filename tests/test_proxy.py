#
# Copyright (c) 2022
#

import pytest
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
    # TODO
    pass


################################################################################
# Kill server after all tests are completed                                    #
################################################################################

def test_KillServer():
    prime.utils.kill_server()
