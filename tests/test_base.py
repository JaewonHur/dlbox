from prime.base import NAME

from prime.proxy import _client
from tests.common import *

def test_base():
    assert NAME == "prime"

    export_f_output(_client)
