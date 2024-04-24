#
# Copyright (c) 2022
#

profiles = {
    'count'     : 0,
    'rpc'       : 0,
    'serialize' : 0,
    'taint'     : 0,
    'op'        : 0,
}

class Profile:
    def __init__(self):
        self.serialize = 0
        self.taint = 0
        self.op = 0