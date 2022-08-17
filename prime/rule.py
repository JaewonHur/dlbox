#
# Copyright (c) 2022
#
from __future__ import annotations

class TagError(Exception):
    def __init__(self, msg: str):
        self.msg = msg

    def __str__(self) -> str:
        return self.msg
