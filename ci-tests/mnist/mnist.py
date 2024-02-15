#!/usr/bin/env python3

import os
import sys
import codecs
import torch
import numpy as np
from os.path import abspath, dirname


def get_int(b: bytes) -> int:
    return int(codecs.encode(b, 'hex'), 16)

def read_sn3_pascalvincent_tensor(path: str, strict: bool = True) -> torch.Tensor:
    SN3_PASCALVINCENT_TYPEMAP = {
        8:  torch.uint8,
        9:  torch.int8,
        11: torch.int16,
        12: torch.int32,
        13: torch.float32,
        14: torch.float64,
    }

    with open(path, 'rb') as f:
        data = f.read()

    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256

    assert 1 <= nd <= 3
    assert 8 <= ty <= 14
    torch_type = SN3_PASCALVINCENT_TYPEMAP[ty]
    s = [get_int(data[4 * (i + 1) : 4 * (i + 2)]) for i in range(nd)]

    num_bytes_per_value = torch.iinfo(torch_type).bits // 8
    needs_byte_reversal = sys.byteorder == 'little' and num_bytes_per_value > 1
    parsed = torch.frombuffer(bytearray(data), dtype=torch_type, offset=(4 * (nd + 1)))

    if needs_byte_reversal:
        parsed = parsed.flip(0)

    assert parsed.shape[0] == np.prod(s) or not strict
    return parsed.view(*s)


def read_image_file(path: str) -> torch.Tensor:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    assert x.dtype == torch.uint8
    assert x.ndimension() == 3
    return x

def read_label_file(path: str) -> torch.Tensor:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    assert x.dtype == torch.uint8
    assert x.ndimension() == 1
    return x.long()

def sample_init() -> (torch.Tensor, torch.Tensor):
    pwd = dirname(abspath(__file__))

    samples = read_image_file(f'{pwd}/train-images-idx3-ubyte')
    labels = read_label_file(f'{pwd}/train-labels-idx1-ubyte')

    return (samples, labels)

def test_init() -> (torch.Tensor, torch.Tensor):
    pwd = dirname(abspath(__file__))

    samples = read_image_file(f'{pwd}/t10k-images-idx3-ubyte')
    labels = read_label_file(f'{pwd}/t10k-labels-idx1-ubyte')

    return (samples, labels)
