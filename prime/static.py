#
# Copyright (c) 2022
#

from __future__ import annotations

import os
from typing import List

import pyt
from pyt.core.project_handler import get_directory_modules, get_modules
from pyt.core.ast_helper import generate_ast
from pyt.cfg import make_cfg
from pyt.ml_frameworks import FrameworkAdaptor, is_pytorch_lightning_training_step
from pyt.analysis.constraint_table import initialize_constraint_table
from pyt.analysis.fixed_point import analyse
from pyt.violations.violations import Violation, find_violations


def check_violations(source: str) -> List[Violation]:
    fname = '/tmp/model.py'
    with open(fname, 'w') as fd:
        fd.write(source)

    cfg_list = []
    for path in sorted([fname]):
        directory = os.path.dirname(path)
        project_modules = get_modules(directory)
        local_modules = get_directory_modules(directory)

        tree = generate_ast(path)
        cfg = make_cfg(
            tree,
            project_modules,
            local_modules,
            path
        )

        cfg_list.append(cfg)

        need_tainted = is_pytorch_lightning_training_step

        FrameworkAdaptor(
            cfg_list,
            project_modules,
            local_modules,
            need_tainted
        )

    initialize_constraint_table(cfg_list)
    analyse(cfg_list)

    violations = find_violations(cfg_list)

    os.remove(fname)

    return violations
