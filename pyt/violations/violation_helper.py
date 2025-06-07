""" This module contains nodes and helpers."""
from ..core.node_types import (
    AssignmentNode,
    IfNode
)

class SourceNode():
    def __init__(
        self,
        cfg_node,
        secondary_nodes=[]
    ):
        self.cfg_node = cfg_node
        self.secondary_nodes = secondary_nodes


class SinkNode():
    def __init__(
        self,
        cfg_node,
        rule,
        is_violation,
    ):
        self.cfg_node = cfg_node
        self.rule = rule
        self.is_violation = lambda nodes: is_violation(self, nodes)


def find_sink_nodes(cfg, lattice):
    sinks = list()
    for rule_name, rule, is_violation in sink_rules:
        sinks.extend(
            [SinkNode(s, rule_name, is_violation) for s in rule(cfg, lattice)]
        )

    return sinks


"""
Violation Rules
"""

# 1. Assignment to instance member variable
def assign_to_instance_member(cfg, lattice):
    assignment_nodes = filter_cfg_nodes(cfg, AssignmentNode)

    sinks = [node for node in assignment_nodes
             if node.left_hand_side.startswith('self')]

    return sinks


def _assign_to_instance_member(self, nodes_in_constraint):
    sources = [n.left_hand_side for n in nodes_in_constraint]

    if set(sources).intersection(
        set(self.cfg_node.right_hand_side_variables)
    ):
        return True

    return False

# 2. Assignment to global (class member) variable
# 3. Subindexing tainted variable
# 4. Branching on tainted variable


sink_rules = [
    ('Assign to instance member', assign_to_instance_member, _assign_to_instance_member),
]


def filter_cfg_nodes(cfg, cfg_node_type):

    return [node for node in cfg.nodes
            if isinstance(node, cfg_node_type)]
