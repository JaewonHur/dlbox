"""Module for checking dataflow based on a definitions file."""
import ast
from collections import defaultdict

from ..analysis.definition_chains import build_def_use_chain
from ..analysis.lattice import Lattice
from ..core.node_types import AssignmentNode, TaintedNode
from .violation_helper import (
    SourceNode,
    SinkNode,
    filter_cfg_nodes,
    find_sink_nodes
)

class Violation():
    def __str__(self): pass


class CtrlViolation(Violation):
    def __init__(self, node, rule):
        self.node = node
        self.rule = rule

    def __str__(self):
        return (
            'Violation: {}\n'
            'File: {}\n'
            ' > Violates in line {}, code "{}"'.format(
                self.rule,
                self.node.path,
                self.node.line_number, self.node.label
            )
        )


class DataViolation(Violation):
    def __init__(self, source, sink, rule, reassignment_nodes):
        self.source = source
        self.sink = sink
        self.rule = rule
        self.reassignment_nodes = reassignment_nodes

    def __str__(self):
        reassigned_str = _get_reassignment_str(self.reassignment_nodes)
        return (
            'Violation: {}\n'
            'File: {}\n'
            ' > Training data at line {}, source "{}":\n'
            '{}\n'
            'File: {}\n'
            ' > Violates in line {}, sink "{}"'.format(
                self.rule,
                self.source.path,
                self.source.line_number, self.source.label,
                reassigned_str,
                self.sink.path,
                self.sink.line_number, self.sink.label
            )
        )


def _get_reassignment_str(reassignment_nodes):
    reassignments = ''
    if reassignment_nodes:
        reassignments += '   Reassigned in:\n      '
        reassignments += '\n      '.join([
            f'File: {node.path}\n       > Line {node.line_number}: {node.label}'
            for node in reassignment_nodes
        ])

    return reassignments


def find_violations(cfg_list):
    """Find violations in a list of CFGs

    Args:
        cfg_list(list[CFG]): the list of CFGs to scan.
    Returns:
        A list of violations
    """
    violations = list()

    for cfg in cfg_list:
        find_ctrl_violations(
            cfg,
            violations
        )

        find_data_violations(
            cfg,
            Lattice(cfg.nodes),
            violations
        )

    return violations

def find_ctrl_violations(cfg, violations_list):
    """Find control flow divergence in a cfg.

    Args:
        cfg(CFG): The CFG to find violations in.
        violations_list(list): That we append to when we find violations.
    """

    for i, n in enumerate(cfg.nodes):
        if i > 0 and len(n.outgoing) > 1:
            v = CtrlViolation(n, 'Control flow divergence')
            violations_list.append(v)

        elif i > 0 and n.has_ifexp:
            v = CtrlViolation(n, 'Conditional assignment')
            violations_list.append(v)


def find_data_violations(cfg, lattice, violations_list):
    """Find violations in a cfg.

    Args:
        cfg(CFG): The CFG to find violations in.
        lattice(Lattice): The lattice we're analysing.
        violations_list(list): That we append to when we find violations.
    """

    sources = identify_sources(cfg, lattice)
    sinks = identify_sinks(cfg, lattice)

    for sink in sinks:
        for source in sources:
            violation = get_violation(
                source,
                sink,
                lattice,
                cfg
            )
            if violation:
                violations_list.append(violation)


def identify_sources(cfg, lattice):
    """Identify sources in a CFG.

    Args:
        cfg(CFG): CFG to find sources in.
        lattice(Lattice): The lattice we're analysing.

    Returns:
        List of SourceNode.
    """
    assignment_nodes = filter_cfg_nodes(cfg, AssignmentNode)
    tainted_nodes = filter_cfg_nodes(cfg, TaintedNode)

    sources = [SourceNode(n) for n in tainted_nodes]
    find_secondary_sources(assignment_nodes, sources, lattice)

    return sources

def identify_sinks(cfg, lattice):
    """Identify sinks in a CFG.

    Args:
        cfg(CFG): CFG to find sinks in.
        lattice(Lattice): The lattice we're analysing.

    Returns:
        List of SinkNode.
    """

    sinks = find_sink_nodes(cfg, lattice)

    return sinks


def find_secondary_sources(assignment_nodes, sources, lattice):
    """Sets the secondary_nodes attribute of each source in the sources list.

    Args:
        assignment_nodes(list[AssignmentNode])
        sources(list[SourceNode])
        lattice(Lattice)
    """
    for source in sources:
        source.secondary_nodes = find_assignments(assignment_nodes, source, lattice)


def find_assignments(assignment_nodes, source, lattice):
    old = list()
    # propagate reassignments of the source node
    new = [source.cfg_node]

    while new != old:
        update_assignments(new, assignment_nodes, source.cfg_node, lattice)
        old = new

    # remove source node from result
    del new[0]

    return new


def update_assignments(assignment_list, assignment_nodes, source, lattice):
    for node in assignment_nodes:
        for other in assignment_list:
            if node not in assignment_list and lattice.in_constraint(other, node):
                append_node_if_reassigned(assignment_list, other, node)


def append_node_if_reassigned(assignment_list, secondary, node):
    if (
        secondary.left_hand_side in node.right_hand_side_variables or
        secondary.left_hand_side == node.left_hand_side # TODO: why?
    ):
        assignment_list.append(node)


def get_violation(source, sink, lattice, cfg):
    """Get violation between source and sink if it exists.

    Args:

    Returns:
        A Violation if it exists, else None
    """
    nodes_in_constraint = [
        secondary
        for secondary in reversed(source.secondary_nodes)
        if lattice.in_constraint(
                secondary,
                sink.cfg_node
        )
    ]
    # TODO: Why don't check source.cfg_node?
    nodes_in_constraint.append(source.cfg_node)

    if sink.is_violation(nodes_in_constraint):
        violation_deets = {
            'source': source.cfg_node,
            'sink': sink.cfg_node,
            'rule': sink.rule
        }

        def_use = build_def_use_chain(cfg.nodes, lattice)

        for chain in get_violation_chains(
            source.cfg_node,
            sink.cfg_node,
            def_use
        ):
            violation_deets['reassignment_nodes'] = chain

            return DataViolation(**violation_deets)


def get_violation_chains(current_node, sink, def_use, chain=[]):
    """Traverses the def-use graph to find all paths from source to sink that cause the violation

    """
    for use in def_use[current_node]:
        if use == sink:
            yield chain
        else:
            viol_chain = list(chain)
            viol_chain.append(use)
            yield from get_violation_chains(use, sink, def_use, viol_chain)


# TODO: handle blackbox mapping
# def how_violate(chain, violation_deets):
#     """Iterates through the chain of nodes and checks."""

#     for i, current_node in enumerate(chain):
