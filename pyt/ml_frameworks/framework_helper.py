"""Provides helper functions that help with determining if a function needs tainted."""
import ast


def is_pytorch_lightning_training_step(ast_node):
    if ast_node.name == 'training_step':
        return True
    else:
        return False
