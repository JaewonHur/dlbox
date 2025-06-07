from pyt import (
    core,
    cfg,
    ml_frameworks,
    analysis,
    violations
)

__all__ = [
    'core',
    'cfg',
    'ml_frameworks',
    'analysis',
    'violations'
]


def clear_context():
    core.project_handler._local_modules.clear()
    core.module_definitions.project_definitions.clear()
    analysis.constraint_table.constraint_table.clear()
