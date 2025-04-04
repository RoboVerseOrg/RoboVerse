"""This folder provides dynamic task configuration for GAPartManip tasks.

It includes functionality to dynamically import and configure tasks
based on their names.
"""

from .gapartmanip_task_metacfg import _dynamic_import_task


def __getattr__(name):
    if name.startswith("GAPartManip") and name.endswith("Cfg"):
        task_name = name[: -len("Cfg")]
        return _dynamic_import_task(task_name)
    else:
        raise AttributeError(f"Module {__name__} has no attribute {name}")
