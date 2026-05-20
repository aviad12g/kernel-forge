"""Task definitions and importers."""

from openkernelforge.tasks.base import KernelTask, TaskTolerance
from openkernelforge.tasks.fused_tasks import get_fused_tasks
from openkernelforge.tasks.simple_tasks import get_builtin_tasks, get_task, get_task_map

__all__ = [
    "KernelTask",
    "TaskTolerance",
    "get_builtin_tasks",
    "get_fused_tasks",
    "get_task",
    "get_task_map",
]
