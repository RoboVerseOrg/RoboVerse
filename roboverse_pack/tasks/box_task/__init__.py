"""Box-task replay rendering.

Registers ``box_task.replay`` — an OpenArm Wuji bimanual replay task
with a fixed front table and three pre-grasped objects (cardboard_box,
feast_soda_can, feast_scented_candle). Designed for trajectory replay
plus photoreal rendering with switchable Kujiale backgrounds.
"""

from .box_task_replay import BoxTaskReplayEnv  # noqa: F401
