"""RoboTwin → MetaSim/Sapien parity harness (scaffold).

RoboTwin (https://github.com/RoboTwin-Platform/RoboTwin) is a 50-task
dual-arm benchmark built on Sapien 3.0.0b1 + mplib 0.2.1 + curobo. It
ships its own task base class and per-task scene+reward+success modules
under `envs/<task_name>.py`, plus an asset bundle (~1.47 TB total; we
only need the 220 MB `embodiments.zip`).

This package mirrors the mjlab/maniskill harness layout:
- `inventory.py` — catalog of RoboTwin tasks we care about
- `aloha_demo.py` — sanity-check that ALOHA-AgileX loads + steps in
  MetaSim/Sapien3 and renders a rollout

Full task ports (per-task asset wiring + reward replication) are
deferred — see the report at
/home/ghr/projects/RoboVerse/reports/robotwin_integration/index.html.
"""
