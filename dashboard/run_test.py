from __future__ import annotations

import itertools
import os
import shutil
import subprocess
from dataclasses import dataclass

import tyro
import yaml
from loguru import logger as log
from rich.logging import RichHandler

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

conda_envs = {
    "isaaclab": "metasim",
    "mujoco": "metasim",
    "sapien3": "metasim",
    "pybullet": "metasim",
    "genesis": "metasim",
    "isaacgym": "metasim_isaacgym",
}


@dataclass
class Args:
    run_all: bool = False
    run_failed: bool = False
    run_unfinished: bool = False

    tasks: list[str] | None = None
    task_groups: list[str] | None = None
    robots: list[str] | None = None
    sims: list[str] | None = None
    commands: list[str] | None = None

    def __post_init__(self):
        assert self.run_all or self.run_failed or self.run_unfinished, (
            "At least one of run_all, run_failed, or run_unfinished must be True"
        )


args = tyro.cli(Args)


def single(command, log_dir, result_dir):
    log.info(f'Running command: "{command}"')
    with open(os.path.join(log_dir, "command.sh"), "w") as f:
        f.write(command)
    result = subprocess.run(
        command,
        shell=True,
        stdout=open(os.path.join(log_dir, "stdout.log"), "w"),
        stderr=open(os.path.join(log_dir, "stderr.log"), "w"),
        text=True,
        check=False,
    )
    if result.returncode == 0:
        log.info("Command executed successfully")
    else:
        log.error(f"Command failed with return code {result.returncode}")
    return result.returncode


def should_run(log_dir):
    if args.run_all:
        return True
    if args.run_unfinished and (
        not os.path.exists(os.path.join(log_dir, "status.txt"))
        or open(os.path.join(log_dir, "status.txt")).read() == "unfinished"
    ):
        return True
    if (
        args.run_failed
        and os.path.exists(os.path.join(log_dir, "status.txt"))
        and open(os.path.join(log_dir, "status.txt")).read() != "0"
    ):
        return True
    return False


def main():
    with open("dashboard/conf.yml") as f:
        conf = yaml.safe_load(f)

    if args.task_groups is not None or args.tasks is not None:
        tasks = []
        if args.task_groups is not None:
            for task_group in args.task_groups:
                tasks += conf["tasks"][task_group]
        if args.tasks is not None:
            tasks += args.tasks
    else:
        tasks = []
        for task_group in conf["tasks"]:
            tasks += conf["tasks"][task_group]
    tasks = list(dict.fromkeys(tasks))  # remove duplicates and keep order
    robots = conf["robots"] if args.robots is None else args.robots
    simulators = conf["simulators"] if args.sims is None else args.sims
    command_names = conf["commands"] if args.commands is None else args.commands

    base_log_dir = "dashboard/logs"

    for task, robot, simulator, command_name in itertools.product(tasks, robots, simulators, command_names):
        log_dir = os.path.join(base_log_dir, f"{task}_{robot}_{simulator}", command_name)
        result_dir = os.path.join(log_dir, "results")
        if command_name == "minimal":
            command = f"conda run --no-capture-output -n {conda_envs[simulator]} python dashboard/minimal_test_script.py --task={task} --robot={robot} --sim={simulator} --save_dir={result_dir}"
        elif command_name == "replay_demo":
            command = f"conda run --no-capture-output -n {conda_envs[simulator]} python scripts/advanced/replay_demo.py --task={task} --robot={robot} --sim={simulator} --save-video-path={os.path.join(result_dir, 'video.mp4')} --headless --stop-on-runout"
        else:
            raise ValueError(f"Command {command_name} not found")

        if not should_run(log_dir):
            log.info(f"Skipping {log_dir} because it is already finished")
            continue

        shutil.rmtree(log_dir, ignore_errors=True)
        shutil.rmtree(result_dir, ignore_errors=True)

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)

        with open(os.path.join(log_dir, "status.txt"), "w") as f:
            f.write("unfinished")

        returncode = single(command, log_dir, result_dir)

        with open(os.path.join(log_dir, "status.txt"), "w") as f:
            f.write(f"{returncode}")


if __name__ == "__main__":
    main()
