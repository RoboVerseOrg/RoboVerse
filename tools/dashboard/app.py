import itertools
import os
from dataclasses import dataclass

import tyro
import yaml
from flask import Flask, render_template, request, send_file


@dataclass
class Args:
    port: int = 5000


args = tyro.cli(Args)

app = Flask(__name__)
BASE_LOG_DIR = "logs"


class Image:
    def __init__(self, path, caption):
        self.path = path
        self.caption = caption


class TestCase:
    def __init__(self, simulator, task, robot, command_name):
        self.simulator = simulator
        self.task = task
        self.robot = robot
        self.command_name = command_name
        self.path = os.path.join(BASE_LOG_DIR, f"{task}_{robot}_{simulator}", command_name)
        self.stdout_log = os.path.join(self.path, "stdout.log")
        self.stderr_log = os.path.join(self.path, "stderr.log")
        self.command = os.path.join(self.path, "command.sh")
        self.results_dir = os.path.join(self.path, "results")
        status_file = os.path.join(self.path, "status.txt")
        self.status = open(status_file).read().strip() if os.path.exists(status_file) else "unfinished"
        self.video_path = os.path.join(self.results_dir, "video.mp4")

    @property
    def has_stdout(self):
        return os.path.exists(self.stdout_log)

    @property
    def has_stderr(self):
        return os.path.exists(self.stderr_log)

    @property
    def has_command(self):
        return os.path.exists(self.command)

    @property
    def images(self):
        imgs = []
        for i in [0, 1]:
            imgs.append(Image(os.path.join("results", self.results_dir, f"rgb_{i:04d}.png"), f"Frame {i}"))
        return imgs

    def has_image(self, image_idx):
        return os.path.exists(os.path.join(self.path, "results", f"rgb_{image_idx:04d}.png"))

    @property
    def has_video(self):
        return os.path.exists(self.video_path)


@app.route("/")
def index():
    conf = yaml.safe_load(open("conf.yml"))
    goal_conf = yaml.safe_load(open("goal.yml"))
    task_groups = conf["tasks"]  # A dict mapping group name -> list of tasks.
    robots = conf["robots"]
    simulators = conf["simulators"]
    command_names = conf["commands"]
    task_group_goals = [goal_conf[k] for k in task_groups]
    task_group_current_cnts = [len(task_groups[k]) for k in task_groups]

    # Get the selected task group from the query parameter; default to the first available group.
    selected_group = request.args.get("group")
    if selected_group not in task_groups:
        selected_group = next(iter(task_groups))

    tasks = task_groups.get(selected_group, [])

    # Generate test cases only for tasks in the selected task group.
    cases = []
    for simulator, task, robot, command_name in itertools.product(simulators, tasks, robots, command_names):
        cases.append(TestCase(simulator, task, robot, command_name))

    # Group cases by (task, robot) for each simulator.
    grouped_cases = {}
    for case in cases:
        key = (case.task, case.robot, case.simulator)
        if key not in grouped_cases:
            grouped_cases[key] = {}
        grouped_cases[key][case.command_name] = case

    return render_template(
        "dashboard.html",
        simulators=simulators,
        tasks=tasks,
        robots=robots,
        grouped_cases=grouped_cases,
        task_groups=task_groups,
        selected_group=selected_group,
        zip=zip,
        task_group_goals=task_group_goals,
        task_group_current_cnts=task_group_current_cnts,
    )


@app.route("/logs/<path:log_path>")
def show_log(log_path):
    return send_file(os.path.join("logs", log_path), mimetype="text/plain")


@app.route("/results/<path:image_path>")
def show_image(image_path):
    return send_file(image_path)


@app.route("/favicon.ico")
def favicon():
    return send_file("static/icon.png")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port, debug=True)
