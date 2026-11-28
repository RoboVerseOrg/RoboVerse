Any contribution that you make to this repository will
be under the Apache 2 License, as dictated by that
[license](http://www.apache.org/licenses/LICENSE-2.0.html):

~~~
5. Submission of Contributions. Unless You explicitly state otherwise,
   any Contribution intentionally submitted for inclusion in the Work
   by You to the Licensor shall be under the terms and conditions of
   this License, without any additional terms or conditions.
   Notwithstanding the above, nothing herein shall supersede or modify
   the terms of any separate license agreement you may have executed
   with Licensor regarding such Contributions.
~~~

---

Before making your first commit, please install pre-commit hooks:
```bash
sudo apt install pre-commit
pre-commit install
```

And install the [ruff vscode extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) may help you to format the code automatically.

The `.vscode/settings.json` is configured aligning with the pre-commit hooks. Whenever you save the file, it will be formatted automatically.

To migrate new tasks, please refer to the [developer guide](https://roboverse.wiki/metasim/developer_guide/new_task).

Branches, pull-request gates, versioning and the release checklist are defined in [`RELEASING.md`](./RELEASING.md); engineering rules in [`AGENTS.md`](./AGENTS.md).

## The gate is the review

Content is accepted on evidence, not on reading. A task is registered when its demos replay at 0.80 or
better on their own backend, the hold policy scores 0 and two replays agree (`python -m metasim bench health
--floor --repeat 2`); a robot lands with one gated task that uses it; a benchmark number enters by result
file (`results/`, validated against the schema). Reviewers spend their time on contracts and names, not on
whether the task runs. The walk-throughs: [evaluate a policy](docs/source/start/evaluate_a_policy.md),
[add a task](docs/source/start/add_a_task.md), [add a robot](docs/source/start/add_a_robot.md).

Every `roboverse_data/...` path you write must exist on the Hub (`python tools/assets/check_manifest.py
--hub-only`), and a module may register at most 16 task classes (variants are parameters). A task that goes
red on the nightly gate twice is unregistered by the next pull request and listed in
`roboverse_pack/tasks/RETIRED_TASKS.md` with the commit to restore it from.
