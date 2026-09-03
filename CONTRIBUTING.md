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
