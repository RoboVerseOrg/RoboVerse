# Auto Testing

## Run test locally

RoboVerse uses pytest for testing. 

Since installing different simulators in the same python environment may conflict with each other, you should only run tests for one simulator at a time:
```
pytest -k ${sim}
```

For example, to test the functionality of the MuJoCo simulator, run:
```
pytest -k mujoco
```
### Adding Test cases
When you add a new feature, you should only write tests for the simulator you have actually tested.
For example, if you add a feature for MuJoCo, you can write a test in the following format and place it in a subfolder of `metasim/test`

```
@pytest.mark.parametrize("sim,num_envs", get_test_parameters())
def test_consistency(sim, num_envs):
    if sim not in ("mujoco",):
        pytest.skip(f"Skipping simulator {sim} for this test)
```

## Run test in CI

CI is automatically triggered every time a PR is ready to be merged (i.e., added to the [merge queue](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-a-merge-queue)).

To launch the CI manually, please refer to [Manually running a workflow](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow). Specifically, go to Actions tab, select the target workflow, and click the "Run workflow" button, as illustrated below.
![Run CI](images/run_ci_test.png)
