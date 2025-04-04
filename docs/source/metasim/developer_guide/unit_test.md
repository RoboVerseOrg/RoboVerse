# Unit testing
For each and every feature newly supported, we highly encourage every developer to write a corresponding unit test for it. The test cases are located at `metasim/test/`, where you can already find some test case samples and follow the format to add your own test cases.

```python
# Test case to detect runtime errors
class SimulatorRuntimeErrors(unittest.TestCase):

    testing_task = "close_box"
    testing_robot = "franka"

    def test_pybullet(self):

        do_all_tests(self.testing_task, "pybullet", self.testing_robot)

    def test_mujoco(self):

        do_all_tests(self.testing_task, "mujoco", self.testing_robot)
```

To run the test cases, simply `python -m metasim.test.XXX_test` . The testing result will be printed after all test cases are finished.

When adding new features, please make sure the code modification doesn't interfere with old features (old cases should pass), and the new feature should be correct (passes the new test case) .

For more testing results, see [RoboVerse Dashboard](https://roboverse-dashboard.robofisher.xyz/)
