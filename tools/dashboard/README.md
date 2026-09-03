# RoboVerse Dashboard

## Configuration

Tasks that are ready to test are configured in `conf.yml`. Feel free to add more tasks.

## Launch the dashboard webpage

Build the dashboard:
```
cd dashboard
python app.py
```

## Run the test

To see the usage of the script, run:
```bash
python tools/dashboard/run_test.py -h
```

For the sake of testing, you can just run two typical tasks, `stack_cube` (for multiple objects) and `close_box` (for articulated object):

```bash
python tools/dashboard/run_test.py --tasks close_box stack_cube --robots franka --run-all
```

If the testing process is accidentally interrupted, you can resume it by:
```bash
python tools/dashboard/run_test.py --tasks close_box stack_cube --robots franka --run-unfinished
```

Sometimes one or two tasks might fail due to random reasons with IsaacLab handler. You can rerun them by:
```bash
python tools/dashboard/run_test.py --tasks close_box stack_cube --robots franka --run-failed --sims isaaclab
```

## View the test results
The test results will be shown on the dashboard webpage. You may need to force refresh the page to see the latest results. Specifically, if you are using Chrome, `Ctrl+Shift+R`/`Cmd+Shift+R` is the shortcut.

All the test cases should be passed with a ✅.

Unfinished tasks will be shown with a ❓.

Failed tasks will be shown with a ❌. You can debug by viewing the log (shown by `Stdout` and `Stderr` buttons), or reproducing the task by running the command in your local terminal (shown by `Command` button).
