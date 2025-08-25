# Test

RoboVerse uses pytest for testing.

Every time you should only run the test for one single simulator:
```
pytest -k ${sim}
```

For example, to test the functionality of the isaaclab simulator, you can run:
```
pytest -k isaaclab
```
