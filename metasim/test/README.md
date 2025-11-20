### test
```
python3 -m pytest -k genesis
```

###
```
python metasim/test/test_state_consistency.py genesis 2
```

### Special Case
For `isaacgym`, to ensure that `isaacgym` is imported first, please use instead
```
python metasim/test/isaacgym_entry.py <test_folder/file> -k isaacgym
```