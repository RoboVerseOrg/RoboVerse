import os

paths = open("paths.txt").read().split("\n")

for path in paths:
    with open(os.path.join(os.path.dirname(path), "status.txt"), "w+") as f:
        f.write("success")
