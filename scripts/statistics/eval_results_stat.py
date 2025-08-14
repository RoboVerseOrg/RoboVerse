import glob

paths = glob.glob("tmp/*/0000.txt")


for path in sorted(paths):
    task_name = path.split("/")[-2]
    eval_paths = glob.glob(f"tmp/{task_name}/*.txt")

    success_once_count = 0
    success_end_count = 0
    total_count = 0
    for eval_path in eval_paths:
        with open(eval_path) as f:
            lines = f.readlines()

        for line in lines:
            if "SuccessOnce" in line:
                # Extract boolean value from tensor string
                success_once = "True" in line
                if success_once:
                    success_once_count += 1
            elif "SuccessEnd" in line:
                # Extract boolean value
                success_end = "True" in line
                if success_end:
                    success_end_count += 1
        total_count += 1
    task_name = task_name[:40].ljust(40)
    print(f"Task Name: {task_name}", end="\t")
    print(f"Success Once Rate: {success_once_count / total_count:.2%}", end="\t")
    print(f"Success End Rate: {success_end_count / total_count:.2%}", end=" \t")
    print(f"Total Episodes: {total_count}")
