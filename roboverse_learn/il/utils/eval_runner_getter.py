def get_runner(algo):
    if algo == "diffusion_policy":
        from roboverse_learn.il.eval_runner.dp_eval_runner import DPEvalRunner

        return DPEvalRunner
    else:
        raise NotImplementedError(
            f"algorithm {algo} currently not supported by evaluation! Check roboverse_learn/eval_runner and roboverse_learn/eval_runner/utils/common/eval_runner_getter.py"
        )
