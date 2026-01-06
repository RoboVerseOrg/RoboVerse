import os
import pathlib
import sys

import hydra
from omegaconf import OmegaConf

here = os.path.dirname(os.path.abspath(__file__))
#
# Make `import roboverse_learn...` work reliably when running this file directly.
# We need the *repo root* (parent of `roboverse_learn/`) on sys.path, not `roboverse_learn/` itself.
repo_root = os.path.dirname(os.path.dirname(here))
sys.path.insert(0, repo_root)
from roboverse_learn.il.runners.base_runner import BaseRunner

abs_config_path = str(pathlib.Path(__file__).resolve().parent.joinpath("configs").absolute())
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(config_path=abs_config_path, version_base="1.3")
def main(cfg):
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)

    runner: BaseRunner = cls(cfg)
    runner.run()


if __name__ == "__main__":
    main()
