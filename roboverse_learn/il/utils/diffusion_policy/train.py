import sys

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import pathlib

import hydra
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from omegaconf import OmegaConf

import rootutils
rootutils.setup_root(__file__, pythonpath=True)

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


abs_config_path = str(pathlib.Path(__file__).resolve().parent.joinpath("diffusion_policy", "config").absolute())

@hydra.main(
    version_base=None,
    config_path=abs_config_path,
)
def main(cfg: OmegaConf):

    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)

    workspace: BaseWorkspace = cls(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
