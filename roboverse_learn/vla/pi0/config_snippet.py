# Configuration snippet to add to openpi/src/openpi/training/config.py
# This file is used by train_pi0.sh to automatically register RoboVerse data config

import dataclasses
import pathlib
from typing import override

from openpi.training import config as _config
from openpi.training import model as _model
from openpi.training import transforms as _transforms
from openpi.policies import roboverse_policy


@dataclasses.dataclass(frozen=True)
class LeRobotRoboVerseDataConfig(_config.DataConfigFactory):
    """Data configuration for RoboVerse demonstrations in LeRobot format."""

    extra_delta_transform: bool = True

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> _config.DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        # "observation/wrist_image": "wrist_image",  # RoboVerse has a single view now
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[roboverse_policy.RoboVerseInputs(model_type=model_config.model_type)],
            outputs=[roboverse_policy.RoboVerseOutputs()],
        )

        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(-2, 7)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = _config.ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
