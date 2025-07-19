from isaacgym.torch_utils import *

import phc.env.tasks.humanoid_im_getup as humanoid_im_getup
import phc.env.tasks.humanoid_im_mcp as humanoid_im_mcp


class HumanoidImMCPGetup(humanoid_im_getup.HumanoidImGetup, humanoid_im_mcp.HumanoidImMCP):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=physics_engine,
            device_type=device_type,
            device_id=device_id,
            headless=headless,
        )
        return
