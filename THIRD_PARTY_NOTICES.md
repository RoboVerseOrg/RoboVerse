# Third-Party Notices

RoboVerse is licensed under Apache-2.0 (see `LICENSE`). It also contains, adapts, or
integrates code from the projects listed below, each under its own license. This file is
the authoritative index of that code.

**Every file in this repository that was copied or adapted from another project carries a
header naming the upstream project, its license, and what we changed — except the components
recorded under [Unresolved](#unresolved--must-be-settled-before-the-next-release) below, whose
provenance or license we could not establish.** The header format is:

```python
# Copyright (c) <year> <upstream copyright holder>
# SPDX-License-Identifier: <license>
#
# Adapted from <project> (<url>).
# Changes: <what RoboVerse changed>, or "none (vendored verbatim)".
# Full license: <path to the license text in this repo>
```

If you add code derived from another project, add that header and a row in the table
below. If you cannot name the upstream and its license, do not merge it.

## How to read the "Relationship" column

| Term | Meaning |
|---|---|
| **Vendored** | Upstream source copied into this repo (verbatim or near-verbatim). The upstream license travels with it and is reproduced in-tree. |
| **Adapted** | Upstream source used as the basis for a modified file. Same obligations as vendored, plus a statement of changes. |
| **Reimplemented** | Written against RoboVerse APIs from the upstream's *ideas* (algorithm, reward shaping, task design); no upstream source copied. Cited out of good practice. |
| **Integration** | We import the real upstream package as a dependency and only wrap it. No upstream code in this repo. |

## Components

| Path in this repo | Upstream | License | Relationship | License text |
|---|---|---|---|---|
| `roboverse_learn/rl/fast_td3/` | [FastTD3](https://github.com/younggyoseo/FastTD3) © 2025 Younggyo Seo | MIT | Vendored | `roboverse_learn/rl/fast_td3/LICENSE` |
| `examples/rl/fast_td3/` | [FastTD3](https://github.com/younggyoseo/FastTD3) © 2025 Younggyo Seo | MIT | Vendored | `examples/rl/fast_td3/LICENSE` |
| `roboverse_learn/rl/clean_rl/` | [CleanRL](https://github.com/vwxyzjn/cleanrl) © 2019 CleanRL developers | MIT | Adapted | `roboverse_learn/rl/clean_rl/LICENSE` |
| `roboverse_learn/rl/clean_rl/buffer.py` | [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) © 2019 Antonin Raffin | MIT | Adapted | `roboverse_learn/rl/clean_rl/LICENSE.stable-baselines3` |
| `roboverse_learn/il/policies/dp/` | [diffusion_policy](https://github.com/real-stanford/diffusion_policy) © 2023 Columbia AI & Robotics Lab | MIT | Vendored | `roboverse_learn/il/policies/dp/LICENSE` |
| `roboverse_learn/il/policies/dp/models/bet/libraries/mingpt/` | [minGPT](https://github.com/karpathy/minGPT) © Andrej Karpathy | MIT | Vendored | `roboverse_learn/il/policies/dp/models/bet/libraries/mingpt/LICENSE` |
| `roboverse_learn/il/policies/dp/models/bet/libraries/loss_fn.py` | [pytorch-multi-class-focal-loss](https://github.com/AdeelH/pytorch-multi-class-focal-loss) © Adeel Hassan | MIT | Adapted | `roboverse_learn/il/policies/dp/models/bet/libraries/LICENSE.focal_loss` |
| `roboverse_learn/il/policies/act/` (everything except the DETR-derived files in the next row) | [ACT](https://github.com/tonyzhaozh/act) © 2023 Tony Z. Zhao | MIT | Vendored | `roboverse_learn/il/policies/act/LICENSE` |
| `roboverse_learn/il/policies/act/detr/models/backbone.py`, `roboverse_learn/il/policies/act/detr/models/position_encoding.py`, `roboverse_learn/il/policies/act/detr/models/transformer.py`, `roboverse_learn/il/policies/act/detr/util/` | [DETR](https://github.com/facebookresearch/detr) © Facebook, Inc. (reached via ACT's fork of DETR) | Apache-2.0 | Vendored | `roboverse_learn/il/policies/act/detr/LICENSE` |
| `roboverse_learn/il/policies/act/detr/main.py`, `roboverse_learn/il/policies/act/detr/models/__init__.py` | mixed: DETR's argument parser / package init (© Facebook, Inc.) plus ACT's model-build path (© 2023 Tony Z. Zhao) | Apache-2.0 AND MIT | Adapted | `roboverse_learn/il/policies/act/detr/LICENSE`, `roboverse_learn/il/policies/act/LICENSE` |
| `roboverse_learn/il/runners/base_eval_runner.py` | [ACT](https://github.com/tonyzhaozh/act) © 2023 Tony Z. Zhao (temporal ensembling only) | MIT | Adapted | `roboverse_learn/il/policies/act/LICENSE` |
| `roboverse_learn/il/utils/robomimic_util.py` | [robomimic](https://github.com/ARISE-Initiative/robomimic) © 2021 Stanford Vision and Learning Lab / [diffusion_policy](https://github.com/real-stanford/diffusion_policy) | MIT | Adapted | `roboverse_learn/il/policies/dp/LICENSE` |
| `roboverse_learn/il/utils/pymunk_override.py` | [pymunk](https://github.com/viblo/pymunk) © 2007-2016 Victor Blomqvist, via [diffusion_policy](https://github.com/real-stanford/diffusion_policy)'s copy | MIT | Adapted | pymunk's MIT notice, reproduced verbatim at the top of `roboverse_learn/il/utils/pymunk_override.py` |
| `roboverse_learn/vla/rlds_utils/` | [rlds_dataset_builder](https://github.com/kpertsch/rlds_dataset_builder) © Karl Pertsch | MIT | Adapted | `roboverse_learn/vla/rlds_utils/LICENSE` |
| `packages/metasim/metasim/utils/configclass.py` | [Isaac Lab](https://github.com/isaac-sim/IsaacLab) © 2022-2024 The Isaac Lab Project Developers | BSD-3-Clause | Adapted | `packages/metasim/LICENSE.isaaclab` |
| `packages/metasim/metasim/utils/dict.py` | [Isaac Lab](https://github.com/isaac-sim/IsaacLab) © 2022-2024 The Isaac Lab Project Developers | BSD-3-Clause | Adapted | `packages/metasim/LICENSE.isaaclab` |
| `packages/metasim/metasim/utils/math.py` | [Isaac Lab](https://github.com/isaac-sim/IsaacLab) © 2022-2024 The Isaac Lab Project Developers | BSD-3-Clause | Adapted | `packages/metasim/LICENSE.isaaclab` |
| `packages/metasim/metasim/utils/string_util.py` | [Isaac Lab](https://github.com/isaac-sim/IsaacLab) © 2022-2024 The Isaac Lab Project Developers | BSD-3-Clause | Adapted | `packages/metasim/LICENSE.isaaclab` |
| `roboverse_learn/rl/configs/rsl_rl/algorithm.py` | [Isaac Lab](https://github.com/isaac-sim/IsaacLab) © 2022-2025 The Isaac Lab Project Developers | BSD-3-Clause | Adapted | [below](#bsd-3-clause-isaac-lab-and-rsl-rl) |
| `roboverse_pack/tasks/beyondmimic/` (most of the tree, incl. `isaaclab/mdp/`, `isaaclab/configs/`, `isaaclab/robots/g1*.py`, `metasim/`, `scripts/`) | [BeyondMimic / whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking) | MIT (see note) | Adapted | `roboverse_pack/tasks/beyondmimic/LICENSE.beyondmimic` |
| `roboverse_pack/tasks/beyondmimic/isaaclab/envs/tracking_*.py`, `.../isaaclab/robots/actuator.py`, `.../scripts/convert_urdf.py`, `.../metasim/utils/misc.py`, `.../metasim/utils/string.py`, `.../metasim/configs/cfg_randomizers.py` | [Isaac Lab](https://github.com/isaac-sim/IsaacLab) © 2022-2025 The Isaac Lab Project Developers | BSD-3-Clause | Adapted | `roboverse_pack/tasks/beyondmimic/LICENSE.isaaclab` |
| `roboverse_pack/tasks/simpler_env/_native/control/`, `roboverse_pack/tasks/simpler_env/_native/overlay.py` | [ManiSkill2-real2sim](https://github.com/simpler-env/ManiSkill2_real2sim) (the sim stack behind [SimplerEnv](https://github.com/simpler-env/SimplerEnv)) | Apache-2.0 | Vendored | `roboverse_pack/tasks/simpler_env/_native/control/LICENSE` |
| `roboverse_pack/tasks/maniskill/_native/control.py` | [ManiSkill](https://github.com/haosulab/ManiSkill) | Apache-2.0 | Adapted | `roboverse_pack/tasks/simpler_env/_native/control/LICENSE` |
| `roboverse_pack/tasks/robosuite/_osc.py`, `roboverse_pack/tasks/robosuite/native.py` | [robosuite](https://github.com/ARISE-Initiative/robosuite) © 2022 Stanford Vision and Learning Lab & UT Robot Perception and Learning Lab | MIT | Adapted (port) | `roboverse_pack/tasks/robosuite/LICENSE` |
| `roboverse_pack/tasks/mjlab/` | [mjlab](https://github.com/mujocolab/mjlab) | Apache-2.0 | Adapted (port) | `roboverse_pack/tasks/mjlab/LICENSE` |
| `roboverse_pack/tasks/mujoco_playground/`, `roboverse_pack/tasks/pick_place/` | [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) © Google DeepMind | Apache-2.0 | Reimplemented | `roboverse_pack/tasks/mujoco_playground/LICENSE` |
| `scripts/mesh_tools/mesh2obj.py` | [MuJoCo](https://github.com/google-deepmind/mujoco) © 2023 DeepMind Technologies Limited | Apache-2.0 | Adapted | `scripts/mesh_tools/LICENSE.mujoco` |
| `scripts/conversion/convert_mjcf.py`, `scripts/conversion/mjcf2usd.py`, `scripts/conversion/urdf2usd.py` | [Isaac Lab](https://github.com/isaac-sim/IsaacLab) © 2022-2025 The Isaac Lab Project Developers | BSD-3-Clause | Adapted | [below](#bsd-3-clause-isaac-lab-and-rsl-rl) |
| `third_party/gsnet/pointnet2/` | [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch), as repackaged by Facebook © Facebook, Inc. and its affiliates (the notice each file carries) | MIT | Vendored | `third_party/gsnet/LICENSE.pointnet2` |
| `third_party/gsnet/knn/` | unattributed CUDA k-NN extension bundled by the graspness upstream; carries no notice — see [Unresolved](#unresolved--must-be-settled-before-the-next-release) | unknown | Vendored | none in tree |
| `roboverse_learn/rl/rsl_rl/` | [rsl_rl](https://github.com/leggedrobotics/rsl_rl) © ETH Zurich, NVIDIA | BSD-3-Clause | Integration (imports `rsl-rl-lib`) | [below](#bsd-3-clause-isaac-lab-and-rsl-rl) |
| `roboverse_learn/rl/sb3/` | [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) © 2019 Antonin Raffin | MIT | Integration (imports `stable-baselines3`) | in-file header |
| `roboverse_learn/vla/{OpenVLA,pi0,SmolVLA}/` | respective upstreams | see each | Integration (eval scripts only) | n/a |

### Note on BeyondMimic

`roboverse_pack/tasks/beyondmimic/` is a port of
[BeyondMimic / whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking),
licensed **MIT**. Upstream's license file is spelled `LICENCE` (British spelling); it is
reproduced verbatim at `roboverse_pack/tasks/beyondmimic/LICENSE.beyondmimic`.

**Caveat — upstream's copyright line is very likely wrong.** That MIT text reads
*"Copyright (c) 2024, The Isaac Lab Project Developers."*, apparently because the project was
forked from Isaac Lab's extension template and the holder was never updated (Isaac Lab itself
is BSD-3-Clause, not MIT). We reproduce the notice exactly as published and do **not**
propagate "Isaac Lab Project Developers" as BeyondMimic's copyright holder in our file
headers, since asserting that would be worse than saying nothing. **Confirm the license and
copyright holder with the BeyondMimic authors before redistributing.**

**On the `isaaclab/` subdirectory:** despite its name it is *not* Isaac Lab source. It is
BeyondMimic's own Isaac-Lab-API variant of the tracking task (as opposed to the `metasim/`
variant), and it is attributed to BeyondMimic. Only these files inside the tree are genuinely
Isaac Lab-derived and carry the BSD-3-Clause notice: `isaaclab/envs/tracking_base_env.py`,
`isaaclab/envs/tracking_rl_env.py`, `isaaclab/robots/actuator.py` (Isaac Lab's
`DelayedPDActuator` logic, reached via BeyondMimic), `scripts/convert_urdf.py`,
`metasim/utils/misc.py`, `metasim/utils/string.py`, and `metasim/configs/cfg_randomizers.py`.

## Unresolved — must be settled before the next release

These cannot be fixed by attribution alone. They are recorded here rather than left silent.
The files named here are the only ones in the repo that hold third-party-derived code without
an attribution header; `tests/test_attribution.py` checks that this list stays in sync.

1. **`scripts/advanced/isaacgym_animate.py` — NVIDIA Isaac Gym sample (proprietary).**
   The file carries NVIDIA's header: *"Any use, reproduction, disclosure or distribution of
   this software and related documentation without an express license agreement from NVIDIA
   CORPORATION is strictly prohibited."* That is **not** an open-source license, and
   RoboVerse's Apache-2.0 `LICENSE` purports to grant downstream recipients rights over it
   that RoboVerse does not hold. Attribution does not cure this. **The file must be removed,
   rewritten from the public API without NVIDIA's sample code, or covered by an express
   license from NVIDIA.**

2. **`third_party/gsnet/` — upstream has no license.**
   The graspness implementation this vendors
   ([rhett-chen/graspness_implementation](https://github.com/rhett-chen/graspness_implementation))
   publishes **no license** (GitHub reports `NOASSERTION`). Absent a license grant, default
   copyright applies and it is **not redistributable**. The bundled `pointnet2` code is MIT
   (© Facebook, Inc. and its affiliates; the license text each file points at is now in-tree at
   `third_party/gsnet/LICENSE.pointnet2`) and is fine. The bundled `knn` extension
   (`third_party/gsnet/knn/`) carries **no copyright or license notice at all** and we could not
   verify its provenance. **Obtain a license from the graspness authors or remove the vendored
   copy.**

## Not third-party (recorded to stop the next reader re-opening the question)

`roboverse_pack/teleop/transforms.py` and `roboverse_pack/tasks/benchmark/cube_reach.py` say their
code was *"ported/migrated from BiDexBench"*. BiDexBench is a **RoboVerse-internal predecessor
project**, not an outside one, so no third-party attribution is owed. Both files now say so inline.

## Full license texts

### BSD-3-Clause (Isaac Lab, and rsl_rl)

Isaac Lab: Copyright (c) 2022-2025, The Isaac Lab Project Developers
(https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md). All rights reserved.

rsl_rl: Copyright (c) ETH Zurich; Copyright (c) NVIDIA CORPORATION & AFFILIATES. All rights
reserved.

```
SPDX-License-Identifier: BSD-3-Clause

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### MIT

Every MIT-licensed component ships its upstream license text verbatim, in-tree, at the path
given in the table above. MIT requires the copyright notice and permission notice to travel
with any copy or substantial portion of the software; those in-tree files are what satisfy
that requirement. For reference, the MIT texts live at:

| Upstream | In-tree license text |
|---|---|
| FastTD3 | `roboverse_learn/rl/fast_td3/LICENSE`, `examples/rl/fast_td3/LICENSE` |
| CleanRL | `roboverse_learn/rl/clean_rl/LICENSE` |
| stable-baselines3 | `roboverse_learn/rl/clean_rl/LICENSE.stable-baselines3` |
| diffusion_policy | `roboverse_learn/il/policies/dp/LICENSE` |
| minGPT | `roboverse_learn/il/policies/dp/models/bet/libraries/mingpt/LICENSE` |
| pytorch-multi-class-focal-loss | `roboverse_learn/il/policies/dp/models/bet/libraries/LICENSE.focal_loss` |
| robomimic | `roboverse_learn/il/utils/LICENSE.robomimic` |
| pymunk | in-file notice, `roboverse_learn/il/utils/pymunk_override.py` |
| robosuite | `roboverse_pack/tasks/robosuite/LICENSE` |
| ACT | `roboverse_learn/il/policies/act/LICENSE` |
| BeyondMimic | `roboverse_pack/tasks/beyondmimic/LICENSE.beyondmimic` |
| rlds_dataset_builder | `roboverse_learn/vla/rlds_utils/LICENSE` |
| Pointnet2_PyTorch (as repackaged by Facebook) | `third_party/gsnet/LICENSE.pointnet2` |

### Apache-2.0

The Apache-2.0 components ship the full license text at the paths given in the table above.
Per Apache-2.0 §4(b), files we modified carry a statement of changes in their header.

## Datasets and assets

Asset and dataset provenance (robot models, scenes, motion capture, demonstration
trajectories) is documented in `README.md` and the per-integration pages under
`docs/source/dataset_benchmark/`. Those carry their own licenses; follow the original terms.
