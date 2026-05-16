# RoboVerse

![RoboVerse](./_static/tea.jpg)

<p align="center">
  <a href="https://roboverseorg.github.io"><img src="https://img.shields.io/badge/project-page-brightgreen" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2504.18904"><img src="https://img.shields.io/badge/paper-preprint-red" alt="Paper"></a>
  <a href="https://roboverse.wiki/roboverse/"><img src="https://img.shields.io/badge/doc-page-orange" alt="Documentation"></a>
  <a href="https://github.com/RoboVerseOrg/RoboVerse/issues"><img src="https://img.shields.io/github/issues/RoboVerseOrg/RoboVerse?color=yellow" alt="Issues"></a>
  <a href="https://github.com/RoboVerseOrg/RoboVerse/discussions"><img src="https://img.shields.io/github/discussions/RoboVerseOrg/RoboVerse?color=blueviolet" alt="Discussions"></a>
  <a href="https://discord.gg/6e2CPVnAD3"><img src="https://img.shields.io/discord/1356345436927168552?logo=discord&color=blue" alt="Discord"></a>
</p>

---

## What is RoboVerse?

**RoboVerse** is the dataset, benchmark, task-pack, asset, and learning layer of the RoboVerse ecosystem.
It builds the downstream `roboverse-py` package and depends on the standalone
[MetaSim](https://github.com/RoboVerseOrg/MetaSim) simulator framework.

Use the ecosystem landing page at <a href="/roboverse/">/roboverse/</a> for RoboVerse docs and
<a href="/metasim/">/metasim/</a> for MetaSim docs.

---

## Quick Start

::::{grid} 2
:gutter: 3

:::{grid-item-card} Dataset & Benchmark
:link: dataset_benchmark/index
:link-type: doc

Explore RoboVerse tasks, assets, robot configurations, and benchmark protocols.
:::

:::{grid-item-card} RoboVerse Learn
:link: roboverse_learn/index
:link-type: doc

Train and evaluate policies with IL, RL, and VLA workflows.
:::

:::{grid-item-card} MetaSim Installation
:link: /metasim/get_started/installation.html
:link-type: url

Install the simulator framework used by RoboVerse.
:::

:::{grid-item-card} MetaSim Quick Start
:link: /metasim/get_started/quick_start/index.html
:link-type: url

Create scenes and control robots through the standalone simulator docs.
:::

::::

---

## Documentation Overview

::::{grid} 2
:gutter: 3

:::{grid-item-card} Dataset & Benchmark
:link: dataset_benchmark/index
:link-type: doc

Explore tasks, robot configurations, object assets, scene definitions, and benchmark results.
:::

:::{grid-item-card} RoboVerse Learn
:link: roboverse_learn/index
:link-type: doc

Learning algorithms: imitation learning, reinforcement learning, and vision-language-action methods.
:::

:::{grid-item-card} MetaSim
:link: /metasim/
:link-type: url

Simulation framework documentation, concepts, simulator support, and API reference.
:::

:::{grid-item-card} Ecosystem Landing
:link: /
:link-type: url

Project overview, architecture, community links, and citation.
:::

::::

---

## RoboVerse Scope

RoboVerse focuses on the content and learning pieces that sit on top of MetaSim:

- **RoboVerse Pack**: pre-configured robots, tasks, scene assets, and package-discovery entry points
- **Dataset & Benchmark**: task inventory, asset descriptions, evaluation protocols, and benchmark results
- **RoboVerse Learn**: imitation learning, reinforcement learning, and VLA training workflows

For the core simulation architecture, state protocol, handler system, and API reference, use the
[MetaSim documentation](https://roboverse.wiki/metasim/).

---

## Community & Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/RoboVerseOrg/RoboVerse/issues)
- **GitHub Discussions**: [Ask questions and share ideas](https://github.com/RoboVerseOrg/RoboVerse/discussions)
- **Discord**: [Join our community](https://discord.gg/6e2CPVnAD3)

---

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@misc{geng2025roboverse,
      title={RoboVerse: Towards a Unified Platform, Dataset and Benchmark for Scalable and Generalizable Robot Learning}, 
      author={Haoran Geng and Feishi Wang and Songlin Wei and Yuyang Li and Bangjun Wang and Boshi An and Charlie Tianyue Cheng and Haozhe Lou and Peihao Li and Yen-Jen Wang and Yutong Liang and Dylan Goetting and Chaoyi Xu and Haozhe Chen and Yuxi Qian and Yiran Geng and Jiageng Mao and Weikang Wan and Mingtong Zhang and Jiangran Lyu and Siheng Zhao and Jiazhao Zhang and Jialiang Zhang and Chengyang Zhao and Haoran Lu and Yufei Ding and Ran Gong and Yuran Wang and Yuxuan Kuang and Ruihai Wu and Baoxiong Jia and Carlo Sferrazza and Hao Dong and Siyuan Huang and Yue Wang and Jitendra Malik and Pieter Abbeel},
      year={2025},
      eprint={2504.18904},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2504.18904}, 
}
```

```{toctree}
:caption: Dataset & Benchmark
:maxdepth: 2
:titlesonly:
:hidden:

Overview <dataset_benchmark/index>
Tasks Overview <dataset_benchmark/tasks/overview>
Task Descriptions <dataset_benchmark/tasks/descriptions>
Task Groups <dataset_benchmark/tasks/task_groups>
Robots <dataset_benchmark/dataset/robots>
Objects <dataset_benchmark/dataset/objects>
Scenes <dataset_benchmark/dataset/scenes>
Benchmark Overview <dataset_benchmark/benchmark/overview>
Benchmark Results <dataset_benchmark/benchmark/results>
Benchmark Usage <dataset_benchmark/benchmark/usage>
```

```{toctree}
:caption: Imitation Learning
:maxdepth: 2
:titlesonly:
:hidden:

Overview <roboverse_learn/index>
Diffusion Policy <roboverse_learn/imitation_learning/diffusion_policy>
ACT <roboverse_learn/imitation_learning/ACT>
OpenVLA <roboverse_learn/imitation_learning/openvla>
SmolVLA <roboverse_learn/imitation_learning/smolvla>
RDT <roboverse_learn/imitation_learning/rdt>
Octo <roboverse_learn/imitation_learning/octo>
Contributing <roboverse_learn/imitation_learning/contributing>
```

```{toctree}
:caption: Reinforcement Learning
:maxdepth: 2
:titlesonly:
:hidden:

PPO <roboverse_learn/reinforcement_learning/ppo>
FastTD3 <roboverse_learn/reinforcement_learning/fast_td3>
SAC <roboverse_learn/reinforcement_learning/sac>
TD3 <roboverse_learn/reinforcement_learning/td3>
SkillBlender <roboverse_learn/reinforcement_learning/skillblender_rl>
Humanoid <roboverse_learn/reinforcement_learning/humanoid>
```

```{toctree}
:caption: FAQ
:maxdepth: 2
:titlesonly:
:hidden:

FAQ/index
```
