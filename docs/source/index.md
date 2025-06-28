# RoboVerse
![RoboVerse](./metasim/images/tea.jpg)



<p align="center">
  <a href="https://roboverseorg.github.io"><img src="https://img.shields.io/badge/project-page-brightgreen" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2504.18904"><img src="https://img.shields.io/badge/paper-preprint-red" alt="Paper"></a>
  <a href="https://roboverse.wiki"><img src="https://img.shields.io/badge/doc-page-orange" alt="Documentation"></a>
  <a href="https://github.com/RoboVerseOrg/RoboVerse/issues"><img src="https://img.shields.io/github/issues/RoboVerseOrg/RoboVerse?color=yellow" alt="Issues"></a>
  <a href="https://github.com/RoboVerseOrg/RoboVerse/discussions"><img src="https://img.shields.io/github/discussions/RoboVerseOrg/RoboVerse?color=blueviolet" alt="Discussions"></a>
  <a href="https://discord.gg/6e2CPVnAD3"><img src="https://img.shields.io/discord/1356345436927168552?logo=discord&color=blue" alt="Discord"></a>
  <a href="docs/source/_static/wechat.jpg"><img src="https://img.shields.io/badge/wechat-QR_code-green" alt="WeChat"></a>
</p>

## Interactive System Diagram
<p align="center">
  <img src="_static/Robo.png"
       usemap="#metasim-map"
       class="map-responsive"
       style="max-width:100%; height:auto;" />
  <map name="metasim-map">
    <area shape="rect"
          coords="2836,492,3309,632"
          href="metasim/user_guide/cross_sim.html"
          alt="Cross-Sim Guide"
          style="cursor: pointer;" />
    <area shape="rect"
        coords="2830,929,3313,1081"
        href="metasim/user_guide/cross_embodiment.html"
        alt="Cross Embodiment"
        style="cursor: pointer;" />
    <area shape="rect"
        coords="771,876,367,773"
        href="metasim/user_guide/real2sim.html"
        alt="Real2Sim"
        style="cursor: pointer;" />
    <area shape="rect"
        coords="780,1241,365,1139"
        href="metasim/user_guide/teleoperate_demo.html"
        alt="Real2Sim"
        style="cursor: pointer;" />
  </map>
</p>

<script src="https://cdn.jsdelivr.net/npm/image-map-resizer@1.0.10/js/imageMapResizer.min.js"></script>
<script>
  window.onload = function () {
    imageMapResize();
  };
</script>


## Acknowledgement
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

<!-- ## Table of Contents -->
```{toctree}
:hidden:
:titlesonly:

metasim/index
dataset_benchmark/index
roboverse_learn/index
API/index
FAQ/index
```
