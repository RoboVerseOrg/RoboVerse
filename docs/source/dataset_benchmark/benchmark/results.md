# Benchmark Results
  The following results are trained on 50 episodes and evaluated on another 50 episodes.

## In-distribution Evaluation

<table class="benchmark">
  <thead>
    <tr>
      <th rowspan="2">Task name</th>
      <th>RGB</th>
      <th colspan="3"><center>RGBD</center></th>
      <th colspan="2">PointCloud</th>
    </tr>
    <tr>
      <th>resnet18</th>
      <th>resnet18</th>
      <th>ViT</th>
      <th>MultiViT</th>
      <th>pointnet</th>
      <th>spUnet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>CloseBoxL0</td>
      <td>0.81</td>
      <td>0.91</td>
      <td>0.89</td>
      <td>0.80</td>
      <td>0.82</td>
      <td><b>0.92</b></td>
    </tr>
    <tr>
      <td>CloseBoxL1</td>
      <td>0.40</td>
      <td>0.58</td>
      <td>0.40</td>
      <td>0.42</td>
      <td>0.73</td>
      <td><b>0.88</b></td>
    </tr>
    <tr>
      <td>CloseBoxL2</td>
      <td>0.42</td>
      <td>0.30</td>
      <td>0.30</td>
      <td>0.32</td>
      <td><b>0.82</b></td>
      <td>0.62</td>
    </tr>
    <tr>
      <td>StackCubeL0</td>
      <td><b>0.91</b></td>
      <td>0.87</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td>StackCubeL1</td>
      <td><b>0.01</b></td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td>StackCubeL2</td>
      <td><b>0.01</b></td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>

## Out-of-distribution Evaluation(Zero-shot)

<table class="benchmark">
  <thead>
    <tr>
      <th rowspan="2">Task name</th>
      <th>RGB</th>
      <th colspan="3"><center>RGBD</center></th>
      <th colspan="2">PointCloud</th>
    </tr>
    <tr>
      <th>resnet18</th>
      <th>resnet18</th>
      <th>ViT</th>
      <th>MultiViT</th>
      <th>pointnet</th>
      <th>spUnet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>CloseBoxL0</td>
      <td>0.52</td>
      <td>0.72</td>
      <td>0.68</td>
      <td>0.80</td>
      <td>0.60</td>
      <td><b>0.94</b></td>
    </tr>
    <tr>
      <td>CloseBoxL1</td>
      <td>0.20</td>
      <td>0.50</td>
      <td>0.36</td>
      <td>0.34</td>
      <td>0.77</td>
      <td><b>0.88</b></td>
    </tr>
    <tr>
      <td>CloseBoxL2</td>
      <td>0.32</td>
      <td>0.38</td>
      <td>0.40</td>
      <td>0.32</td>
      <td>0.38</td>
      <td><b>0.42</b></td>
    </tr>
    <tr>
      <td>StackCubeL0</td>
      <td><b>0.29</b></td>
      <td>0.19</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td>StackCubeL1</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td>StackCubeL2</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>

  </tbody>
</table>
