

# GSNet

## Installation

```bash
cd third_party/gsnet
```

- graspness_implementation

```bash
git clone https://github.com/rhett-chen/graspness_implementation.git
```

- Minkowski Engine

```bash
conda install openblas-devel -c anaconda
git clone https://github.com/NVIDIA/MinkowskiEngine.git # if not exist
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
cd ..
```

- knn (code adapted from [pytorch_knn_cuda](https://github.com/chrischoy/pytorch_knn_cuda))

```bash
cd knn
python setup.py install
cd ..
```

- pointnet2 operators (code adapted from [votenet](https://github.com/facebookresearch/votenet))

```bash
cd pointnet2
python setup.py install
cd ..
```

- graspnetAPI for evaluation

```bash
git clone https://github.com/graspnet/graspnetAPI.git # if not exists
cd graspnetAPI
pip install .
cd ..
```


## Checkpoint and Test Case
### Download Checkpoint and Test Case

Download from Google Drive:
through this [link](https://drive.google.com/drive/folders/1g6NCoFwtRtJLGtwSq79aPgOylW8CxVQK?usp=sharing)

### Test Case

```bash
python third_party/gsnet/gsnet.py
```
