# ceiling-graph

Extensions to the TGN Framework proposed by Rossi et. al. <https://arxiv.org/abs/2006.10637>.

## Running One Hop Neighbourhood Aggregation experiments

### For PyTorch 1.8 and CUDA version cu101, :

```sh
$ pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
$ pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
$ pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
$ pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
$ pip install git+https://github.com/rusty1s/pytorch_geometric.git
```
If your PyTorch and/or CUDA versions are different, just replace the appropriate versions instead of 1.8.0 and cu101

### Other requirements

```
torch>=1.7
pandas==1.1.5
numpy==1.19.5
sklearn==0.22.2
```

### Then, clone the repo and follow the instructions

```
cd src
python run.py
```

or run the iPython Notebook src/run.ipynb
