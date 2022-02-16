# ceiling-graph

Extensions to the TGN Framework proposed by Rossi et. al. <https://arxiv.org/abs/2006.10637>.

## Running One Hop Neighbourhood Aggregation experiments

Install PyTorch Geometric from [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) or follow the instructions below. 
 
### For PyTorch 1.8 and CUDA version cu101, :

```sh

$ vpip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
$ pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
$ pip install torch-geometric

```
Just replace {TORCH} and {CUDA} with the appropriate PyTorch and CUDA versions like of 1.8.0 and cu101

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
