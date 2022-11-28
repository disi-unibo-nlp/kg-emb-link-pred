# Comprehensive Analysis of Knowledge Graph Embedding Techniques Benchmarked on Link Prediction

## Install requirements
All the code has been tested with `python=3.6.10`
```
pip install -r requirements.txt
```

Install the correct pytorch and torch geometric versions (i.e., the ones compatible with your cuda device). For instance:

```
pip install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install torch-geometric

```


## Train and test R-GCN on OGB-BioKG

```
python train_ogb.py --wanb_log
```


## Train and test QuatE and DualE

```
python pykg2vec.py <model_name> <dataset_name>
```


## Install requirements for PyKeen
All the code has been tested with `python=3.8.13`

Install the correct pytorch version (i.e., the ones compatible with your cuda device). For instance:
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```

Install PyKeen, with the wandb and plotting extension:
```
pip install pykeen[wandb, plotting]
```
N.B. For other options see https://pykeen.readthedocs.io/en/stable/installation.html
N.B. Before using wandb it is necessary to login. 

For testing on ogb 
```
conda install -c conda-forge ogb
```


## Train and test TransE, RotatE, DistMult, ComplEx, ConvE, ConvKB, CompGCN and NodePiece

```
python pykeen.py <model_name> <dataset_name>
```

## License

This project is released under the CC-BY-NC-SA 4.0 license (see `LICENSE`).
