# guess-the-link
Guess The Link: Deciphering the Best Knowledge Graph Completion Techniques in Dense Spaces [MDPI Electronics - Graph Machine Learning]

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

## Train and test TransE, RotatE, DistMult, ComplEx, ConvE, and ConvKB 

```
python pykeen.py TransE ogbbiokg
```

## Train and test QuatE DualE

```
python pykeen.py QuatE ogbbiokg
```
