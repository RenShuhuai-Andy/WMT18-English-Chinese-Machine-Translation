# WMT18-en-zh-Machine-Translation

Assignment for PKU Computational Linguistics 2020 fall.

Achieve WMT18 en-zh Machine Translation with [Vanilla Transformer](https://arxiv.org/abs/1706.03762), [Explicit Sparse Transformer](https://arxiv.org/abs/1912.11637), and [PRIME (PaRallel Intersected Multi-scale AttEntion)](https://arxiv.org/abs/1911.09483).

## Prepare environment

```
conda create -n mt python=3.6
conda activate mt
conda install pytorch torchvision cudatoolkit=10.0 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
pip install -r requirements.txt 
```

## Process data

```
bash prepare-wmt18en-zh.sh
sh process.sh joint
```
- `joint` denotes `--joined-dictionary`
## Train

```
sh train_transformer.sh
```

## Evaluation

```
sh evaluate.sh -m transformer_wmt_en_zh -c 0 -n 10
```
- `-m` denotes the model architecture.
- `-c` denotes the index of CUDA device.
- `-n` denotes the number of checkpoints for average.

## Model Architecture
|Model Architecture| (Sparse) Transformer-Base | (Sparse) Transformer-Big| Prime |
| --------------------- | ---- | ---- | ---- |
|Encoder Embedding Size |512 |512|384|
|Encoder Feed-forward Size |1024| 2048|768|
|Encoder Attention Head Size |4| 8|4|
|Encoder Layer Number |4| 6|8|
|Decoder Embedding Size |512| 512|384|
|Decoder Feed-forward Size |1024 |2048|768|
|Decoder Attention Head Size |4 |8|4|
|Decoder Layer Number |4 |6|8|

## Result


|                       | BLEU |
| --------------------- | ---- |
| transformer_wmt_en_zh | 21.56, 57.8/28.5/16.1/9.6 |
| + average checkpoints| 21.83, 57.9/28.7/16.3/9.7 |
| transformer_wmt_en_zh_big | 21.80, 57.8/28.7/16.3/9.7 |
| + average checkpoints| 22.02, 57.3/28.4/16.2/9.7 |
| transformer_wmt_en_zh_sparse_topk8 | 21.45, 58.0/28.5/16.2/9.6  |
| + average checkpoints| 21.62, 58.4/28.9/16.5/9.8 |
| transformer_wmt_en_zh_sparse_topk8_big | 21.68, 57.5/28.4/16.1/9.6 |
| + average checkpoints| 21.78, 57.4/28.4/16.1/9.6 |
| transformer_wmt_en_zh_prime | **22.37, 57.7/28.9/16.6/10.0** |
| + average checkpoints| **22.74, 57.7/29.1/16.8/10.1** |


