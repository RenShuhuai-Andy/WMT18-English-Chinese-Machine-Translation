# WMT18-en-zh-Machine-Translation

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
sh process.sh jointed
```

## Train

```
sh train.sh
```

## Evaluation

```
sh evaluate.sh -m transformer_wmt_en_zh -c 0 -n 10
```

## Result


|                       | BLEU |
| --------------------- | ---- |
| transformer_wmt_en_zh | 21.45, 57.9/28.5/16.1/9.5 |
| + average checkpoints| 21.78, 58.4/29.0/16.5/9.8 |
| transformer_wmt_en_zh_big | 21.80, 57.8/28.7/16.3/9.7 |
| + average checkpoints| 22.32, 58.0/29.0/16.6/10.0 |
| transformer_wmt_en_zh_sparse_topk8 | 21.43, 57.8/28.4/16.0/9.5  |
| + average checkpoints| 21.59, 58.8/29.2/16.6/9.9 |
| transformer_wmt_en_zh_sparse_topk8_big | 21.68, 57.0/28.0/15.9/9.4 |
| + average checkpoints| 22.16, 58.1/29.0/16.6/10.0 |
| transformer_wmt_en_zh_prime | 22.45, 57.6/28.8/16.5/9.9 |
| + average checkpoints| 22.94, 58.3/29.5/17.1/10.4 |


