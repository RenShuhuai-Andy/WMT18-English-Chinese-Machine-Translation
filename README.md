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
sh process.sh jointed
```

## Train

```
sh train.sh
```

## Evaluation

```
sh evaluate.sh
```

## Result

|                       | BLEU |
| --------------------- | ---- |
| transformer_wmt_en_zh | 14.80, 52.8/22.4/11.2/5.9 |
| transformer_wmt_en_zh_big| 19.27, 54.8/25.7/14.1/8.1 |
| transformer_wmt_en_zh_sparse_topk87| 18.40, 55.9/25.9/14.1/8.1 | 
