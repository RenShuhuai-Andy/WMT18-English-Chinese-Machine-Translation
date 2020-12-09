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
sh process.sh
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
| transformer_wmt_en_zh | 14.8 |

