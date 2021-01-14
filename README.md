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

#### zgx hyper-parameters

|                       | BLEU | average checkpoints |
| --------------------- | ---- | ---- |
| transformer_wmt_en_zh |  | |
| transformer_wmt_en_zh_big|  | |
| transformer_wmt_en_zh_sparse_topk8|   | |
| transformer_wmt_en_zh_sparse_topk8_big |  |  |
| transformer_wmt_en_zh_prime|  | 21.84, 57.8/28.6/16.2/9.7 |
| transformer_wmt_en_zh_prime_big|  | |

#### rxc hyper-parameters

|                       | BLEU | average checkpoints |
| --------------------- | ---- | ---- |
| transformer_wmt_en_zh |  | |
| transformer_wmt_en_zh_big|  | |
| transformer_wmt_en_zh_sparse_topk8|   | |
| transformer_wmt_en_zh_sparse_topk8_big | 21.39, 57.0/27.9/15.7/9.3  | 21.15, 56.5/27.5/15.5/9.2 |
| transformer_wmt_en_zh_prime|  | |
| transformer_wmt_en_zh_prime_big|  | |

#### share all

|                       | BLEU | average checkpoints |
| --------------------- | ---- | ---- |
| transformer_wmt_en_zh | 21.29, 57.5/28.1/15.9/9.4 | |
| transformer_wmt_en_zh_big|  | |
| transformer_wmt_en_zh_sparse_topk8| 21.39, 57.7/28.3/16.0/9.4  | 21.59, 58.8/29.2/16.6/9.9 |
| transformer_wmt_en_zh_sparse_topk8_big(2048) | 21.45, 57.6/28.4/16.0/9.5 | 22.01, 56.5/27.8/15.8/9.5 |
| transformer_wmt_en_zh_prime| 21.68, 58.0/28.7/16.3/9.7 | 22.07, 58.5/29.2/16.7/10.0 |
| transformer_wmt_en_zh_prime_big| 21.53, 57.7/28.5/16.2/9.6  | 22.01, 56.5/27.8/15.8/9.5 |

#### share decoder input-output embedding

|                       | BLEU | average checkpoints |
| --------------------- | ---- | ---- |
| transformer_wmt_en_zh |  | |
| transformer_wmt_en_zh_big|  | |
| transformer_wmt_en_zh_sparse_topk8| 21.21, 57.9/28.4/16.0/9.5  | 21.42, 58.8/29.1/16.5/9.8 |
| transformer_wmt_en_zh_sparse_topk8_big |  | |
| transformer_wmt_en_zh_prime| 21.62, 57.8/28.5/16.2/9.6 | |
| transformer_wmt_en_zh_prime_big|  | |

#### w/o data clean & bpe & wrong evaluation

|                       | BLEU |
| --------------------- | ---- |
| transformer_wmt_en_zh | 14.80, 52.8/22.4/11.2/5.9 |
| transformer_wmt_en_zh_big| 19.27, 54.8/25.7/14.1/8.1 |
| transformer_wmt_en_zh_sparse_topk8| 18.40, 55.9/25.9/14.1/8.1 |
| transformer_wmt_en_zh_sparse_topk8_big | 18.73, 55.9/26.2/14.3/8.3 |
| transformer_wmt_en_zh_prime| 18.72, 54.8/25.6/13.9/8.0 |
| transformer_wmt_en_zh_prime_big| 18.94, 54.9/25.7/14.1/8.1 |

