#MODEL_FILE=checkpoints/transformer_wmt_en_zh/checkpoint_best.pt
MODEL_FILE=checkpoints/sparse_transformer_wmt_en_zh_topk8/checkpoint_best.pt
CUDA_VISIBLE_DEVICES="7" fairseq-interactive data/data-bin-disjointed --user-dir model --print-alignment  --path $MODEL_FILE --beam 1 --source-lang en --target-lang zh  
