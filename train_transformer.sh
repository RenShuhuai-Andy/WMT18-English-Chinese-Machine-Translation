model_signature=transformer_wmt_en_zh
GPU="0,1,2,3"
save_tag=${model_signature}

CUDA_VISIBLE_DEVICES=$GPU fairseq-train \
    data/data-bin-joint --share-all-embeddings \
    --user-dir model --fp16  \
    --arch $model_signature \
    --task translation \
    --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 0.0 \
    --save-dir checkpoints/$save_tag \
    --max-update 25000 --save-interval-updates 2000  --validate-interval 3 \
    --keep-interval-updates 40 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --dropout 0.3 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 4, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric 2>&1 | tee -a logs/$save_tag.log


