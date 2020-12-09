model_signature=transformer_wmt_en_zh

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fairseq-train \
    data/data-bin-disjointed --fp16 \
    --user-dir model \
    --arch $model_signature \
    --task translation \
    --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 0.0 \
    --save-dir checkpoints/$model_signature \
    --max-update 3000 --save-interval-updates 500 \
    --keep-interval-updates 40 \
    --lr 7e-4 --lr-scheduler inverse_sqrt --warmup-updates 400 \
    --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --encoder-normalize-before --decoder-normalize-before \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 4, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --tensorboard-logdir tensorboard-logdir/$model_signature