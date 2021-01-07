model_signature=prime_transformer_wmt_en_zh
GPU="0,1,2,3"
attn_dynamic_cat=1
attn_dynamic_type=2
kernel_size=0
save_tag=${model_signature}_jointed_sharedio

CUDA_VISIBLE_DEVICES=$GPU fairseq-train \
    data/data-bin-jointed --share-decoder-input-output-embed \
    --user-dir model --fp16  \
    --arch $model_signature \
    --task translation \
    --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 0.0 \
    --save-dir checkpoints/${save_tag} \
    --max-update 10000 --save-interval-updates 2000  --validate-interval 3 \
    --update-freq 4 \
    --keep-interval-updates 40 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 2000 \
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --dropout 0.3 \
    --attn_dynamic_type ${attn_dynamic_type} \
    --kernel_size ${kernel_size} \
    --attn_wide_kernels [3,15]  \
    --dynamic_gate 1 \
    --attn_dynamic_cat ${attn_dynamic_cat} \
    --max-tokens 4096 \


