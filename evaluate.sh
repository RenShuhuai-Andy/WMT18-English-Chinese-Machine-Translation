model_signature=transformer_wmt_en_zh
result_dir=results/$model_signature

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fairseq-generate \
      data/data-bin-disjointed \
      --user-dir model \
      --task translation \
      --results-path $result_dir/test.out \
      --path checkpoints/$model_signature/checkpoint_best.pt \
      --lenpen 0.6 \
      --beam 4

LC_ALL=en_US.UTF-8 python extract_generate_output.py \
      --output $result_dir/test.out/generate-test \
      --srclang en \
      --tgtlang zh \
      $result_dir/test.out/generate-test.txt

perl multi-bleu.perl data/dataset/test.zh < $result_dir/test.out/generate-test.zh | tee -a $result_dir/evaluate_results.txt