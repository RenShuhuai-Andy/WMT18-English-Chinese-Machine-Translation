while getopts ":m:c:n:" opt
do
    case $opt in
        m)
          echo "model_signature=$OPTARG"
          model_signature=$OPTARG
        ;;
        c)
          echo "CUDA_VISIBLE_DEVICES=$OPTARG"
          CUDA_VISIBLE_DEVICES=$OPTARG
        ;;
        n)
          echo "num_epoch_checkpoints=$OPTARG"
          num_epoch_checkpoints=$OPTARG
        ;;
        ?)
          echo "unknown parameters"
        exit 1;;
    esac
done

OUTPUT_PATH=checkpoints/$model_signature
python average_checkpoints.py --inputs $OUTPUT_PATH \
  --num-epoch-checkpoints $num_epoch_checkpoints --output $OUTPUT_PATH/avg_$num_epoch_checkpoints.pt

result_dir=results/$model_signature

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES fairseq-generate \
      data/data-bin-joint \
      --user-dir model \
      --task translation \
      --results-path $result_dir/avg_$num_epoch_checkpoints \
      --path $OUTPUT_PATH/avg_$num_epoch_checkpoints.pt \
      --batch-size 256 \
      --remove-bpe \
      --beam 5

echo "evaluation on average_checkpoints $num_epoch_checkpoints:" | tee -a logs/$model_signature.log
tail -1 $result_dir/avg_$num_epoch_checkpoints/generate-test.txt | tee -a logs/$model_signature.log

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES fairseq-generate \
      data/data-bin-joint \
      --user-dir model \
      --task translation \
      --results-path $result_dir \
      --path checkpoints/$model_signature/checkpoint_best.pt \
      --batch-size 256 \
      --remove-bpe \
      --beam 5

echo "evaluation on best checkpoint:" | tee -a logs/$model_signature.log
tail -1 $result_dir/generate-test.txt | tee -a logs/$model_signature.log

# Compute BLEU score
#grep ^H $result_dir/generate-test.txt | cut -f3- > $result_dir/generate-test.txt.sys
#grep ^T $result_dir/generate-test.txt | cut -f2- > $result_dir/generate-test.txt.ref
#fairseq-score --sys $result_dir/generate-test.txt.sys --ref $result_dir/generate-test.txt.ref

#LC_ALL=en_US.UTF-8 python extract_generate_output.py \
#      --output $result_dir/test.out/generate-test \
#      --srclang en \
#      --tgtlang zh \
#      $result_dir/test.out/generate-test.txt

#perl multi-bleu.perl data/wmt18_en_zh/test.zh < $result_dir/test.out/generate-test.zh | tee -a $result_dir/evaluate_results.txt