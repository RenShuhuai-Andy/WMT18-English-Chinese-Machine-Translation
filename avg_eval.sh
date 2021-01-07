while getopts ":m:c:" opt
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
        ?)
          echo "unknown parameters"
        exit 1;;
    esac
done

OUTPUT_PATH=checkpoints/$model_signature
python average_checkpoints.py --inputs $OUTPUT_PATH \
  --num-epoch-checkpoints 10 --output $OUTPUT_PATH/avg_10.pt

result_dir=results/$model_signature

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES fairseq-generate \
      data/data-bin-jointed \
      --user-dir model \
      --task translation \
      --results-path $result_dir/avg_10 \
      --path $OUTPUT_PATH/avg_10.pt \
      --batch-size 256 \
      --remove-bpe \
      --beam 5

tail -1 $result_dir/avg_10/generate-test.txt
