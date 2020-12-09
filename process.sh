TEXT=data/dataset
DICT=$1

python data/data_process.py

case $DICT in
  disjointed)
  fairseq-preprocess \
    --source-lang en --target-lang zh \
    --trainpref $TEXT/train --validpref $TEXT/dev \
    --testpref $TEXT/test \
    --nwordssrc 32768 --nwordstgt 32768 \
    --destdir data/data-bin-disjointed \
    --workers 20
  ;;
  jointed)
  fairseq-preprocess \
    --source-lang en --target-lang zh \
    --joined-dictionary \
    --trainpref $TEXT/train --validpref $TEXT/dev \
    --testpref $TEXT/test \
    --thresholdtgt 0 --thresholdsrc 0 \
    --destdir data/data-bin-jointed \
    --workers 20
  ;;
  *)
  echo "DICT must in [jointed, disjointed]"
esac