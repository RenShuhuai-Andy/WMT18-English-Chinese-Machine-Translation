TEXT=data/wmt18_en_zh
DICT=$1

case $DICT in
  disjoint)
  fairseq-preprocess \
    --source-lang en --target-lang zh \
    --trainpref $TEXT/train --validpref $TEXT/dev \
    --testpref $TEXT/test \
    --nwordssrc 32768 --nwordstgt 32768 \
    --destdir data/data-bin-disjoint \
    --workers 20
  ;;
  joint)
  fairseq-preprocess \
    --source-lang en --target-lang zh \
    --joined-dictionary \
    --trainpref $TEXT/train --validpref $TEXT/dev \
    --testpref $TEXT/test \
    --nwordssrc 32768 --nwordstgt 32768 \
    --destdir data/data-bin-joint \
    --workers 20
  ;;
  *)
  echo "DICT must in [joint, disjoint]"
esac