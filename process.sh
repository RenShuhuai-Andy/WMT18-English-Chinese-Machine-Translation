TEXT=data/dataset

python data/data_process.py

fairseq-preprocess \
  --source-lang en --target-lang zh \
  --trainpref $TEXT/train --validpref $TEXT/dev \
  --testpref $TEXT/test \
  --nwordssrc 32768 --nwordstgt 32768 \
  --destdir data/data-bin-disjointed \
  --workers 20