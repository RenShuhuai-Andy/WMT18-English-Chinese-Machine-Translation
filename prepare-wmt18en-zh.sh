#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=40000


if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

OUTDIR=data/wmt18_en_zh
src=en
tgt=zh
lang=en-zh
prep=$OUTDIR
tmp=$prep/tmp
jieba=$tmp/jieba
orig=data/dataset

mkdir -p $orig $tmp $prep $jieba

echo "pre-processing data..."
python data/data_process.py

echo "cleaning data..."
for l in $src $tgt; do
    for d in train dev test; do
        cat $orig/$d.$l | \
            perl $NORM_PUNC $l | \
            perl $REM_NON_PRINT_CHAR | \
            perl $LC | \
            sed -e "s/\&amp;/\&/g" | \
            sed -e "s/ampamp;/\&/g" | \
            sed -e "s/amp#160;//g" | \
            sed -e "s/lamp#160;//g" | \
            sed -e "s/amp#45;//g" | \
            sed -e "s/ampnbsp;//g" | \
            sed -e "s/\&nbsp;//g" | \
            sed -e "s/\&#160;//g" | \
            sed -e "s/\&#45;//g" | \
            sed -e "s/\&#124;/\|/g" | \
            sed -e "s/\&lt;/\</g" | \
            sed -e "s/amplt;/\</g" | \
            sed -e "s/\&gt;/\>/g" | \
            sed -e "s/ampgt;/\>/g" | \
            sed -e "s/\&apos;/\'/g" | \
            sed -e "s/\&quot;/\"/g" | \
            sed -e "s/ampquot;/\"/g" | \
            sed -e "s/&mdash;/-/g" | \
            sed -e "s/ \. /\./g" | \
            sed -e "s/\. /\./g" | \
            sed -e "s/ \./\./g" | \
            sed -e "s/\&#91;/\[/g" | \
            sed -e "s/\&#93;/\]/g" > $tmp/$d.$l
    done
done

echo "tokenizing zh data by jieba..."
for l in zh; do
    for d in train dev test; do
        python -m jieba -d " " $tmp/$d.$l > $jieba/$d.$l
    done
done

for d in train dev test; do
  mv $tmp/$d.en $jieba
done

echo "tokenizing data..."
for l in $src $tgt; do
    for d in train dev test; do
        cat $jieba/$d.$l | \
            perl $TOKENIZER -threads 8 -a -l $l > $tmp/$d.$l
    done
done

TRAIN=$tmp/train.en-zh
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L dev.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $tmp/bpe.$f
    done
done

perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt $prep/train 1 250
perl $CLEAN -ratio 1.5 $tmp/bpe.dev $src $tgt $prep/dev 1 250

for L in $src $tgt; do
    cp $tmp/bpe.test.$L $prep/test.$L
done