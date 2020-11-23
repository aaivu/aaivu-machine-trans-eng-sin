#!/bin/bash

SRC=en
TGT=si

BPESIZE=5000   # bpe vocabulary size
TRAIN_MINLEN=0


ROOT=$(dirname "$0")
SCRIPTS=$ROOT/scripts
DATA=$ROOT/data
TMP=$DATA/${SRC}_${TGT}_bpe${BPESIZE}
DATABIN=$ROOT/data-bin/${SRC}_${TGT}_bpe${BPESIZE}
mkdir -p $TMP $DATABIN

SPM_TRAIN=$SCRIPTS/spm_train.py
SPM_ENCODE=$SCRIPTS/spm_encode.py


TRAIN_SET="train-dataset/train"
VALID_SET="valid-dataset/valid"
TEST_SET="test-dataset/test"


# learn joined BPE with sentencepiece over the tokenized source and target
python $SPM_TRAIN \
  --input=$DATA/${TRAIN_SET}.$SRC,$DATA/${TRAIN_SET}.$TGT \
  --model_prefix=$DATABIN/sentencepiece.bpe \
  --vocab_size=$BPESIZE \
  --character_coverage=1.0 \
  --model_type=bpe

# encode train
python $SPM_ENCODE \
  --model $DATABIN/sentencepiece.bpe.model \
  --output_format=piece \
  --inputs $DATA/${TRAIN_SET}.$SRC $DATA/${TRAIN_SET}.$TGT \
  --outputs $TMP/train.bpe.$SRC $TMP/train.bpe.$TGT \
  --min-len $TRAIN_MINLEN 

# encode valid
python $SPM_ENCODE \
    --model $DATABIN/sentencepiece.bpe.model \
    --output_format=piece \
    --inputs $DATA/${VALID_SET}.$SRC $DATA/${VALID_SET}.$TGT \
    --outputs $TMP/valid.bpe.$SRC $TMP/valid.bpe.$TGT

# encode test
python $SPM_ENCODE \
    --model $DATABIN/sentencepiece.bpe.model \
    --output_format=piece \
    --inputs $DATA/${TEST_SET}.$SRC $DATA/${TEST_SET}.$TGT \
    --outputs $TMP/test.bpe.$SRC $TMP/test.bpe.$TGT


# binarize data
fairseq-preprocess \
  --source-lang $SRC --target-lang $TGT \
  --trainpref $TMP/train.bpe --validpref $TMP/valid.bpe --testpref $TMP/test.bpe \
  --destdir $DATABIN \
  --joined-dictionary \
  --workers 4
