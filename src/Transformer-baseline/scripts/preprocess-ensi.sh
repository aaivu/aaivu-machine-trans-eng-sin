#!/bin/bash

SRC=en
TGT=si

ROOT=$(dirname "$0")
DATA=$ROOT/data
DATABIN=$ROOT/data-bin/${SRC}_${TGT}
mkdir -p  $DATABIN

TRAIN_SET="train-dataset/train"
VALID_SET="valid-dataset/valid"
TEST_SET="test-dataset/test"

# binarize data
fairseq-preprocess \
  --source-lang $SRC --target-lang $TGT \
  --trainpref $DATA/${TRAIN_SET} --validpref $DATA/${VALID_SET} --testpref $DATA/${TEST_SET} \
  --destdir $DATABIN \
  --joined-dictionary \
  --workers 4
