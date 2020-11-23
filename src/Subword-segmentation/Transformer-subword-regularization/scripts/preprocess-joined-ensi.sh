#!/bin/bash

SRC=en
TGT=si

VOCABSIZE=5000   # vocabulary size
TRAIN_MINLEN=0


ROOT=$(dirname "$0")
SCRIPTS=$ROOT/scripts
DATA=$ROOT/data
TMP=$DATA/${SRC}_${TGT}_unigram${VOCABSIZE}
DATABIN=$ROOT/data-bin/${SRC}_${TGT}_unigram${VOCABSIZE}
mkdir -p $TMP $DATABIN

SPM_TRAIN=$SCRIPTS/spm_train.py
SPM_ENCODE=$SCRIPTS/spm_encode.py


TRAIN_SET="train-dataset/train"
VALID_SET="valid-dataset/valid"
TEST_SET="test-dataset/test"


# joined model for unigram based subword segementation with sentencepiece over the tokenized source and target
python $SPM_TRAIN \
  --input=$DATA/${TRAIN_SET}.$SRC,$DATA/${TRAIN_SET}.$TGT \
  --model_prefix=$DATABIN/sentencepiece.unigram \
  --vocab_size=$VOCABSIZE \
  --character_coverage=1.0 \
  --model_type=unigram

# encode train
python $SPM_ENCODE \
  --model $DATABIN/sentencepiece.unigram.model \
  --output_format=piece \
  --inputs $DATA/${TRAIN_SET}.$SRC $DATA/${TRAIN_SET}.$TGT \
  --outputs $TMP/train.unigram.$SRC $TMP/train.unigram.$TGT \
  --min-len $TRAIN_MINLEN \
  --nbest_size 64 \
  --alpha 0.1 

# encode valid
python $SPM_ENCODE \
    --model $DATABIN/sentencepiece.unigram.model \
    --output_format=piece \
    --inputs $DATA/${VALID_SET}.$SRC $DATA/${VALID_SET}.$TGT \
    --outputs $TMP/valid.unigram.$SRC $TMP/valid.unigram.$TGT \
	--nbest_size 64 \
	--alpha 0.1

# encode test
python $SPM_ENCODE \
    --model $DATABIN/sentencepiece.unigram.model \
    --output_format=piece \
    --inputs $DATA/${TEST_SET}.$SRC $DATA/${TEST_SET}.$TGT \
    --outputs $TMP/test.unigram.$SRC $TMP/test.unigram.$TGT \
	--nbest_size 64 \
	--alpha 0.1


# binarize data
fairseq-preprocess \
  --source-lang $SRC --target-lang $TGT \
  --trainpref $TMP/train.unigram --validpref $TMP/valid.unigram --testpref $TMP/test.unigram \
  --destdir $DATABIN \
  --joined-dictionary \
  --workers 4
