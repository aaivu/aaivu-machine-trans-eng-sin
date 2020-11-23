#!/bin/bash

SRC=en
TGT=si


TRAIN_MINLEN=0

ROOT=$(dirname "$0")
SCRIPTS=$ROOT/scripts
DATA=$ROOT/data
TMP=$DATA/${SRC}_${TGT}_unigram
DATABIN=$ROOT/data-bin/${SRC}_${TGT}_unigram
mkdir -p $TMP $DATABIN

SPM_TRAIN=$SCRIPTS/spm_train.py
SPM_ENCODE=$SCRIPTS/spm_encode.py


TRAIN_SET="train-dataset/train"
VALID_SET="valid-dataset/valid"
TEST_SET="test-dataset/test"


# independent models for unigram based subword segementation with sentencepiece over the tokenized source and target
# English
python $SPM_TRAIN \
  --input=$DATA/${TRAIN_SET}.$SRC \
  --model_prefix=$DATABIN/sentencepiece.$SRC.unigram \
  --vocab_size=5000 \
  --character_coverage=1.0 \
  --model_type=unigram


#Sinhala
python $SPM_TRAIN \
  --input=$DATA/${TRAIN_SET}.$TGT \
  --model_prefix=$DATABIN/sentencepiece.$TGT.unigram \
  --vocab_size=8000 \
  --character_coverage=1.0 \
  --model_type=unigram


# encode train
python $SPM_ENCODE \
  --model $DATABIN/sentencepiece.$SRC.unigram.model \
  --output_format=piece \
  --inputs $DATA/${TRAIN_SET}.$SRC  \
  --outputs $TMP/train.unigram.$SRC  \
  --min-len $TRAIN_MINLEN \
  --nbest_size 64 \
  --alpha 0.1 

python $SPM_ENCODE \
  --model $DATABIN/sentencepiece.$TGT.unigram.model \
  --output_format=piece \
  --inputs $DATA/${TRAIN_SET}.$TGT \
  --outputs $TMP/train.unigram.$TGT \
  --min-len $TRAIN_MINLEN \
  --nbest_size 64 \
  --alpha 0.1 

# encode valid
python $SPM_ENCODE \
    --model $DATABIN/sentencepiece.$SRC.unigram.model \
    --output_format=piece \
    --inputs $DATA/${VALID_SET}.$SRC  \
    --outputs $TMP/valid.unigram.$SRC \
	--nbest_size 64 \
	--alpha 0.1 

python $SPM_ENCODE \
    --model $DATABIN/sentencepiece.$TGT.unigram.model \
    --output_format=piece \
    --inputs $DATA/${VALID_SET}.$TGT \
    --outputs $TMP/valid.unigram.$TGT \
	--nbest_size 64 \
	--alpha 0.1 

# encode test
python $SPM_ENCODE \
    --model $DATABIN/sentencepiece.$SRC.unigram.model \
    --output_format=piece \
    --inputs $DATA/${TEST_SET}.$SRC  \
    --outputs $TMP/test.unigram.$SRC \
	--nbest_size 64 \
	--alpha 0.1 

python $SPM_ENCODE \
    --model $DATABIN/sentencepiece.$TGT.unigram.model \
    --output_format=piece \
    --inputs $DATA/${TEST_SET}.$TGT \
    --outputs $TMP/test.unigram.$TGT \
	--nbest_size 64 \
	--alpha 0.1 


# binarize data
fairseq-preprocess \
  --source-lang $SRC --target-lang $TGT \
  --trainpref $TMP/train.unigram --validpref $TMP/valid.unigram --testpref $TMP/test.unigram \
  --destdir $DATABIN \
  --joined-dictionary \
  --workers 4
