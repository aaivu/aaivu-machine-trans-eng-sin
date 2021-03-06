#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import shlex
import sys

import sentencepiece as spm


if __name__ == "__main__":
    spm.SentencePieceTrainer.Train(" ".join(map(shlex.quote, sys.argv[1:])))
