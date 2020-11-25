# English to Sinhala Neural Machine Translation

![project] ![research]



- <b>Project Lead(s) / Mentor(s)</b>
    1. Name (talk forum profile link)
    2. Name (talk forum profile link)
- <b>Contributor(s)</b>
    1. Name (talk forum profile link)
    2. Name (talk forum profile link)

<b>Usefull Links </b>

- GitHub : <project_url>
- Talk Forum : <talk_forum_link>

---

## Summary

This research is about developing a NMT system using Transformer architecture for the under-resourced,  domain-specific English to Sinhala translation task. The translation quality is improved by exploring effective ways of incorporating Part-of-Speech (POS) information and subword techniques.

## Description

This project consists the following

- Transformer baseline 
- Transformer with subword segmentation 
    - Byte Pair Encoding
    - Unigram
- Transformer with Part of Speech (POS) 
    - Input embedding
    - Positional encoding

The following instructions will guide to produce our results. 

### Requirements

We use [fairseq](https://github.com/pytorch/fairseq) for training, [sentencepiece](https://github.com/google/sentencepiece) for preprocessing & [sacrebleu](https://github.com/mjpost/sacrebleu) to produce BLEU scores.

**Transformer Baseline**

```
pip install fairseq sacrebleu 
```

**Transformer with BPE**

```
pip install fairseq sacrebleu sentencepiece

```
**Transformer with Unigram**

```
pip install fairseq sacrebleu sentencepiece

```

**Transformer with POS**

```
pip install sacrebleu sentencepiece

```
Since POS is implemented withing the fairseq-transformer, navigate to the project directory and install fairseq as following

```
pip install --editable ./

```

### Train the baseline transformer model 

Navigate to `src/Transformer-baseline`. Follow the instructions given in the `Readme.md`. 




- Project phases
- Diagrams
- Approches

## More references

1. Reference
2. Link

---

### License

Apache License 2.0

### Code of Conduct

Please read our [code of conduct document here](https://github.com/aaivu/aaivu-introduction/blob/master/docs/code_of_conduct.md).

[project]: https://img.shields.io/badge/-Project-blue
[research]: https://img.shields.io/badge/-Research-yellowgreen
