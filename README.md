# English to Sinhala Neural Machine Translation

![project] ![research]



- <b>Project Mentor</b>
    1. Dr Uthayasanker Thayasivam
- <b>Contributors</b>
    1. Thilakshi Fonseka
    2. Rashmini Naranpanawa
    3. Ravinga Perera

---

## Summary

This research is about developing a NMT system using Transformer architecture for the under-resourced, domain-specific English to Sinhala translation task. The translation quality is improved by exploring effective ways of incorporating Part-of-Speech (POS) information and subword techniques.

## Description

This project consists of the following.

- Transformer baseline 
- Transformer with subword segmentation 
    - Byte Pair Encoding
    - Unigram based subword regularization
- Transformer with Part-of-Speech (POS) 
    - Input embedding
    - Positional encoding
    
### Architecture Diagrams

Following are the architecture diagrams for the POS integration with the input embedding and positional encoding respectively.

<p align="center">
<img src="https://github.com/aaivu/aaivu-machine-trans-eng-sin/blob/master/docs/images/Architecture-diagram-pos-integration-input-embedding.jpg" width="600">
<img src="https://github.com/aaivu/aaivu-machine-trans-eng-sin/blob/master/docs/images/Architecture-diagram-pos-integration-positional-encoding.jpg" width="600">
</p>

The following instructions will guide to produce our results. 

### Requirements

We use [fairseq](https://github.com/pytorch/fairseq) for training, [sentencepiece](https://github.com/google/sentencepiece) for preprocessing & [sacrebleu](https://github.com/mjpost/sacrebleu) to produce BLEU scores.

**Transformer Baseline**

```
pip install fairseq sacrebleu 
```

**Transformer with subword segmentation**

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

- Navigate to `src/Transformer-baseline`. Follow the instructions given in the `README.md`.

### Train the subword segmented transformer models 

- To train the Transformer BPE model, navigate to `src/Subword-segmentation/Transformer-BPE`. Follow the instructions given in the `README.md`.
- To train the Transformer subword regularization model, navigate to `src/Subword-segmentation/Transformer-subword-regularization`. Follow the instructions given in the `README.md`.

### Train the POS integrated transformer models

- Navigate to `src/POS-implementation`. Follow the instructions given in the `README.md`.
---

### Publications

T. Fonseka, R. Naranpanawa, R. Perera and U. Thayasivam, "English to Sinhala Neural Machine Translation," 2020 International Conference on Asian Language Processing (IALP), Kuala Lumpur, Malaysia, 2020, pp. 305-309, doi: [10.1109/IALP51396.2020.9310462](https://doi.org/10.1109/IALP51396.2020.9310462).

R. Naranpanawa, R. Perera, T. Fonseka and U. Thayasivam, "Analyzing Subword Techniques to Improve English to Sinhala Neural Machine Translation," International Journal of Asian Language Processing (IJALP), vol. 30, no. 04, p. 2050017, 2020, doi: [10.1142/s2717554520500174](https://doi.org/10.1142/S2717554520500174).

### License

Apache License 2.0

### Code of Conduct

Please read our [code of conduct document here](https://github.com/aaivu/aaivu-introduction/blob/master/docs/code_of_conduct.md).

[project]: https://img.shields.io/badge/-Project-blue
[research]: https://img.shields.io/badge/-Research-yellowgreen
