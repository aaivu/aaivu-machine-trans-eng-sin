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

Machine translation is the task of automatically converting source text in one language to text in another language. That is, when given an input which is a sequence of symbols in some language, the computer program must convert it to a sequence of symbols in another language. This challenging task is perhaps one of the most difficult chores in artificial intelligence, since there is no one single best translation owing to the natural ambiguity and flexibility of human language.

Neural networks, also known as Artificial Neural Networks (ANN) or Simulated Neural Networks (SNN), form a subset of machine learning and are at the heart of deep learning algorithms. Neural networks mimic the way in which the biological neurons in human brain signal to one another. This network is composed of an input node (artificial neuron) layer, one or more hidden node layers and an output node layer. When nodes are activated, output of one layer is passed as input to the next layer of network. This network relies on training data to learn and improve accuracy over time. 

When such neural network models are employed in machine translation, it is regarded as Neural Machine Translation, or NMT for short. Unlike the traditional phrase-based translation system which consists of many small sub-components that are tuned separately, NMT attempts to build and train a single, large neural network that reads a sentence and outputs a correct translation. It requires only one model for the translation rather than a pipeline of fine-tuned models. The strength of NMT, which achieves state of the art results, is in its ability to learn the mapping from input text to associated output text, directly in an end-to-end fashion.

This research utilizes Transformer architecture in the development of NMT system for the under-resourced, domain-specific English to Sinhala translation. Transformer is a neural network architecture that makes use of self-attention. (Given a word sequence, we recognize that some words within it are more closely related with one another than others. Accordingly, when a given word "attends to" other words in the sequence, the concept is self-attention is formulated.) The Transformer is the first 'input sequence to output sequence conversion' model that relies entirely on self-attention to compute input and output representations without using sequence aligned recurrent neural networks like LSTM, which stands for Long-Short-Term-Memory. Since, translation deals with sequence-dependent data, LSTM modules, which can give meaning to sequence by remembering (or forgetting) important (or unimportant) parts, were popularly used. However, Transformer doesn't use LSTM and it showed that a feed-forward network used with self-attention is sufficient.

In this research we actively engage in exploration of effective ways to incorporate Part-of-Speech, abbreviated as POS, so as to enhance translation quality. POS is the categorization of words of a language into classes according to their form changes and their grammatical relationships. The traditional parts of speech are verbs, nouns, pronouns, adjectives, adverbs, conjunctions and interjections.

Further, subword techniques are adopted to improve the quality of translation. NMT models typically operate with a fixed vocabulary, but translation is an open-vocabulary. Encoding rare and unknown words as sequences of Subword units comes handy in making NMT model capable of open-vocabulary translation. This is based on the intuition that various word classes are translatable via smaller units than words.

This project consists of the following.

- Transformer baseline 
- Transformer with subword segmentation 
    - Byte Pair Encoding
    
    Byte pair encoding or diagram coding is a simple form of data compression in which the most common pair of consecutive bytes of data is replaced with a byte that doesn't       occur within that data. This process could continue recursively until there are no such pairs of bytes that occur more than once. To decompress this and build original         data, these replacements are simply performed in the reverse order.
    
    - Unigram based subword regularization
    
    Though subword units are effective in alleviating open-vocabulary problems in NMT, subword segmentation is ambiguous and multiple segmentations are possible even with the     same vocabulary. Harnessing this ambiguity by regularizing the subword segmentation is necessary. Here, a unigram language model is used for regularization of these           subwords. A language model computes either the probability of a sequence of words or the probability of an upcoming word. One such language model, that is based on single     words, is the unigram model. This assumes that the probability of each word is independent of any words before it and instead, it only depends on the fraction of times         this word appears among all the words in the training text.    
    
- Transformer with Part-of-Speech (POS) 
    - Input embedding
    
    Embedding, a remarkably successful use of deep learning, is a method used to represent discrete variables as continuous vectors. Embeddings have purposes such as input to     a machine learning model for supervised task and tool for visualization of concepts and relations between categories. Word embeddings are widely used for machine               translation.
    
    - Positional encoding
    
    Position and order of words are the essential parts of any language. They define the grammar and thus the actual semantics of a sentence. The Transformer model doesn't         have any sense of position or order for each word. Still there exists the need to encompass the order of the words into this model. A possible solution is to add a piece       of information to each word about its position in the sentence. This "piece of information" is what we call as the positional encoding.
    
    
### Architecture Diagrams

Following are the architecture diagrams for the POS integration with the input embedding and positional encoding respectively.

<p align="center">
<img src="https://github.com/aaivu/aaivu-machine-trans-eng-sin/blob/master/docs/images/Architecture-diagram-pos-integration-input-embedding.jpg" width="600">
<img src="https://github.com/aaivu/aaivu-machine-trans-eng-sin/blob/master/docs/images/Architecture-diagram-pos-integration-positional-encoding.jpg" width="600">
</p>

The following instructions will guide to produce our results. 

### Requirements

We use [fairseq](https://github.com/pytorch/fairseq) for training, [sentencepiece](https://github.com/google/sentencepiece) for preprocessing & [sacrebleu](https://github.com/mjpost/sacrebleu) for producing BLEU scores.

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
Since POS is implemented within the fairseq-transformer, navigate to the project directory and install fairseq as follows.

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
