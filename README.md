## TensorFlow Implementation of End-to-End Speech Recognition

### Requirements
- TensorFlow 1.1.0
- tqdm 4.11.2
- python-Levenshtein 0.12.0
- setproctitle 1.1.10
- seaborn 0.7.1

### Corpus
#### TIMIT
- phone-level (39 or 48 or 61 phones)
- character-level

#### CSJ (Corpus of Spontaneous Japanese)
- phone-level
- Japanese kana character-level
- Japanese grapheme-level (including kanji characters)

These corpuses will be added in the future.
- Switchboard
- WSJ
- [LibriSpeech](http://www.openslr.org/12/)
- [AMI](http://groups.inf.ed.ac.uk/ami/corpus/)

This repository does'nt include pre-processing and pre-processing is based on [this repo](https://github.com/hirofumi0810/asr_preprocessing).
If you want to do pre-processing, please look at this repo.

### Model
#### Connectionist Temporal Classification (CTC) [graves+ 2006](http://dl.acm.org/citation.cfm?id=1143891)
- LSTM-CTC
- Bidirectional LSTM-CTC
- GRU-CTC
- Bidirectional GRU-CTC
- Multitask CTC (CTC where the second task is tha same layer as the main task)
- Hierarchical CTC

##### Options
- [projection layer](https://arxiv.org/abs/1402.1128)
- weight decay
- dropout
- gradient clipping
- activation clipping
- [frame-stacking](https://arxiv.org/abs/1507.06947)
- multitask

#### Attention Mechanism
Under implementation


### How to use
Comming soon
