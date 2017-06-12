## TensorFlow Implementation of End-to-End Speech Recognition
### Requirements
- TensorFlow 1.2.0rc1
- tqdm 4.14.0
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
#### Connectionist Temporal Classification (CTC) [\[Graves+ 2006\]](http://dl.acm.org/citation.cfm?id=1143891)
- LSTM-CTC
- GRU-CTC
- Bidirectional LSTM-CTC (BLSTM-CTC)
- Bidirectional GRU-CTC (BGRU-CTC)
- Multitask CTC (you can set the CTC layer in the second task to the aubitrary layer.)

##### Options
###### General technique
- weight decay
- dropout
- gradient clipping
- activation clipping
- multitask learning

###### Awesome technique
- projection layer[\[Sak+ 2014\]](https://arxiv.org/abs/1402.1128)
- frame-stacking[\[Sak+ 2015\]](https://arxiv.org/abs/1507.06947)

#### Attention Mechanism
Under implementation


### Usage
Comming soon


### Lisense
MIT


### Contact
hiro.mhbc@gmail.com
