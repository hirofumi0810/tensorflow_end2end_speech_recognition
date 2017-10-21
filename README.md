## TensorFlow Implementation of End-to-End Speech Recognition
### Requirements
- TensorFlow >= 1.3.0
- tqdm >= 4.14.0
- python-Levenshtein >= 0.12.0
- setproctitle >= 1.1.10
- seaborn >= 0.7.1


### Corpus
#### [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1)
- Phone (39, 48, 61 phones)
- character

#### [LibriSpeech](http://www.openslr.org/12/)
- Phone (under implementation)
- Character
- Word

#### [CSJ (Corpus of Spontaneous Japanese)](http://pj.ninjal.ac.jp/corpus_center/csj/en/)
- Phone (under implementation)
- Japanese kana character (about 150 classes)
- Japanese kanji characters (about 3000 classes)

These corpuses will be added in the future.
- Switchboard
- WSJ
- [AMI](http://groups.inf.ed.ac.uk/ami/corpus/)

This repository does'nt include pre-processing and pre-processing is based on [this repo](https://github.com/hirofumi0810/asr_preprocessing).
If you want to do pre-processing, please look at this repo.


### Model
#### Encoder
- BLSTM
- LSTM
- BGRU
- GRU
- VGG-BLSTM
- VGG-LSTM
- Multi-task BLSTM
  - you can set another CTC layer to the aubitrary layer.
- Multi-task LSTM
- VGG


#### Connectionist Temporal Classification (CTC) [\[Graves+ 2006\]](http://dl.acm.org/citation.cfm?id=1143891)
- Greedy decoder
- Beam Search decoder
- Beam Search decoder w/ CharLM (under implementation)

##### Options
- Frame-stacking [\[Sak+ 2015\]](https://arxiv.org/abs/1507.06947)
- Multi-GPUs training (synchronous)
- Splicing
- Down sampling (under implementation)


#### Attention Mechanism
##### Decoder
- Greedy decoder
- Beam search decoder (under implementation)

##### Attention type
- Bahdanau's content-based attention
- Bahdanau's normed content-based attention (under implementation)
- location-based attention
- Hybrid attention
- Luong's dot attention
- Luong's scaled dot attention (under implementation)
- Luong's general attention
- Luong's concat attention
- Baidu's attention (under implementation)

###### Options
- Sharpning
- Temperature regularization in the softmax layer (Output posteriors)
- Joint CTC-Attention [\[Kim 2016\]](https://arxiv.org/abs/1609.06773.)
- Coverage (under implementation)


### Usage
Please refer to docs in each corpuse
- TIMIT
- LibriSpeech
- CSJ


### Lisense
MIT


### Contact
hiro.mhbc@gmail.com
