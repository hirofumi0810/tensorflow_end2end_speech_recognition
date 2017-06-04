# TensorFlow Implementation of End-to-End Speech Recognition

## Requirements
- TensorFlow 1.1.0
- tqdm 4.11.2
- python-Levenshtein 0.12.0
- setproctitle 1.1.10
- seaborn 0.7.1

## Corpus
### TIMIT
- phone-level
- character-level

### CSJ
- phone-level
- Japanese kana character-level (not including kanji characters)
- Japanese kana character-level (including kanji characters)

These corpuses will be added in the future.
- Switchboard
- WSJ
- LibriSpeech
- AMI

This repository does'nt include pre-processing and pre-processing is based on [this repo](https://github.com/hirofumi0810/asr_preprocessing).
If you want to do pre-processing, please look at this repo.

## Model
### Connectionist Temporal Classification (CTC)
- LSTM-CTC
- Bidirectional LSTM-CTC
- GRU-CTC
- Bidirectional GRU-CTC

### Attention Mechanism
Under implementation


## How to use
