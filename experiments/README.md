## Usage

At first, please choose the corpus and
```
cd each_corpus
```

### Training
```
cd training
```
and
```
./run_*.sh path_to_config_file gpu_index
```
For example,
```
./run_ctc.sh ../config/ctc/blstm_rmsprop_phone61.yml 0
```

### Restoration & Evaluation
```
cd evaluation
```
and
```
python eval_*.py path_to_trained_model
```

### Visualization
```
cd visualization
```
#### Plot LER in training
```
python plot_ler.py path_to_trained_model
```
#### Plot loss in training
```
python plot_loss.py path_to_trained_model
```
#### CTC
##### Plot CTC posteriors
```
python plot_ctc_posteriors.py path_to_trained_model
```
##### Decoding
```
python decode_ctc.py path_to_trained_model
```
#### Attention
comming soon

### Restoration & Fine-tuning
comming soon
