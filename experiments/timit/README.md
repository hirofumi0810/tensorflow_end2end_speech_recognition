## Usage
### Training
```
cd training
./run_*.sh path_to_config_file gpu_index
```
For example,
```
cd training
./run_ctc.sh ../config/ctc/blstm_rmsprop_phone61.yml 0
```

### Restoration & Evaluation
```
cd evaluation
python eval_*.py path_to_trained_model
```

### Visualization
#### Plot LER in training
```
cd visualization
python plot_ler.py path_to_trained_model
```
#### Plot loss in training
```
cd visualization
python plot_loss.py path_to_trained_model
```
#### CTC
##### Plot CTC posteriors
```
cd visualization
python plot_(multitask_)ctc_posteriors.py path_to_trained_model
```
##### Decoding
```
cd visualization
python decode_(multitask_)ctc.py path_to_trained_model
```
#### Attention
##### Plot attention weights
```
cd visualization
python plot_attention_weights.py path_to_trained_model
```
##### Decoding
```
cd visualization
python decode_attention.py path_to_trained_model
```
#### Joint CTC-Attention
comming soon
