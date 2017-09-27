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
python eval_*.py --model_path path_to_trained_model (--epoch epoch_num)
```

### Visualization
#### Plot LER in training
```
cd visualization
python plot_ler.py --model_path path_to_trained_model
```

#### CTC
##### Plot CTC posteriors
```
cd visualization
python plot_(multitask_)ctc_posteriors.py --model_path path_to_trained_model (--epoch epoch_num)
```
##### Decoding
```
cd visualization
python decode_(multitask_)ctc.py --model_path path_to_trained_model (--epoch epoch_num)
```
#### Attention
##### Plot attention weights
```
cd visualization
python plot_attention_weights.py --model_path path_to_trained_model (--epoch epoch_num)
```
##### Decoding
```
cd visualization
python decode_attention.py --model_path path_to_trained_model (--epoch epoch_num)
```
#### Joint CTC-Attention
comming soon
