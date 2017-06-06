## How to use

At first, please choose the corpus and command
```
cd corpus
```

### Training
```
cd trainer
```
and command
```
./run_ctc.sh gpu_numebr path_to_config
```

### Restoration & Evaluation
```
cd restore
```
and command
```
python restore_ctc.py
```

### Restoration & Fine-tuning
```
fine_tuning
```
and command
```
python finetune_ctc.py
```

### Visualization
