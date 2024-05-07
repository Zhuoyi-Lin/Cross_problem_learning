# Paper Name 
Cross-Problem Learning for Solving Vehicle Routing Problems
## Paper


```
``` 

## Dependencies

* Python>=3.8
* NumPy
* SciPy
* [PyTorch](http://pytorch.org/)>=1.7
* tqdm
* [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)
* Matplotlib (optional, only for plotting)

## Usage

### Main differences to Attention-learn-to-solve routing problems
We typically get 3 additional options when run training and evaluating, that is: "finetune_ways","rank","activation_func"
The "finetune_ways" is to set the training ways, 
if set to "normal", it's for full-finetuning and from-scratch in paper;
if set to "inside_tuning", you should use with "activation_func" to select the type of activations in adapters;
if set to "lora", you should use with "rank" to set the rank for LoRA module;
if set to "side_tuning", if's for side-tuning in paper.


### Generating data

Training data is generated on the fly. To generate validation and test data (same as used in the paper) for all problems:
```bash
python generate_data.py --problem all --name validation --seed 4321
python generate_data.py --problem all --name test --seed 1234
```

### Training

For training OP instances with 20 nodes and using rollout as REINFORCE baseline and using the generated validation set.
and load weight from TSP 20 pretrain model and do the full fine-tuning:
```bash
python run.py --graph_size 20 --baseline rollout --data_distribution const --run_name 'op20_rollout_full_finetuning' --val_dataset data/op/op_const20_validation_seed4321.pkl --finetune_ways normal --load_path pretrain_checkpoints/tsp20/tsp20_pretrain/epoch-99.pt 
```

For training OP instances with 20 nodes and load weight from TSP 20 pretrain model and do the lora fine-tuning:
```bash
python run.py --graph_size 20 --baseline rollout --data_distribution const --run_name 'op20_rollout_lora' --val_dataset data/op/op_const20_validation_seed4321.pkl --finetune_ways lora --rank 2 --load_path pretrain_checkpoints/tsp20/tsp20_pretrain/epoch-99.pt 
```

For training OP instances with 20 nodes and load weight from TSP 20 pretrain model and do the side-tuning:
```bash
python run.py --graph_size 20 --baseline rollout --data_distribution const --run_name 'op20_rollout_side_tuning' --val_dataset data/op/op_const20_validation_seed4321.pkl --finetune_ways side_tuning --load_path pretrain_checkpoints/tsp20/tsp20_pretrain/epoch-99.pt 
```

For training OP instances with 20 nodes and load weight from TSP 20 pretrain model and do the inside-tuning with leakyrelu activation:
```bash
python run.py --graph_size 20 --baseline rollout --data_distribution const --run_name 'op20_rollout_inside_tuning_leakyrelu' --val_dataset data/op/op_const20_validation_seed4321.pkl --finetune_ways inside_tuning --activation_func leakyrelu --load_path pretrain_checkpoints/tsp20/tsp20_pretrain/epoch-99.pt 
```

For others settings, you could change the parameters accordingly.


### Evaluation
To evaluate a model, you can use `eval.py`, which will additionally measure timing and save the results
Note that, you need to add the additional parameters like training, to specify the model type:
```bash
python eval.py data/op/op_const20_test_seed1234.pkl --model pretrain_checkpoints/op20/op_full_finetuning --finetune_ways normal --epochs 99  --decode_strategy greedy
```

#### Sampling
To report the best of 1280 sampled solutions, use
```bash
python eval.py data/op/op_const20_test_seed1234.pkl --model pretrain_checkpoints/op20/op_full_finetuning --finetune_ways normal --epochs 99 --decode_strategy sample --width 1280 --eval_batch_size 1
```

#### To run baselines
Baselines for different problems are within the corresponding folders and can be ran (on multiple datasets at once) as follows
```bash
python -m problems.tsp.tsp_baseline farthest_insertion data/tsp/tsp20_test_seed1234.pkl data/tsp/tsp50_test_seed1234.pkl data/tsp/tsp100_test_seed1234.pkl
```
To run baselines, you need to install [Compass](https://github.com/bcamath-ds/compass) by running the `install_compass.sh` script from within the `problems/op` directory and [Concorde](http://www.math.uwaterloo.ca/tsp/concorde.html) using the `install_concorde.sh` script from within `problems/tsp`. [LKH3](http://akira.ruc.dk/~keld/research/LKH-3/) should be automatically downloaded and installed when required. To use [Gurobi](http://www.gurobi.com), obtain a ([free academic](http://www.gurobi.com/registration/academic-license-reg)) license and follow the [installation instructions](https://www.gurobi.com/documentation/8.1/quickstart_windows/installing_the_anaconda_py.html).

### Other options and help
You could run the command bellow or see the comments in options.py or eval.py
```bash
python run.py -h
python eval.py -h
```