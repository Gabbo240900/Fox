### Generate TREES and SEQUENCIES (Traing 100 trees - need around 5x to get to target)

python simulate_input_trees.py --num_trees 500 \
--min_leaves 15 --max_leaves 50 \
--output_dir ./generated_trees/ \
--output_dir_tgl ./generated_trees/Datasets/ \
--jar_path ./cophylogeny-ML/code/coala/TGLGenerator.jar \
--num_threads 8

python alisim.py generated_trees/Datasets \
--substitution LG \
--gamma GC \
--iqtree ../bin/bin_macos/iqtree_2.2.0 \
--length 500 \
--max-attempts 1 \
--allow-duplicate-sequences 

### Generate TREES and SEQUENCIES (Test)

python simulate_input_trees.py --num_trees 10
--min_leaves 15 --max_leaves 50
--output_dir /Users/gabriele/Co-phyloformer/generate_host_freq/test_data/
--output_dir_tgl /Users/gabriele/Co-phyloformer/generate_host_freq/test_data/Datasets/
--jar_path /Users/gabriele/Co-phyloformer/generate_host_freq/cophylogeny-ML/code/coala/TGLGenerator.jar

python alisim.py test_data/Datasets
--substitution LG
--gamma GC
--iqtree /Users/gabriele/Co-phyloformer/bin/bin_macos/iqtree_2.2.0
--length 250
--max-attempts 1
--allow-duplicate-sequences

### Train model

python train.py

No need to create our own self attention network sinc MSAencoder (TransformerEconder and TransformerEncoderLayer work exactly as we need)
Cross attention is built upon results from TrasnformerEconder and VirtualNode as Parameters



### Simulate with tree ducken

python simulate_input_files.py \
  --h_lambda 0.5 1.2 \
  --c_lambda 0.7 1.6 \
  --s_lambda 0.5 1.2 \
  --s_her 0.05 0.3 \
  --num_trees 20 \
  --n_cores 8


python alisim.py generated_trees/Datasets \
  --substitution LG \
  --gamma GC \
  --iqtree ../bin/bin_macos/iqtree_2.2.0 \
  --length 500 \
  --max-attempts 1 \
  --allow-duplicate-sequences \
  --n_cores 8 \
  --temp-dir ../alisim_tmp


python train.py --dataset_dir ../generate_treeducken/generated_trees/Datasets 

time_to_sim -> Do different times for same parameters; set up a grid of time for which to un all different set of parameters [0.5 ; 4.5] # Not possible 


RESUME_CKPT=/Users/gabriele/Co-phyloformer/co-phyloformer/checkpoints/epoch2_batch4.pth python train.py
RESUME_CKPT=/lustre/fswork/projects/rech/vcu/commun/Co-Phyloformer/co-phyloformer/checkpoints/epoch9_batch9885.pth
--------------------------------------------------
# First train 
10 Datasets, only training
100 Epochs
1e-3 Learning Rate
F.softplus

No Schduler no temperature no grad clipping
sim_time is in the model.

# Second train
50 Datasets, training + validation (80-20)
100 Epochs
1e-4 LR
F.softplus

No Scheduler no temperature no
sim_time is in the model

# Third Train
100 datasets, train + validation (80-20)
100 epochs
1e-4 LR

F.softplus

No Scheduler no temperature no
sim_time is in the model

Model was Overfitting a lot on train set (epoch_loss train was diminishing but val_epoch loss started rising at 40 epochs)

# Fourth train 
100 datasets, train
100 epochs
1e-4 LR
Changed vritual node to mean of host and parasite
F.softplus
mappings were never used... (big problem with colalte_fn and mappings it was unbale to understand which mappings were belonging to which sample in a single batch)

No Scheduler no temperature no dropout
sim_time is in the model

# Fifth train 
100 datasets, train
300 epochs
1e-4 LR

F.softplus
Implemented Mapping correctly 


No Scheduler no temperature no drop out
sim_time is in the model

Model works perfectly with high overfitting but perfect predictions

# Six train 
80 datasets, train + validation (80/20 split)
200 epochs
1e-4 LR

High overfitting on training, validation is not so good. 

It took 2 hours to train 

# Seven Train
80 Dataset, train + validation (80/20 split)
200 epochs
AdamW instead of Adam with 1e-4 LR and weight decay 1e-4 

Also implemented scheduler with get_cosine_schedule_with_warmup

# Eight training 
60 Dataset, train + validation (80/20 split)
300 epochs

Implemented Dropout (p=0.2) in model.py and weight decay 5e-4

# Nine training 
150 Dataset, train + validation (80/20 split)
300 epochs

SmoothL1 instead of MSE Loss

No Scheduler 

Implemented Dropout (p=0.2) in model.py and weight decay 5e-4

hidden_dim = 128 and num_heads reduced to 2 to speed up process

Better than more heavy models since we have more data. 

# Ten training 
200 Dataset, train + validation (80/20 split)
300 epochs

SmoothL1 instead of MSE Loss

Dropout (p=0.2) and weight decay 1e-3

hidden_dim = 128 and num_heads reduced to 2 to speed up process

Separate Heads for Cospeciation and Host Switch

Added separate losses for each head 

Stopped early 



# First Run huber
Training worked perfectly on 200 datasets with 0.000081 loss. ADD VALIDATION RESULTS 

Masking

1 dropout layer (0.1) after transformer

optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)

CLS token 

sim_time used with FiLm

Bigger model for cosp and switch 

self.cospeciation_head = nn.Sequential(
    nn.LayerNorm(self.concat_dim),
    nn.Linear(self.concat_dim, hidden_dim),
    nn.ReLU(),
    nn.Identity(),
    nn.Linear(hidden_dim, hidden_dim // 2),
    nn.ReLU(),
    nn.Identity(),
    nn.Linear(hidden_dim // 2, 1)
)
self.switch_head = nn.Sequential(
    nn.LayerNorm(self.concat_dim),
    nn.Linear(self.concat_dim, hidden_dim),
    nn.ReLU(),
    nn.Identity(),
    nn.Linear(hidden_dim, hidden_dim // 2),
    nn.ReLU(),
    nn.Identity(),
    nn.Linear(hidden_dim // 2, 1)
)