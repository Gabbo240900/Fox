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
  --c_lambda 0.2 1.6 \
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
