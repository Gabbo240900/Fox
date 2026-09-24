<h1>Fox<img src="fox_scientific_research_logo.svg" alt="Fox logo" width="160" align="right"><br clear="right"></h1>

Transformer model that infers host–symbiont **cophylogenetic event frequencies** — Speciation, Host-switch / HGT, Loss, Duplication — straight from a pair of multiple sequence alignments (MSAs).

Feed Fox two MSAs (one for the host taxa, one for the symbiont taxa) plus a host↔symbiont leaf mapping. It returns the relative frequencies of the four events that shaped their shared evolutionary history. The model is trained end-to-end on simulated host/symbiont trees and sequences. **No tree inference at prediction time.**

---

## Table of contents

1. [What the model does](#what-the-model-does)
2. [Repository layout](#repository-layout)
3. [Installation](#installation)
4. [Using the pretrained model (`fox.ckpt`)](#using-the-pretrained-model-foxckpt)
5. [Generating training data](#generating-training-data)
6. [Training — local](#training--local)
7. [Training — external GPU / cluster](#training--external-gpu--cluster)
8. [Training outputs](#training-outputs)
9. [Other folders](#other-folders)

---

## What the model does

**Input:** a host MSA, a symbiont MSA, and a list of host↔symbiont leaf pairs.

**Output:** four numbers that sum to 1 —

```
[ Speciation, HGT, Loss, Duplication ]
```

Example: `[0.84, 0.05, 0.01, 0.11]` → this host/symbiont pair evolved mostly by speciation, with a bit of duplication and little host-switching or loss.

Architecture (see [training/model.py](training/model.py)):

- **MSAEmbedder** — amino-acid tokens → one-hot → linear projection. Vocabulary: 23 tokens (20 AAs + gap + UNK + PAD). Max sequence length: 250.
- **PairEmbedder** — PAD-aware mean pooling per sequence, then an outer sum that seeds the pair track from the host/symbiont mapping.
- **Axial attention blocks** — alternating row / column attention over each MSA, with an asymmetric pair-bias track that links host and symbiont.
- **Cross-attention blocks** — bidirectional host ↔ symbiont attention.
- **Prediction head** — softmax over the four event classes.

Jukes–Cantor distance matrices (computed from each MSA) feed the pair track; they are computed on the fly when not supplied.

---

## Repository layout

```
Fox/
├── fox/                       # Inference package + `fox` CLI
│   ├── __init__.py            # exports: predict_tgl, load_model, EVENT_NAMES
│   ├── cli.py                 # `fox predict --tgl …`
│   ├── core.py                # predict_tgl(): .tgl → 4 event frequencies
│   ├── io.py                  # read_tgl(): parse a .tgl bundle
│   ├── encoding.py            # AA tokenisation + Jukes–Cantor distances
│   └── model_loader.py        # load_model(): build Fox from fox.ckpt (cached)
├── training/                  # Model + training pipeline
│   ├── model.py               # Fox architecture
│   ├── train.py               # Training loop (Lightning Fabric)
│   ├── data.py                # Dataset loaders + bucket sampling
│   ├── validation.py          # Validation loop + W&B logging
│   ├── pre_encoder.py         # .tgl → .pt pre-encoding
│   ├── prepare_buckets.py     # Bucket metadata for the sampler
│   ├── plot.py                # Plot predictions during training
│   ├── plot_from_csv.py       # Replot from saved CSVs
│   └── post_process.py        # Standalone plotting from CSVs
├── generate_data/             # AsymmeTree + AliSim simulation pipeline
│   ├── generate_trees.py      # Simulate paired host/symbiont trees
│   ├── alisim.py              # Simulate MSAs along trees (AliSim)
│   ├── filter_data.py         # Filter .pt by taxa / seq length
│   ├── analyze_data.py        # Summary plots over a .pt directory
│   └── generated_trees/       # Default output root + priors
├── test_data/                 # Held-out evaluation data
│   ├── Datasets/              # Ground-truth .tgl files (40 datasets)
│   ├── fox_data/              # Pre-encoded .pt samples for Fox
│   ├── amocoala_data/         # Per-dataset AmoCoala outputs
│   ├── amocoala_pipeline/     # Inputs / intermediates for AmoCoala
│   └── pt_samples/            # Extra .pt samples
├── train_data_distributions/  # Plots of the training-set distributions
├── results/                   # train/val prediction CSVs + plots
├── bin/                       # Bundled binaries (bin_linux/, bin_macos/)
├── fox.ckpt                   # Released pretrained checkpoint (~76 MB)
├── test_model.ipynb           # Notebook: load ckpt, predict, AmoCoala comparison
├── install.sh                 # Conda env bootstrap
├── pyproject.toml             # Packaging + `fox` console script
├── requirements.txt
└── README.md
```

---

## Installation

**Requires** Python ≥ 3.11. A CUDA GPU is recommended for training; CPU is fine for inference on small inputs.

**Inference only** — just want to run `fox predict`? PyTorch is the only runtime dependency:

```bash
git clone https://github.com/Gabbo240900/Fox.git
cd Fox
pip install -e .            # installs torch + the `fox` CLI
fox predict --tgl test_data/Datasets/Dataset11.tgl
```

**Full install** (training + simulation + notebook) — one-shot conda env named `fox`:

```bash
git clone https://github.com/Gabbo240900/Fox.git
cd Fox
./install.sh                # or: ./install.sh my_env_name
conda activate fox
pip install -e .            # adds the `fox` CLI on top
```

Manual:

```bash
conda create -n fox python=3.11
conda activate fox
pip install -r requirements.txt
```

Bundled binary under `bin/` (pick the subfolder for your OS):

- `iqtree_2.2.0` — MSA simulation via AliSim. Third-party (GNU GPL v2); see [`bin/THIRD_PARTY_LICENSES.md`](bin/THIRD_PARTY_LICENSES.md).

For the optional AmoCoala comparison, get `AmoCoala.jar` from https://github.com/sinaimeri/AmoCoala.

---

## Using the pretrained model (`fox.ckpt`)

The released checkpoint at the repo root (`fox.ckpt`, ~76 MB) is ready to use.

### Quick start: the `fox` command

One well-formed `.tgl` in, four event frequencies out. No notebook, no manual tensor wiring.

```bash
pip install -e .        # from the repo root; registers `fox`, finds fox.ckpt
fox predict --tgl test_data/Datasets/Dataset11.tgl
```

```
Event          Frequency
------------------------
Speciation        0.8352
HGT               0.0484
Loss              0.0100
Duplication       0.1065

Dominant: Speciation (83.5%)
```

**Options:**

| flag | default | purpose |
|------|---------|---------|
| `--tgl` | required | input `.tgl` bundle (host MSA + symbiont MSA + mapping) |
| `--json` | off | emit JSON instead of the table (for scripting) |
| `--ckpt` | repo `fox.ckpt` | use a different checkpoint |
| `--device` | `cpu` | `cpu`, `cuda`, or `mps` |

**From Python:**

```python
from fox import predict_tgl

predict_tgl("test_data/Datasets/Dataset11.tgl")
# {'Speciation': 0.835, 'HGT': 0.048, 'Loss': 0.010, 'Duplication': 0.106}
```

Scoring many files? Load the model once and reuse it:

```python
from fox import load_model, predict_tgl

model = load_model()                       # builds Fox from fox.ckpt (cached)
for path in tgl_files:
    print(predict_tgl(path, model=model))
```

**What a `.tgl` must contain:** a `BEGIN HOST;` block and a `BEGIN PARASITE;` block, each with an `ALIGNMENT` of `name sequence` rows, plus a `BEGIN DISTRIBUTION;` block listing `parasite : host` leaf pairs. See any file under [`test_data/Datasets/`](test_data/Datasets/) for the exact layout.

**Checkpoint not found?** If `fox.ckpt` was moved, or you installed outside the repo, point Fox at it: `export FOX_CKPT=/path/to/fox.ckpt`, or pass `--ckpt`.

### The notebook

Open [`test_model.ipynb`](test_model.ipynb). It covers:

1. Loading `fox.ckpt`.
2. Single-file prediction from a `.pt` or `.tgl` input.
3. Batch evaluation across `test_data/fox_data/` with metrics (MAE, RMSE, R², Pearson r).
4. Three-way comparison — Fox vs. AmoCoala vs. ground truth — on simulated test data.

### Low-level API

Build the model directly and feed it tensors:

```python
import torch
from training.model import Fox

ckpt = torch.load("fox.ckpt", map_location="cpu", weights_only=False)
model = Fox(...).eval()           # constructor args match ckpt["config"]
model.load_state_dict(ckpt["model"])

# host_msa, para_msa  : [N, L] long tensors of AA indices (see training/pre_encoder.py)
# mappings            : list of (host_idx, parasite_idx) tuples
# host_dist, para_dist: Jukes–Cantor distance matrices, [N, N]
out = model(host_msa[None], para_msa[None], [mappings],
            host_dist=host_dist[None], para_dist=para_dist[None])
# out: [1, 4] — softmax over (Speciation, HGT, Loss, Duplication)
```

### Predicting from a raw `.tgl` at the tensor level

Pre-encode the `.tgl` to `.pt`, then load it:

```bash
python training/pre_encoder.py --src path/to/tgl_dir --dst path/to/pt_out
```

Each `.pt` holds `host_msa`, `para_msa`, `host_dist`, `para_dist`, `mappings`, and `labels`. Feed those straight into the model as above. (For a single file, `fox predict` / `predict_tgl` does the pre-encoding for you — this path is for building batches.)

---

## Generating training data

Full pipeline: trees → alignments → pre-encoded `.pt` → bucket metadata.

### 1. Simulate paired host/symbiont trees (AsymmeTree)

```bash
cd generate_data
python generate_trees.py \
  --base-path generated_trees/my_run \
  --num-trees 20 \
  --time-grid 1.5 2 2.5 3 3.5 \
  --seed 42
```

Per simulation, [`generate_trees.py`](generate_data/generate_trees.py) draws:

- `num_leaves ~ U[15, 50]`
- `host_birth_rate ~ U[0.5, 1.2]`, `host_death_rate ~ U[0.2, 0.4] · birth_rate`
- `hgt_rate ~ U[0.05, 0.3]`, `dupl_rate ~ U[0.2, 0.4]`, `loss_rate ~ U[0.2, 0.4]`

Output under `generated_trees/my_run/`:

```
species_trees/      species_tree_<i>.nwk     # host trees, Newick
gene_trees/         gene_tree_<i>.nwk        # symbiont trees, Newick
associations/       associations_<i>.csv     # parasite_leaf → host_leaf
reconciliations/    reconciliation_<i>.csv   # per-node event labels
Datasets/           Dataset<i>.tgl           # NEXUS-style bundle
```

The `.tgl` files are the input for the next stages.

### 2. Simulate MSAs along the trees (AliSim / IQ-TREE)

```bash
python alisim.py generated_trees/my_run/Datasets \
  --substitution LG \
  --gamma GC \
  --iqtree ../bin/bin_macos/iqtree_2.2.0 \
  --length 500 \
  --max-attempts 1 \
  --allow-duplicate-sequences \
  --n_cores 16 \
  --temp-dir ../alisim_tmp
```

Flags ([`alisim.py`](generate_data/alisim.py)):

| flag | default | purpose |
|------|---------|---------|
| `--length / -l` | 250 | alignment length |
| `--substitution / -s` | `LG` | AA substitution model |
| `--gamma / -g` | none | rate-heterogeneity model (`G`, `GC`, …) |
| `--custom-model / -c` | none | path to a custom model definition |
| `--iqtree / -t` | required | path to the IQ-TREE 2 binary |
| `--max-attempts / -m` | 20 | retries per tree on sim failure |
| `--allow-duplicate-sequences / -d` | off | keep alignments with duplicate rows |
| `--n_cores / -n` | 16 | parallel workers |
| `--temp-dir` | system tmp | scratch dir for IQ-TREE |

Each `Dataset<i>.tgl` is augmented in place with the host and parasite MSAs.

### 3. Pre-encode `.tgl` → `.pt`

Turns NEXUS bundles into tokenised tensors for the dataloader:

```bash
python training/pre_encoder.py \
  --src generate_data/generated_trees/my_run/Datasets \
  --dst data/pt/my_run
```

Each `.pt` contains:

```
host_msa  : Long [N_h, 250]    # AA indices, PAD=22
para_msa  : Long [N_p, 250]
host_dist : Float [N_h, N_h]   # Jukes–Cantor pairwise distance
para_dist : Float [N_p, N_p]
mappings  : list[(host_idx, para_idx)]
labels    : {Speciation, HGT, Loss, Duplication}
```

### 4. (Optional) filter the dataset

[`filter_data.py`](generate_data/filter_data.py) prunes `.pt` samples by taxa count, sequence length, or missing-event flags — moving or deleting offenders and writing a CSV report:

```bash
python generate_data/filter_data.py data/pt/my_run \
  --min_taxa 4 --max_taxa 128 \
  --min_seq_len 50 --max_seq_len 250 \
  --move --side_dir data/pt/my_run_rejected \
  --csv_out data/pt/my_run_filter_report.csv
```

### 5. (Optional) summarise the dataset

```bash
python generate_data/analyze_data.py data/pt/my_run --bins 30
```

Histograms of taxa counts, sequence lengths, and event-frequency distributions — like the plots in [`train_data_distributions/`](train_data_distributions/).

### 6. Build bucket metadata (required before training)

The bucketed sampler needs per-file shape metadata:

```bash
python training/prepare_buckets.py --dirs data/pt/my_run data/pt/my_val
```

Writes a `bucket_meta.tsv` next to each `.pt` directory.

---

## Training — local

`training/train.py` is driven by **environment variables**, not CLI flags (apart from `train` / `resume`).

Minimum run:

```bash
cd training
TRAIN_DIR=../data/pt/my_run \
VAL_DIR=../data/pt/my_val \
python train.py train
```

Resume from a checkpoint:

```bash
TRAIN_DIR=../data/pt/my_run VAL_DIR=../data/pt/my_val \
python train.py resume checkpoints/last_epoch.ckpt
```

### Key environment variables

| variable | default | meaning |
|----------|---------|---------|
| `TRAIN_DIR`, `VAL_DIR` | — *(required)* | pre-encoded `.pt` directories |
| `EPOCHS` | 50 | training epochs |
| `BATCH_SIZE` | 64 | per-GPU batch |
| `GRAD_ACCUM` | 4 | gradient accumulation steps |
| `LR` | 1e-4 | base learning rate |
| `WEIGHT_DECAY` | 0.05 | AdamW weight decay |
| `HUBER_DELTA` | 1.0 | Huber loss δ |
| `UNDER_PENALTY` | 2.5 | extra weight when underestimating an event |
| `TAIL_WEIGHT` | 2.0 | upweight rare-event tails |
| `HGT_LOSS_WEIGHT` | 2.0 | extra weight for the HGT class |
| `MID_EPOCH_VALS` | 1 | mid-epoch validation passes |
| `NUM_WORKERS` | 8 | dataloader workers |
| `VAL_BATCH_MULT` | 1 | validation batch-size multiplier |
| `CKPT_DIR` | `checkpoints` | output dir for `.ckpt` files |
| `EARLY_STOP_PATIENCE` | 0 (off) | epochs without improvement before stop |
| `AXIAL_LAYERS` | 2 | number of axial attention blocks |
| `CROSS_LAYERS` | 1 | number of cross-attention blocks |
| `HIDDEN_DIM` | 256 | model hidden dim |
| `DROPOUT` | 0.1 | dropout |
| `USE_OPM` | 0 | enable outer-product mean update |
| `USE_FLEX_ATTENTION` | 0 | flex_attention (PyTorch ≥ 2.5) |
| `USE_COMPILE` | 0 | `torch.compile` |
| `GRADIENT_CHECKPOINTING` | 0 | trade compute for memory |
| `USE_BUCKETED_BATCHES` | 1 | shape-bucketed sampler |
| `BUCKET_SIZE` | 8 | bucket granularity |
| `HOST_MAX_LEAVES` | 51 | host-tree leaf cap |
| `PARA_MAX_LEAVES` | 142 | parasite-tree leaf cap |
| `WANDB_MODE` | `offline` | set to `online` to stream to W&B |
| `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_NAME` | … | W&B identifiers |

Hardware:

- **Single GPU** — run as above. Lightning Fabric auto-detects CUDA and uses bf16-mixed.
- **Multi-GPU, one host** — launched automatically through DDP (`devices="auto"`).
- **CPU** — works but slow; lower `BATCH_SIZE` and set `USE_BUCKETED_BATCHES=0` if memory is tight.

---

## Training — external GPU / cluster

Same script, same env vars.

### A. Single remote GPU box (SSH)

```bash
# on the remote machine
git clone https://github.com/Gabbo240900/Fox.git
cd Fox && ./install.sh && conda activate fox

# copy pre-encoded data over
rsync -avz data/pt/ user@gpu-host:~/Fox/data/pt/

# run
cd training
TRAIN_DIR=$HOME/Fox/data/pt/my_run \
VAL_DIR=$HOME/Fox/data/pt/my_val \
EPOCHS=80 BATCH_SIZE=128 \
WANDB_MODE=online WANDB_PROJECT=Fox WANDB_NAME=remote_run \
python train.py train
```

Wrap long runs in `tmux` / `screen` / `nohup` so an SSH drop doesn't kill the job.

### B. Multi-GPU node / Slurm

`train.py` uses `lightning.fabric.Fabric` with `DDPStrategy` and `devices="auto"`.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=fox
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate fox
cd $SLURM_SUBMIT_DIR/training

export TRAIN_DIR=/scratch/$USER/fox/pt/my_run
export VAL_DIR=/scratch/$USER/fox/pt/my_val
export EPOCHS=100
export BATCH_SIZE=64
export NUM_WORKERS=8
export WANDB_MODE=online
export WANDB_PROJECT=Fox
export WANDB_NAME=slurm_${SLURM_JOB_ID}

srun python train.py train
```

Notes:

- Set `FIND_UNUSED_PARAMETERS=1` if you change the model graph and DDP complains.
- `USE_COMPILE=1` and `USE_FLEX_ATTENTION=1` need PyTorch ≥ 2.5; disable on older cluster images.
- Every rank must see the same `TRAIN_DIR` / `VAL_DIR` on a shared filesystem.

---

## Training outputs

During and after a run:

- **`$CKPT_DIR/` (default `training/checkpoints/`)**
  - `last_epoch.ckpt` — end of every epoch (resumable).
  - `latest.ckpt` — every mid-epoch validation.
  - `best.ckpt` — best validation loss so far.
- **`results/train_predictions.csv`, `results/val_predictions.csv`** — one row per sample, columns `true_<event>` and `pred_<event>` for all four events.
- **`results/plots/`** — scatter / density plots from [`training/plot.py`](training/plot.py) and [`training/plot_from_csv.py`](training/plot_from_csv.py). Replot any time:
  ```bash
  python training/post_process.py \
    --train-csv results/train_predictions.csv \
    --val-csv   results/val_predictions.csv \
    --output-dir results/plots --bins 40
  ```
- **W&B run** (if `WANDB_MODE=online`) — losses, per-event metrics, prediction histograms, learning rate.

A reloaded checkpoint holds:

```
ckpt["model"]      # state_dict
ckpt["optimizer"]  # optimizer state
ckpt["scheduler"]  # LR scheduler state
ckpt["epoch"]      # last completed epoch
ckpt["config"]     # env-var snapshot used for the run
```

---

## Other folders

- **`test_data/Datasets/`** — 40 held-out `.tgl` files for benchmarking.
- **`test_data/fox_data/`** — those datasets pre-encoded as `.pt` for direct Fox inference.
- **`test_data/amocoala_data/<DatasetXX>/`** — AmoCoala reconstructions per test dataset (used by the 3-way comparison in [`test_model.ipynb`](test_model.ipynb)).
- **`test_data/amocoala_pipeline/`** — intermediates (`alignments/`, `nexus/`, `sequences/`, `trees/`, `amocoala/`) for re-running AmoCoala from scratch.
- **`test_data/pt_samples/`** — extra example `.pt` inputs for quick checks.
- **`train_data_distributions/`** — reference plots of the training-set distributions.
- **`bin/`** — bundled binaries for the simulation pipeline; pick the subfolder for your OS.
- **`results/`** — train/val prediction CSVs plus plots; safe to delete and regenerate.
- **`fox.ckpt`** — released checkpoint loaded by the notebook and CLI.

---

## Citation

```
@unpublished{Fox2026,
  title  = {Fox: Transformer-based inference of cophylogenetic event frequencies},
  author = {Di Palma, Gabriele},
  year   = {2026}
}
```

## License

MIT — see [`LICENSE`](LICENSE).
