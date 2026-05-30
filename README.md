# Prophet

Transformer-based model for inferring host–symbiont co-phylogenetic event frequencies (Speciation, Host-switch / HGT, Loss, Duplication) directly from paired multiple sequence alignments (MSAs).

Prophet takes two MSAs — one for host taxa, one for symbiont taxa — and predicts the relative frequencies of the four cophylogenetic events that shaped the joint evolutionary history of the two clades. The model learns end-to-end from simulated host/symbiont trees and sequences, with no need for explicit tree inference at inference time.

---

## Repository layout

```
Prophet/
├── training/        # Main model + training/validation/inference pipeline (asymmetric pair track)
├── Prophet_analysis/    # Post-hoc analysis scripts, plots, summary tables
├── generate_data/        # Simulation pipeline based on AsymmeTree (trees + MSAs)
├── example_data/               # Small example datasets for smoke tests
├── bin/                        # Bundled external binaries (iqtree, coevolution simulators)
├── test_model.ipynb            # Notebook demo for loading a checkpoint and running predictions
└── README.md
```

The `training/` folder is the active codebase. Older experimental folders (`Prophet/`, `co_phyloformer_test/`, `Prophet_test2/`) are kept locally but ignored by git.

---

## Installation

Requirements: Python ≥ 3.10, CUDA-capable GPU recommended for training (CPU works for inference on small inputs).

```bash
git clone https://github.com/<user>/Prophet.git
cd Prophet
conda create -n cophylo python=3.10
conda activate cophylo
pip install -r requirements.txt
```

Core dependencies: `torch`, `lightning`, `numpy`, `pandas`, `biopython`, `ete3`, `tqdm`, `wandb`, `matplotlib`.

External tools (already in `bin/`):
- `iqtree` (v2.2.0) — sequence simulation via AliSim
- `TGLGenerator.jar` (CoALA) — alternative host/symbiont tree simulator
- `simulateWithCoevolution` — site-level coevolution simulator

---

## Data generation

### Option A — AsymmeTree pipeline (recommended)

```bash
cd generate_data

# 1. Simulate paired host/symbiont trees
python generate_trees.py

# 2. Simulate MSAs along each tree with AliSim
python alisim.py generated_trees/Datasets \
  --substitution LG \
  --gamma GC \
  --iqtree ../bin/bin_macos/iqtree_2.2.0 \
  --length 500 \
  --max-attempts 1 \
  --allow-duplicate-sequences \
  --temp-dir ../alisim_tmp

# 3. (Optional) Filter / analyse the simulated dataset
python filter_data.py
python analyze_data.py
```

Outputs land in `generate_data/generated_trees/Datasets/` as `.pt` tensors paired with target event-frequency vectors.

### Option B — CoALA / TGLGenerator pipeline (legacy)

```bash
python simulate_input_trees.py \
  --num_trees 500 \
  --min_leaves 15 --max_leaves 50 \
  --output_dir ./generated_trees/ \
  --output_dir_tgl ./generated_trees/Datasets/ \
  --jar_path ./cophylogeny-ML/code/coala/TGLGenerator.jar \
  --num_threads 8

python alisim.py generated_trees/Datasets \
  --substitution LG --gamma GC \
  --iqtree ../bin/bin_macos/iqtree_2.2.0 \
  --length 500 --max-attempts 1 --allow-duplicate-sequences
```

---

## Training

```bash
cd training
python train.py
```

Key training entry points:
- `train.py` — main training loop (Lightning Fabric, DDP-ready, mixed precision).
- `model.py` — `Prophet` architecture: MSA embedder + pair embedder + asymmetric axial attention blocks.
- `data.py` — dataset loaders and bucket sampling for variable-size pairs.
- `validation.py` — periodic validation pass + metric logging to W&B.
- `prepare_buckets.py` — pre-computes bucketed dataset indices for efficient batching.
- `pre_encoder.py` — optional pre-encoding of MSAs to speed up training.

Resume from a checkpoint:

```bash
RESUME_CKPT=training/checkpoints/epoch2_batch4.pth python train.py
```

SLURM job scripts (`*.slurm`) are provided for cluster training: `100H_training.slurm`, `20H_training.slurm`, `dev_training.slurm`.

---

## Inference / testing

```bash
cd training
python test.py --ckpt checkpoints/<your_ckpt>.pth --data <path/to/test_set>
```

For an interactive demo see [`test_model.ipynb`](test_model.ipynb).

---

## Analysis & plots

```bash
cd Prophet_analysis
# scripts/ : per-run analysis scripts
# plots/   : generated figures
# tables/  : summary CSVs
```

See [`Prophet_analysis/SUMMARY.md`](Prophet_analysis/SUMMARY.md) for a rundown of available analyses.

---

## Model architecture

- **MSAEmbedder** — one-hot AA encoding → linear projection → ReLU.
- **PairEmbedder** — PAD-aware mean pooling over sequence positions, then pairwise outer sum to initialise the pair track.
- **Axial attention blocks** — alternating row/column attention over the MSA, with an asymmetric pair-bias track linking host and symbiont representations.
- **Prediction head** — outputs a 4-dim softmax over event frequencies (Speciation, HGT, Loss, Duplication).

Vocabulary: 23 tokens (20 amino acids + gap + UNK + PAD).

---

## Citation

If you use Prophet in published work, please cite:

```
@unpublished{Prophet2026,
  title  = {Prophet: Transformer-based inference of cophylogenetic event frequencies},
  author = {<authors>},
  year   = {2026}
}
```

---

## License

See [`LICENSE`](LICENSE).
