#!/bin/bash
#SBATCH --job-name=preprocess_all
#SBATCH --partition=cpu_p1
#SBATCH --account=vcu@cpu
#SBATCH --time=100:00:00
#SBATCH --qos=qos_cpu-t4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --output=preprocess_all_%j.out
#SBATCH --error=preprocess_all_%j.err

set -euxo pipefail

module purge
module load arch/h100
module load pytorch-gpu/py3/2.6.0
source $WORK/envs/cophylo_pt26/bin/activate

# ── paths (edit these) ────────────────────────────────────────────────────────
BASE=/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_asymmetree/generated_trees
export PREENCODE_SRC_DIR="$BASE/Datasets/"        # raw .tgl in
export PREENCODE_DST_DIR="$BASE/asym_preencoded/" # .pt out — preencode + filter dir
DATA_DIR="$PREENCODE_DST_DIR"
FILTER_SCRIPT=/lustre/fswork/projects/rech/vcu/commun/Co-Phyloformer/generate_asymmetree/filter_data.py

# train/val output dirs — must match train.py (new_train / new_val)
TRAIN_DIR="$BASE/new_train"
VAL_DIR="$BASE/new_val"
VAL_FRAC=0.2
SEED=42

# ── filter thresholds ─────────────────────────────────────────────────────────
MIN_TAXA=5
MAX_TAXA=142

# ── 1. pre-encode: .tgl → .pt ─────────────────────────────────────────────────

# ── 2. filter: drop bad .pt (irreversible --delete) ───────────────────────────
python "$FILTER_SCRIPT" "$DATA_DIR" \
  --min_taxa "$MIN_TAXA" \
  --max_taxa "$MAX_TAXA" \
  --delete

# ── 3. split: move filtered .pt into train/val dirs ───────────────────────────
python split_train_val.py \
  --src "$DATA_DIR" \
  --train_dir "$TRAIN_DIR" \
  --val_dir "$VAL_DIR" \
  --val_frac "$VAL_FRAC" \
  --seed "$SEED"

# ── 4. bucket: write bucket_meta.tsv in both dirs ─────────────────────────────
python prepare_buckets.py --dirs "$TRAIN_DIR" "$VAL_DIR"

echo "DONE: preencode + filter + split + bucket -> $TRAIN_DIR , $VAL_DIR"
