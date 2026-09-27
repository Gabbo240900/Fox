#!/bin/bash
#SBATCH --job-name=zip_asym
#SBATCH --account=vcu@cpu
#SBATCH --partition=prepost
#SBATCH --time=20:00:00
#SBATCH --output=zip_asym_%j.out
#SBATCH --error=zip_asym_%j.out

SRC=/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_asymmetree/generated_trees
DST=/lustre/fswork/projects/rech/vcu/commun/Co-Phyloformer/generate_asymmetree/generated_trees
OUT="$DST/train_val_asym.zip"

mkdir -p "$DST"
rm -f "$OUT"
cd "$SRC" || exit 1

TOTAL=$(du -sbc new_train new_val | tail -1 | cut -f1)
echo "[$(date '+%F %T')] Start. Data to archive: $(numfmt --to=iec "$TOTAL")"

# Progress monitor (zip writes to a temp file zi* in DST, then renames it)
(
  while true; do
    sleep 60
    DONE=$(stat -c %s "$DST"/zi* "$OUT" 2>/dev/null | awk '{s+=$1} END {print s+0}')
    PCT=$(awk -v d="$DONE" -v t="$TOTAL" 'BEGIN {printf "%.1f", 100*d/t}')
    echo "[$(date '+%F %T')] $(numfmt --to=iec "$DONE") / $(numfmt --to=iec "$TOTAL")  (${PCT}%)"
  done
) &
MON=$!

zip -r -q -0 "$OUT" new_train new_val
RC=$?

kill $MON 2>/dev/null
echo "[$(date '+%F %T')] zip finished with exit code $RC"
ls -lh "$OUT"
unzip -l "$OUT" | tail -1
