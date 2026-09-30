"""Worst-case GPU memory check for the current model config (read from the same
env vars as train.py). Builds one batch of BATCH_SIZE of the largest training
samples (from bucket_meta.tsv) and runs forward + backward + optimizer step on a
single GPU with bf16 autocast, then prints peak memory and step time.
"""
import csv
import os
import time

import torch
import torch.optim as optim

from model import Fox
from train import (BUCKET_META_NAME, LazyFoxDataset, collate_fn,
                   axial_layers, cross_layers, dropout, grad_ckpt, hidden_dim,
                   use_flex, use_opm)

train_dir  = os.environ.get("TRAIN_DIR", "/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_asymmetree/generated_trees/new_train")
batch_size = int(os.environ.get("BATCH_SIZE", "32"))
n_steps    = int(os.environ.get("PROBE_STEPS", "5"))


def largest_files():
    with open(os.path.join(train_dir, BUCKET_META_NAME), newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    rows.sort(key=lambda r: (int(r["para_leaves"]), int(r["mapping_count"]), int(r["host_leaves"])),
              reverse=True)
    top = rows[:batch_size]
    print(f"Largest batch: para_leaves {top[-1]['para_leaves']}-{top[0]['para_leaves']}, "
          f"max host_leaves {max(int(r['host_leaves']) for r in top)}, "
          f"max mappings {max(int(r['mapping_count']) for r in top)}")
    return [os.path.join(train_dir, r["file"]) for r in top]


def main():
    device = torch.device("cuda")
    ds = LazyFoxDataset(train_dir, pt_files=largest_files())
    batch = collate_fn([ds[i] for i in range(len(ds))])
    print(f"host_msa {tuple(batch['host_msa'].shape)}  parasite_msa {tuple(batch['parasite_msa'].shape)}")
    for k in ("host_msa", "parasite_msa", "labels", "host_dist", "para_dist"):
        batch[k] = batch[k].to(device)

    model = Fox(
        hidden_dim=hidden_dim, pair_dim=64, cls_dim=512, axial_layers=axial_layers,
        use_opm=use_opm, use_dist_matrix=True,
        gradient_checkpointing=grad_ckpt,
        num_cross_layers=cross_layers,
        use_flexattention=use_flex,
        dropout=dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Config: hidden={hidden_dim} axial={axial_layers} cross={cross_layers} "
          f"grad_ckpt={grad_ckpt} flex={use_flex}  params={n_params / 1e6:.1f}M")
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    torch.cuda.reset_peak_memory_stats()
    times = []
    for step in range(n_steps):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch["host_msa"], batch["parasite_msa"], batch["mappings"],
                        host_dist=batch["host_dist"], para_dist=batch["para_dist"])
            loss = ((out.float() - batch["labels"]) ** 2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.time() - t0)

    peak = torch.cuda.max_memory_allocated() / 2**30
    total = torch.cuda.get_device_properties(device).total_memory / 2**30
    print(f"Peak memory: {peak:.1f} / {total:.1f} GiB ({100 * peak / total:.0f}%)")
    print(f"Step time (worst batch, after warmup): {min(times[1:] or times):.2f} s")
    print("OK" if peak < 0.9 * total else "WARNING: >90% of GPU memory, risk of OOM")


if __name__ == "__main__":
    main()
