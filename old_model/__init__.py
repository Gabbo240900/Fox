"""Frozen copy of the previous Fox model (main branch, fox.ckpt) for comparison.

model.py    <- main:training/model.py   (host/symbiont cross-attention head)
encoding.py <- main:fox/encoding.py     (4-state Jukes-Cantor distances)
io.py       <- main:fox/io.py           (.tgl reader, keeps lost-gene leaves and root row)

Copied from main at commit aefe81fc.
Weights: fox_old.ckpt at the repo root (copy of main's fox.ckpt, not tracked by git).
"""
