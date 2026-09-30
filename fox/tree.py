"""Minimal Newick helpers for simulated (dated, ultrametric) trees.

Simulated gene trees keep lost genes as leaves that end before the present,
and both trees carry a planted root node (P0 / H0) that AliSim writes out as
an extra sequence. Real data has neither, so extant_leaves() tells which
leaves to keep.
"""

import re
from typing import Dict, Set

_LABEL = re.compile(r"([^:,();]*)(?::([^,();]*))?")


def leaf_depths(newick: str) -> Dict[str, float]:
    """Leaf name -> distance from the root (sum of branch lengths)."""
    s = newick.strip().rstrip(";")
    parent, length, names, has_child = [-1], [0.0], [""], [False]
    cur, i = 0, 0
    while i < len(s):
        c = s[i]
        if c in "(,":
            if c == ",":
                cur = parent[cur]
            has_child[cur] = True
            parent.append(cur)
            length.append(0.0)
            names.append("")
            has_child.append(False)
            cur = len(parent) - 1
            i += 1
        elif c == ")":
            cur = parent[cur]
            i += 1
        else:
            m = _LABEL.match(s, i)
            name, ln = m.group(1).strip(), m.group(2)
            if name:
                names[cur] = name
            if ln and ln.strip():
                length[cur] = float(ln)
            i = max(m.end(), i + 1)

    depth = [0.0] * len(parent)
    for k in range(1, len(parent)):  # parents are always created before children
        depth[k] = depth[parent[k]] + length[k]
    return {names[k]: depth[k] for k in range(len(parent)) if not has_child[k]}


def extant_leaves(newick: str, rel_tol: float = 1e-6) -> Set[str]:
    """Leaves that reach the present (max root-to-tip depth) in a dated tree.

    Lost-gene leaves end early and are excluded; internal nodes such as the
    planted root are never leaves, so they are excluded too.
    """
    depths = leaf_depths(newick)
    if not depths:
        return set()
    height = max(depths.values())
    cutoff = height - rel_tol * max(height, 1.0)
    return {name for name, d in depths.items() if d >= cutoff}
