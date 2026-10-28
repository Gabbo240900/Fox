"""Figure-1 trees pruned to Fox's 52 pairs -> AmoCoala input amocoala/fig1_52.nex.

fig1_host.nwk and fig1_symbiont.nwk hold the host and symbiont topologies of figure 1 of Arab et al. (2020),
transcribed by hand (all 55 pairs of the study, no branch lengths). Fox sees the 52 pairs with an unambiguous
label (see make_tgl.py), so this keeps only the pairs listed in Datasets/blattabacterium.tgl, removes the other leaves and collapses the
nodes left with one child (AmoCoala crashes on them), so both tools see the same pairs. Host and symbiont of a
pair share the pair name; they get the prefixes H_ and P_ in the AmoCoala input.

Run from the repository root: python test_data/real_data/blattabacterium/make_fig1_52.py
Then: java -jar AmoCoala.jar -input test_data/real_data/blattabacterium/amocoala/fig1_52.nex \
          -N 2000 -M 100 -R 3 -t 0.1,0.25,0.25 -threads 10
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "amocoala", "fig1_52.nex")

tgl = open(os.path.join(HERE, "Datasets", "blattabacterium.tgl")).read()
pairs = re.findall(r"^\t\t(\S+): \1$", tgl.split("RANGE", 1)[1], flags=re.M)
assert len(pairs) == 52, "rerun make_tgl.py first (Datasets/blattabacterium.tgl must hold all 52 pairs)"


def parse(s, pos=0):
    """Newick without branch lengths -> (name, children)."""
    if s[pos] == "(":
        children = []
        pos += 1
        while True:
            child, pos = parse(s, pos)
            children.append(child)
            if s[pos] == ",":
                pos += 1
            else:
                break
        assert s[pos] == ")"
        return ("", children), pos + 1
    start = pos
    while s[pos] not in ",()":
        pos += 1
    return (s[start:pos], []), pos


def prune(node, keep):
    """Drop leaves not in keep; collapse nodes left with one child."""
    name, children = node
    if not children:
        return node if name in keep else None
    children = [c for c in (prune(c, keep) for c in children) if c is not None]
    if not children:
        return None
    return children[0] if len(children) == 1 else ("", children)


def to_newick(node, prefix):
    name, children = node
    return prefix + name if not children else "(" + ",".join(to_newick(c, prefix) for c in children) + ")"


def leaves(node):
    return [node[0]] if not node[1] else [l for c in node[1] for l in leaves(c)]


trees = {}
for block, nwk, prefix in [("HOST", "fig1_host.nwk", "H_"), ("PARASITE", "fig1_symbiont.nwk", "P_")]:
    s = re.sub(r"\s", "", open(os.path.join(HERE, nwk)).read()).rstrip(";")
    tree, end = parse(s)
    assert end == len(s)
    pruned = prune(tree, set(pairs))
    assert sorted(leaves(pruned)) == sorted(pairs)
    trees[block] = to_newick(pruned, prefix)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write("#NEXUS\n"
            f"BEGIN HOST;\n\tTREE HOST = {trees['HOST']};\nENDBLOCK;\n"
            f"BEGIN PARASITE;\n\tTREE PARASITE = {trees['PARASITE']};\nENDBLOCK;\n"
            "BEGIN DISTRIBUTION;\n\tRANGE\n\t\t" + ", ".join(f"P_{p} : H_{p}" for p in pairs)
            + ";\nENDBLOCK;\n")
print(f"wrote {OUT}: {len(pairs)} pairs")
