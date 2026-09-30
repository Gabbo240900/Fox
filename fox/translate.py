"""Translate a nucleotide .tgl into a protein .tgl that Fox can read.

Fox is trained on amino-acid alignments (20 letters + gap). Real datasets are
often DNA (e.g. mitochondrial COI/COII); feeding those directly makes Fox read
A/C/G/T as the amino acids Ala/Cys/Gly/Thr. This module translates the HOST and
PARASITE ALIGNMENT blocks codon by codon, keeping the alignment:

  - '---'                     -> '-'
  - codon with some gaps      -> '-'  (alignment edge / missing data)
  - codon with ambiguous base -> 'X'  (read as UNK by fox.encoding)
  - stop codon                -> dropped when most sequences stop there (gene
                                 boundary), otherwise 'X' (frameshift / error)

The reading frame is picked automatically as the one with the fewest stop
codons, unless given. Everything else in the .tgl is copied unchanged.

    python -m fox.translate in.tgl out.tgl --table 5
    python -m fox.translate in.tgl out.tgl --host-table 2 --para-table 5
"""

import argparse
import re
from typing import Dict, List, Optional, Tuple

from Bio.Data import CodonTable

from .encoding import MAX_SEQ_LEN

_ALN_START = re.compile(r"^\s*ALIGNMENT\s")


def _codon_tables(table_id: int) -> Tuple[Dict[str, str], set]:
    t = CodonTable.unambiguous_dna_by_id[table_id]
    return dict(t.forward_table), set(t.stop_codons)


def _translate_codon(codon: str, forward: Dict[str, str], stops: set) -> str:
    codon = codon.upper().replace("U", "T")
    if codon == "---":
        return "-"
    if "-" in codon:
        return "-"
    if codon in stops:
        return "*"
    return forward.get(codon, "X")


def _translate_all(msa: Dict[str, str], frame: int, table_id: int) -> Dict[str, str]:
    forward, stops = _codon_tables(table_id)
    out = {}
    for name, seq in msa.items():
        out[name] = "".join(
            _translate_codon(seq[i:i + 3], forward, stops)
            for i in range(frame, len(seq) - 2, 3)
        )
    return out


def best_frame(msa: Dict[str, str], table_id: int) -> int:
    """Frame (0, 1, 2) with the fewest stop codons over all sequences."""
    counts = [sum(p.count("*") for p in _translate_all(msa, frame, table_id).values())
              for frame in range(3)]
    return min(range(3), key=counts.__getitem__)


def translate_msa(msa: Dict[str, str], table_id: int = 5,
                  frame: Optional[int] = None) -> Tuple[Dict[str, str], dict]:
    """Translate one alignment. Returns (protein msa, report)."""
    if frame is None:
        frame = best_frame(msa, table_id)
    prot = _translate_all(msa, frame, table_id)

    # A stop shared by most sequences is a gene boundary: drop that column.
    seqs = list(prot.values())
    drop = {i for i in range(len(seqs[0])) if sum(s[i] == "*" for s in seqs) > len(seqs) / 2}
    report = {"frame": frame, "table": table_id, "dropped_stop_columns": sorted(drop),
              "internal_stops": {}}
    cleaned = {}
    for name, s in prot.items():
        kept = "".join(c for i, c in enumerate(s) if i not in drop)
        if kept.count("*"):
            report["internal_stops"][name] = kept.count("*")
        cleaned[name] = kept.replace("*", "X")
    report["length"] = len(next(iter(cleaned.values())))
    return cleaned, report


def _read_alignments(lines: List[str]) -> List[Dict[str, str]]:
    msas, cur = [], None
    for line in lines:
        if _ALN_START.match(line):
            cur = {}
            continue
        if cur is not None:
            s = line.strip()
            if s == "'" or s.startswith("'"):
                msas.append(cur)
                cur = None
            elif s:
                name, seq = s.split(None, 1)
                cur[name] = cur.get(name, "") + seq.replace(" ", "").replace("?", "N")
    return msas


def translate_tgl(src: str, dst: str, tables=(5, 5), frames=(None, None)) -> List[dict]:
    """Translate the HOST and PARASITE alignments (in file order), each with its own
    genetic code and frame. Returns one report per alignment."""
    with open(src) as f:
        lines = f.readlines()
    msas = _read_alignments(lines)
    results = [translate_msa(m, t, fr) for m, t, fr in zip(msas, tables, frames)]
    prot = [r[0] for r in results]

    out, idx, in_aln = [], -1, False
    for line in lines:
        if _ALN_START.match(line):
            idx += 1
            in_aln = True
            out.append(line)
            written = set()
            continue
        if in_aln:
            s = line.strip()
            if s == "'" or s.startswith("'"):
                in_aln = False
                out.append(line)
            elif s:
                name = s.split(None, 1)[0]
                if name not in written:  # interleaved blocks collapse to one line per taxon
                    indent = line[: len(line) - len(line.lstrip())]
                    out.append(f"{indent}{name}  {prot[idx][name]}\n")
                    written.add(name)
            continue
        out.append(line)

    with open(dst, "w") as f:
        f.writelines(out)
    return [r[1] for r in results]


def main(argv=None):
    p = argparse.ArgumentParser(description="Translate a nucleotide .tgl into a protein .tgl.")
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--table", type=int, default=5,
                   help="NCBI genetic code for both sides (1 standard, 2 vertebrate mito, "
                        "5 invertebrate mito). Default 5.")
    p.add_argument("--host-table", type=int, default=None, help="Override --table for the host.")
    p.add_argument("--para-table", type=int, default=None, help="Override --table for the symbiont.")
    p.add_argument("--frame", type=int, choices=(0, 1, 2), default=None,
                   help="Reading frame offset in alignment columns, both sides. Default: fewest stops.")
    args = p.parse_args(argv)
    tables = (args.host_table or args.table, args.para_table or args.table)
    reports = translate_tgl(args.src, args.dst, tables, (args.frame, args.frame))
    for name, r in zip(("host", "parasite"), reports):
        print(f"{name}: genetic code {r['table']}, frame {r['frame']}, protein length {r['length']} "
              f"(Fox reads the first {MAX_SEQ_LEN}), dropped stop-codon columns {r['dropped_stop_columns']}")
        if r["internal_stops"]:
            print(f"  internal stops -> X (frameshift or bad sequence): {r['internal_stops']}")
    print(f"wrote {args.dst}")


if __name__ == "__main__":
    main()
