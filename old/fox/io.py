"""Input reader: well-formatted .tgl bundle (host MSA + symbiont MSA + mapping).

No external deps. Keeps the install light.
"""

import re
from typing import Dict, List, Tuple

# Canonical labels found in a .tgl. The event frequencies are written as
# "<Event>_freq" (the bare "<Event>" lines are integer counts, not frequencies).
# These labels live in a trailer after the final "END;", so they are parsed at
# top level regardless of the current block.
_FREQ_KEYS = {
    "Speciation_freq": "Speciation",
    "HGT_freq": "HGT",
    "Loss_freq": "Loss",
    "Duplication_freq": "Duplication",
    "Sim_time": "Sim_time",
}


def read_tgl(path: str) -> dict:
    """Parse a well-formatted .tgl bundle.

    Returns a dict with:
      host_msa : {name: sequence}   (HOST ALIGNMENT block)
      sym_msa  : {name: sequence}   (PARASITE ALIGNMENT block)
      mapping  : [(host_name, symbiont_name), ...]  (DISTRIBUTION block;
                 stored there as 'parasite : host', returned host-first)
      labels   : {event: float}  ground-truth freqs + Sim_time, if present
      sim_time : float or None
    """
    host_msa: Dict[str, str] = {}
    sym_msa: Dict[str, str] = {}
    mapping_ps: List[Tuple[str, str]] = []  # (parasite, host) as written
    labels: Dict[str, float] = {}
    section = None
    in_aln = False

    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("BEGIN HOST;"):
                section, in_aln = "HOST", False
                continue
            if s.startswith("BEGIN PARASITE;"):
                section, in_aln = "PARA", False
                continue
            if s.startswith("BEGIN DISTRIBUTION;"):
                section, in_aln = "DIST", False
                continue
            if s in ("ENDBLOCK;", "END;"):
                section, in_aln = None, False
                continue

            if section in ("HOST", "PARA"):
                if re.match(r"TREE \* \S+ = ", s):
                    continue
                if s.startswith("ALIGNMENT"):
                    in_aln = True
                    continue
                if in_aln and s == "'":
                    in_aln = False
                    continue
                if in_aln:
                    parts = s.split()
                    if len(parts) == 2:
                        d = host_msa if section == "HOST" else sym_msa
                        d[parts[0]] = parts[1]
            elif ":" in s:
                # Label keys (incl. the post-END; trailer) are parsed in any
                # section; mapping pairs only count inside the DIST block.
                left, right = (x.strip() for x in s.split(":", 1))
                std = _FREQ_KEYS.get(left)
                if std is not None:
                    try:
                        labels[std] = float(right)
                    except ValueError:
                        pass
                elif section == "DIST" and left and right:
                    mapping_ps.append((left, right))  # parasite, host

    if not host_msa or not sym_msa:
        raise ValueError(
            f"{path!r}: missing HOST or PARASITE alignment. "
            "Expected a well-formatted .tgl with both ALIGNMENT blocks."
        )

    mapping = [(h, p) for p, h in mapping_ps]  # return host-first
    return {
        "host_msa": host_msa,
        "sym_msa": sym_msa,
        "mapping": mapping,
        "labels": labels,
        "sim_time": labels.get("Sim_time"),
    }
