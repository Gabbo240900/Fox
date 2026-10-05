"""Cockroach / Blattabacterium data -> one .tgl for Fox, like the synthetic test sets.

Source: Arab, Bourguignon, Wang, Ho & Lo (2020) Evolutionary rates are correlated
between cockroach symbionts and mitochondrial genomes. Biol. Lett. 16: 20190702.
Data: Dryad doi:10.5061/dryad.v6wwpzgqw (CC0): Cockroaches_AminoAcid_Outgrp.phy
(13 mitochondrial proteins) and Blattabacterium_AminoAcids_Outgrp.phy (104 genes).

Each of the study's 55 pairs is a cockroach species and the symbiont strain taken
from it, so the links are one-to-one. The two .phy files label the same pair
differently (e.g. Anallacta_methanoides / 57_Anallacta_SC), so the table below maps
each pair name to its label in each file; 3 pairs with no unambiguous label are
dropped (52 left). Outgroups and all-gap columns are removed. Fox takes at most 50
hosts, so 50 of the 52 pairs are used (drawn with seed 1). The .tgl holds the full
alignments; Fox reads the first 250 columns.

Real data: no trees and no labels. Pre-encode with
  python training/pre_encoder.py --src test_data/real_data/blattabacterium/Datasets \
      --dst test_data/real_data/blattabacterium/fox_data --keep-all-leaves

Run from the repository root: python test_data/real_data/blattabacterium/make_tgl.py
"""
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
N_PAIRS = 50


def read_phy(path):
    lines = [l.rstrip("\n") for l in open(path) if l.strip()]
    return {l.split(None, 1)[0]: l.split(None, 1)[1].replace(" ", "").upper() for l in lines[1:]}


SYM_LABEL = {
    "Euphyllodromia": "DNA17_Z257_Euphyllodromia", "Amazonina": "Amazonina",
    "Anaplecta_calosoma": None,              # only 'Anaplecta_annotation2': ambiguous
    "Anaplecta_omei": "Anaplecta_omei2", "CHORI_Chorisoserrata_sp": "CHORI_Chorisoserrata_sp",
    "Allacta": "Allacta", "Anallacta_methanoides": "57_Anallacta_SC",
    "Ectoneura_hanitschi": "Ectoneura_hanitschi", "Ectobius_sp": "DNA18_Z254C_Ectobius",
    "Phyllodromica_sp": "PHILODROMICA_sp", "Lamproblatta": "Lamproblatta",
    "Tryonicus": None,                       # '613_Macrogen1' or '639_Macrogen1': ambiguous
    "Therea_regularis": "Threa_regilaris", "Eupolyphaga_sinensis": "81_Eupolyphaga_sinensis",
    "Ischnoptera_deropeltiformis": "83_Ischnoptera_deropeltiformis", "Panchlora_nivea": "44_Panchlora_nivea",
    "Gyna_capucina": "Z139GY_Gyna_capucina", "Epilampra_maya": "95_Epilampra_maya",
    "Nauphoeta_cinerea": "Nauphoeta_cinerea", "Aeluropoda_insignis": "2_Aeluropoda_insignis",
    "Elliptorhina_davidi": "Elliptorhina_davidi", "Gromphadorhina_grandidieri": "30_Gromphadorhina_grandidieri",
    "Galiblatta_cribrosa": "Galiblatta_cribrosa", "Laxta_sp": "AUS_COCK2_Laxta_sp",
    "Neolaxta_mackerrasae": "Neolaxta_mackerasie", "Macropanesthia_rhinoceros": "DNA5_92_Macropanesthia_rhinoceros",
    "Panesthia_Salganea": "Salganea_sp", "Panesthia_angustipennis": "Panesthia_angustipennis",
    "Blaberus_giganteus": "Blaberus_giganteus", "Blaptica_dubia": "56_Blaptica_dubia",
    "Eublaberus_distanti": "Eublaberus_distanti", "Pycnoscelus_femapterus": "48_Pycnoscelus_femapterus",
    "Paranauphoeta_circumda": "PARA_Paranauphoeta_circumda", "Opisthoplatia": "Opisthoplatia",
    "Rhabdoblatta_sp": "RHA_Rhabdoblatta_sp", "Megaloblatta": "DNA7_ECMD1_Megaloblatta",
    "Carbrunneria_paramaxi": "Carbrunneria", "Beybienkoa_karandanensis": "Beybienkoa_Karandanensis",
    "Blattella_germanica": "Blattella_germanica", "Parcoblatta_virginica": "102_Parcoblatta_virginica",
    "Cryptocercus_hirtus": "CRY_HIR_Cryptocercus_hirtus", "Cryptocercus_punctulatus": "Cryptocercus_punctulatus",
    "Mastotermes_darwiniensis": "Mastotermes", "Deropeltis_paulinoi": "69_Deropeltis_paulinoi",
    "Blatta_orientalis": "Blatta_orientalis",
    "Protagonista_lugubris": None,           # '613_Macrogen1' or '639_Macrogen1': ambiguous
    "Shelfordella_lateralis": "80_Shelfordella_SC", "Periplaneta_americana": "Periplaneta_americana",
    "Eurycotis_decipiens": "71_Eurycotis_decipiens", "Methana_sp": "AUS_COCK1_Methana_sp",
    "Melanozosteria_sp": "DNA19_Melanozosteria", "Platyzosteria_sp": "AUS_COCK3_Platyzosteria",
    "Cosmozosteria": "Cosmozosteria", "Balta_sp": "DNA15_Balta",
    "Paratemnopteryx_couloniana": "61_Paratemnopteryx_couloniana",
}
# pair name -> label in the amino-acid files (None = no unambiguous label; pair dropped)
HOST_LABEL = {n: n for n in SYM_LABEL}
HOST_LABEL.update({"Tryonicus": "Tryonicus_parvus", "Balta_sp": "Balta_sp.", "Laxta_sp": "Laxta_sp.",
                   "Melanozosteria_sp": "Melanozosteria_sp.", "Phyllodromica_sp": "Phyllodromica_sp."})
assert len(SYM_LABEL) == 55

names = [n for n in SYM_LABEL if SYM_LABEL[n] is not None]
assert len(set(SYM_LABEL[n] for n in names)) == len(names)


def strip_gap_columns(msa):
    cols = [i for i in range(len(next(iter(msa.values())))) if any(s[i] not in "-X?" for s in msa.values())]
    return {n: "".join(s[i] for i in cols) for n, s in msa.items()}


host_aa = read_phy(os.path.join(HERE, "Cockroaches_AminoAcid_Outgrp.phy"))
sym_aa = read_phy(os.path.join(HERE, "Blattabacterium_AminoAcids_Outgrp.phy"))
host = strip_gap_columns({n: host_aa[HOST_LABEL[n]] for n in names})
sym = strip_gap_columns({n: sym_aa[SYM_LABEL[n]] for n in names})
pairs = sorted(random.Random(1).sample(names, N_PAIRS))

lines = ["#NEXUS", "BEGIN HOST;", "\tALIGNMENT * Host1 = '"]
lines += [f"\t{n}  {host[n]}" for n in pairs]
lines += ["\t'", "ENDBLOCK;", "", "BEGIN PARASITE;", "\tALIGNMENT * Para1 = '"]
lines += [f"\t{n}  {sym[n]}" for n in pairs]
lines += ["\t'", "ENDBLOCK;", "", "BEGIN DISTRIBUTION;", "\tRANGE"]
lines += [f"\t\t{n}: {n}" for n in pairs]   # host and symbiont of a pair share the pair name
lines += ["END;", ""]

out_dir = os.path.join(HERE, "Datasets")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "blattabacterium.tgl")
with open(out, "w") as f:
    f.write("\n".join(lines))
print(f"{len(names)} pairs (dropped {[n for n in SYM_LABEL if SYM_LABEL[n] is None]}); "
      f"wrote {out} with {N_PAIRS} (left out {sorted(set(names) - set(pairs))}); host alignment "
      f"{len(host[pairs[0]])} aa, symbiont alignment {len(sym[pairs[0]])} aa (Fox reads the first 250)")
