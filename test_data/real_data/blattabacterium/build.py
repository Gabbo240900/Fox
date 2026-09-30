"""Blattabacterium / cockroach amino-acid data.

Source: Arab, Bourguignon, Wang, Ho & Lo (2020) Evolutionary rates are correlated
between cockroach symbionts and mitochondrial genomes. Biol. Lett. 16: 20190702.
Data: Dryad doi:10.5061/dryad.v6wwpzgqw (CC0): Cockroaches_AminoAcid_Outgrp.phy
(13 mitochondrial proteins) and Blattabacterium_AminoAcids_Outgrp.phy (104 genes).

Maps the labels of the amino-acid alignments to the 55 cockroach/symbiont pairs
of the study (names as in the authors' NoOutgrp files), removes outgroups and
all-gap columns, and writes pairs.json: {host: {name: seq}, sym: {name: seq},
links: [[host, sym], ...]}; host and symbiont of a pair share the name.
Run from this folder: python build.py
"""
import json


def read_phy(path):
    lines = [l.rstrip("\n") for l in open(path) if l.strip()]
    return {l.split(None, 1)[0]: l.split(None, 1)[1].replace(" ", "").upper() for l in lines[1:]}


host_aa = read_phy("Cockroaches_AminoAcid_Outgrp.phy")
sym_aa = read_phy("Blattabacterium_AminoAcids_Outgrp.phy")
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


host = strip_gap_columns({n: host_aa[HOST_LABEL[n]] for n in names})
sym = strip_gap_columns({n: sym_aa[SYM_LABEL[n]] for n in names})
json.dump({"host": host, "sym": sym, "links": [[n, n] for n in names]}, open("pairs.json", "w"))
print(f"{len(names)} pairs; host alignment {len(next(iter(host.values())))} aa, "
      f"symbiont alignment {len(next(iter(sym.values())))} aa; dropped "
      f"{[n for n in SYM_LABEL if SYM_LABEL[n] is None]}")
