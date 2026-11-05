# Data from: Fox: Predicting rates of host-symbiont co-evolution events with association-aware attention

Dataset DOI: [to be assigned by Dryad]

## Description of the data and file structure

This dataset contains the test data, the real host–symbiont data and the benchmark outputs used to evaluate **Fox**. Fox is a transformer model that estimates the relative frequencies of four cophylogenetic events directly from a pair of multiple sequence alignments (MSAs) and a host–symbiont association list:

- **cospeciation:** the symbiont speciates along with its host
- **host switch:** the symbiont moves to a new host
- **loss:** a symbiont lineage dies out
- **duplication:** the symbiont speciates without its host

There are three groups of data:

1. **AsymmeTree test set:** 100 simulated host/symbiont datasets, held out from training and made with the same simulator Fox was trained on. They are in `Datasets/`, with pre-encoded copies in `fox_data/`.
2. **treeducken test set:** 100 simulated datasets made with an independent simulator (treeducken) that Fox never saw during training. They are in `treeducken/`, with Fox inputs and the outputs of the competing method AmoCoala.
3. **Real data:** 52 cockroach / *Blattabacterium* pairs, with Fox and AmoCoala results. They are in `real_data/blattabacterium/`.

Everything needed to regenerate the figures and tables of the paper from these files is in the Fox code repository (see *Code/Software*).

**Event naming.** The files use the names from the code: `Speciation` means **cospeciation** and `HGT` (or `Horizontal Gene Transfer`) means **host switch**. `Loss` and `Duplication` mean the same in both. The terms *host* and *parasite* in file headers mean host and **symbiont**.

### Files and variables

```
test_data/
├── Datasets/                      # AsymmeTree test set: 100 .tgl files
├── fox_data/                      # The same 100 datasets pre-encoded for Fox (.pt)
├── treeducken/
│   ├── Datasets/                  # treeducken test set: 100 .tgl files
│   ├── fox_data/                  # The same 100 datasets pre-encoded for Fox (.pt)
│   ├── amocoala_3rounds/          # AmoCoala, 3 rounds (results used in the paper)
│   │   ├── DatasetN/              # one folder per dataset (100 folders)
│   │   └── skipped.json           # 8 datasets stopped at the 90-minute time limit
│   └── amocoala_data/             # Earlier AmoCoala run, 1 round (not used in the paper)
│       ├── DatasetN/
│       └── skipped.json
└── real_data/blattabacterium/
    ├── Cockroaches_AminoAcid_Outgrp.phy       # host alignment (from Arab et al. 2020)
    ├── Blattabacterium_AminoAcids_Outgrp.phy  # symbiont alignment (from Arab et al. 2020)
    ├── make_tgl.py                # builds Datasets/blattabacterium.tgl from the two .phy files
    ├── Datasets/blattabacterium.tgl
    ├── fox_data/0000000.pt        # pre-encoded copy (first 250 columns)
    ├── fig1_host.nwk              # host topology, figure 1 of Arab et al. 2020
    ├── fig1_symbiont.nwk          # symbiont topology, figure 1 of Arab et al. 2020
    ├── make_fig1_52.py            # prunes the trees to the 52 pairs -> amocoala/fig1_52.nex
    ├── amocoala/                  # AmoCoala input and outputs (3 rounds)
    ├── fox_full_52.json           # Fox result used in the paper
    └── amocoala_52.json           # AmoCoala result used in the paper
```

#### 1. `.tgl` files (`Datasets/`, `treeducken/Datasets/`, `real_data/blattabacterium/Datasets/`)

These are plain-text files in NEXUS-like format, and each one holds a single dataset. All alignments are protein (amino acids, `-` = gap).

| Block | Content |
|---|---|
| `BEGIN HOST;` | `TREE * Host1 = <Newick>`: the true dated host tree (simulated data only). `ALIGNMENT * Host1 = '...'`: one `name  sequence` row per host leaf. |
| `BEGIN PARASITE;` | The same for the symbiont: `TREE * Para1` and `ALIGNMENT * Para1`. |
| `BEGIN DISTRIBUTION;` / `RANGE` | One `symbiont : host` line per association (symbiont leaf, then the host leaf it lives in). |
| Trailer after `END;` | Ground-truth labels (simulated data only; see below). |

Leaf names in the simulated data are `H<i>` for hosts and `P<j>` for symbionts. Internal nodes are also labelled in the AsymmeTree trees. Both simulated sets have alignments of 250 columns.

**Trailer variables (simulated data only):**

| Variable | Type | Description |
|---|---|---|
| `Speciation`, `Duplication`, `Loss` | integer | Number of cospeciation, duplication and loss events in the true history |
| `Horizontal Gene Transfer` (AsymmeTree) / `HGT` (treeducken) | integer | Number of host switches. In AsymmeTree files this line is left out when the count is 0. |
| `Total_Events` | integer | Sum of the four counts |
| `Speciation_freq`, `HGT_freq`, `Loss_freq`, `Duplication_freq` | float (0–1) | Event count divided by `Total_Events`. These four values are the targets Fox predicts and sum to 1. |
| `Host_num_leaves` | integer | Number of host leaves |
| `Symbiont_num_leaves` | integer | Number of symbiont leaves, including lineages that were lost |
| `Symbiont_num_extant` | integer | Number of symbiont leaves surviving to the present (rows in the symbiont alignment) |
| `Sim_time` | float | Tree age used in the simulation (1.5, 2.0, 2.5, 3.0 or 3.5 time units) |
| `Multi_host_symbionts` (treeducken only) | integer | Number of symbionts living in more than one host |
| `Rates` (treeducken only) | text | Simulation rates: `cosp` = cospeciation, `hbr` / `hdr` = host speciation / extinction without the symbiont (set to 0), `sbr` = symbiont speciation without the host (duplication), `sdr` = symbiont extinction (loss), `her` = host switch |

The real-data `.tgl` file (`blattabacterium.tgl`) has alignments and associations only, with no trees or labels. Each of its 52 associations is one-to-one, and the host and symbiont of a pair share the same name.

#### 2. `.pt` files (`fox_data/`, `treeducken/fox_data/`, `real_data/blattabacterium/fox_data/`)

These are the `.tgl` files pre-encoded for Fox and saved as PyTorch objects (`torch.save`). Load them in Python with `torch.load(path, weights_only=False)`. Each file is a dictionary:

| Key | Type | Description |
|---|---|---|
| `host_msa` | int tensor (n_hosts × L) | Host alignment as tokens: 0–19 = the 20 amino acids, then gap, unknown and padding (vocabulary in `fox/encoding.py`) |
| `para_msa` | int tensor (n_symbionts × L) | Symbiont alignment, encoded the same way |
| `host_dist` | float tensor (n_hosts × n_hosts) | Protein Jukes–Cantor distances between host sequences |
| `para_dist` | float tensor (n_symbionts × n_symbionts) | The same for symbionts |
| `mappings` | list of (int, int) | Associations as (host row index, symbiont row index) |
| `labels` | dict | `Speciation`, `HGT`, `Loss`, `Duplication` (true frequencies) and `Sim_time`. All values are 0 for the real data, which has no labels. |

Files are named `0000000.pt` to `0000099.pt`. The number is the file's position when the `.tgl` file names are sorted alphabetically (`Dataset1`, `Dataset10`, `Dataset100`, `Dataset11`, …), so `0000000.pt` = `Dataset1.tgl`, `0000001.pt` = `Dataset10.tgl`, and so on. Only the first 250 alignment columns are kept. In the simulated sets, symbiont leaves that were lost before the present are removed.

#### 3. AmoCoala files (`treeducken/amocoala_3rounds/`, `treeducken/amocoala_data/`, `real_data/blattabacterium/amocoala/`)

AmoCoala (https://github.com/sinaimeri/AmoCoala) is an approximate Bayesian computation (ABC) method that estimates event probabilities from the host and symbiont trees. Each `DatasetN/` folder contains:

| File | Content |
|---|---|
| `DatasetN.nex` | AmoCoala input: host and symbiont tree topologies (no branch lengths) and associations, converted from `DatasetN.tgl` |
| `DatasetN.nex.simul.round_R.csv` | All parameter vectors simulated in round R (one row per vector, tab-separated, no header) |
| `DatasetN.nex.accep.round_R.csv` | The vectors accepted in round R, in the same format |

Columns 1–4 of both CSV files are the probabilities of **cospeciation, duplication, host switch and loss**, in that order. Columns 5–7 are other AmoCoala outputs that the paper does not use (see the AmoCoala documentation). The paper normalises columns 1–4 of each accepted vector to sum to 1 and reports the mean over the vectors accepted in the last round.

- **Settings for `amocoala_3rounds/` and `real_data/blattabacterium/amocoala/`:** `-N 2000 -M 100 -R 3 -t 0.1,0.25,0.25`. This gives 2000 vectors in round 1, then keeps 200, 50 and 50, so the final estimate is the mean of 50 accepted vectors.
- **`skipped.json`:** a JSON object `{"DatasetN.tgl": seconds}` listing datasets that were stopped at the time limit (5400 s = 90 minutes). For `amocoala_3rounds/`, these are 8 of the 100 datasets, which leaves 92 for the comparison in the paper. Their folders contain only partial output.
- **`amocoala_data/`:** an earlier 1-round run on the treeducken set. It is kept for completeness but is not used in the paper.
- **`real_data/blattabacterium/amocoala/`:** contains the input `fig1_52.nex` and its outputs from 3 rounds.

#### 4. Real data: cockroaches and *Blattabacterium* (`real_data/blattabacterium/`)

- **`Cockroaches_AminoAcid_Outgrp.phy`, `Blattabacterium_AminoAcids_Outgrp.phy`:** relaxed-PHYLIP amino-acid alignments of 13 mitochondrial proteins (host; 67 taxa, 3,751 columns) and 104 symbiont genes (62 taxa, 35,729 columns), including outgroups. They are copied unchanged from Arab et al. (2020); see *Access information*.
- **`make_tgl.py`:** matches the pairs between the two files, which use different labels for the same pair. It drops the outgroups, all-gap columns and 3 pairs with no unambiguous label, then writes the remaining 52 pairs with full alignments to `Datasets/blattabacterium.tgl` (3,736 host and 35,310 symbiont columns).
- **`fig1_host.nwk`, `fig1_symbiont.nwk`:** Newick topologies of the host and symbiont trees in figure 1 of Arab et al. (2020), transcribed by hand. They contain all 55 pairs and have no branch lengths.
- **`make_fig1_52.py`:** prunes these trees to the same 52 pairs and collapses nodes left with one child, then writes the AmoCoala input `amocoala/fig1_52.nex`. In that file, host leaves are prefixed `H_` and symbiont leaves `P_`.
- **`fox_full_52.json`:** Fox prediction on all 52 pairs with the full alignments. Fields: `description`, `date`, `input`, `pairs`, `host_columns`, `symbiont_columns`, `command`, `environment`, `time_seconds`, `peak_memory_gb`, and `predictions` (the four event frequencies).
- **`amocoala_52.json`:** AmoCoala result on `fig1_52.nex` (round 3). Fields: `description`, `settings`, `accepted_vectors` (50), and `mean` and `sd` for each event (population standard deviation over the normalised accepted vectors).

Run both scripts from the root of the Fox code repository, e.g. `python test_data/real_data/blattabacterium/make_tgl.py`.

## Methods

**AsymmeTree test set (`Datasets/`).** Host trees were simulated under a birth–death process and symbiont trees as dated gene trees with duplication, host switch (horizontal transfer) and loss, using AsymmeTree (`generate_data/generate_trees.py` in the code repository). Settings: 15–50 host leaves; host birth rate 0.5–1.2, with death rate 0.2–0.4 × the birth rate; duplication 0.2–0.4; host switch 0.05–0.3; loss 0.2–0.4; tree ages 1.5–3.5. Each age was used for 20 of the 100 datasets. Protein alignments of 250 columns were then simulated along both trees with AliSim (IQ-TREE 2; `generate_data/alisim.py`). These datasets come from the same simulator as the training data but were not used for training.

**treeducken test set (`treeducken/Datasets/`).** These were simulated with treeducken (Dismukes & Heath 2021), using a fork that corrects how host switches are recorded (https://github.com/Gabbo240900/treeducken), so that Fox is tested on a simulator it was not trained on. Rates were drawn from the same ranges as the training data. Host switches use `hs_mode = "switch"`, and extinct symbiont lineages were pruned. Event counts follow the AsymmeTree convention. Commands:

```
Rscript generate_data/treeducken_testset.R out=test_data/treeducken/Datasets n=100 seed=2026
python generate_data/alisim.py test_data/treeducken/Datasets --iqtree <iqtree2 binary> -l 250 -g yes
python training/pre_encoder.py --src test_data/treeducken/Datasets --dst test_data/treeducken/fox_data
```

**Pre-encoding.** All `.pt` files were made with `training/pre_encoder.py`. The real data used `--keep-all-leaves`, because it has no dated trees from which to remove lost leaves.

**AmoCoala.** Each simulated `.tgl` was converted to AmoCoala's NEXUS input by removing branch lengths and internal node labels. AmoCoala was run with the settings above and a time limit for each dataset. The conversion and run code is in `test_model.ipynb` in the code repository.

## Code/Software

- **Fox (code, pretrained model, analysis notebook):** https://github.com/Gabbo240900/Fox, archived on Zenodo at https://doi.org/10.5281/zenodo.23815278. The notebook `test_model.ipynb` reproduces every result in the paper from this dataset. To use it, place the folder in the repository as `test_data/`.
- **Requirements:** Python ≥ 3.11 and PyTorch (to read `.pt` files and run Fox). The `.tgl`, `.nex`, `.csv`, `.json`, `.nwk` and `.phy` files are plain text and can be opened with any text editor.
- **Other software used:** AsymmeTree, treeducken (fork above), IQ-TREE 2 / AliSim, and AmoCoala (https://github.com/sinaimeri/AmoCoala).

Quick example:

```bash
pip install -e .   # from the Fox repository root
fox predict --tgl test_data/Datasets/Dataset11.tgl --simulated
```

## Access information

Other publicly accessible locations of the data:

- A copy of this folder is also in the Fox GitHub repository: https://github.com/Gabbo240900/Fox/tree/main/test_data

Data was derived from the following sources:

- The cockroach and *Blattabacterium* alignments (`*.phy`) and the tree topologies transcribed in `fig1_*.nwk` come from: Arab, D. A., Bourguignon, T., Wang, Z., Ho, S. Y. W., & Lo, N. (2020). Evolutionary rates are correlated between cockroach symbionts and mitochondrial genomes. *Biology Letters*, 16, 20190702. Data: Dryad, https://doi.org/10.5061/dryad.v6wwpzgqw (CC0).
- All other data was simulated by the authors.

## Citation

Di Palma G, Jacob L, Matias C, Sinaimeri B. Fox: Predicting rates of host-symbiont co-evolution events with association-aware attention. *bioRxiv* (2026). https://doi.org/10.64898/2026.10.29.762770
