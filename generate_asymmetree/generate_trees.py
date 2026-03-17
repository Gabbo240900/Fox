import asymmetree.treeevolve as te
from asymmetree.tools.PhyloTreeTools import to_newick
from tralda.datastructures import Tree, LCA
from collections import Counter
import re
import os
import csv 
import random

base_path = "./generated_trees"
num_trees = 200000
time_grid = [1.5, 2, 2.5, 3, 3.5]
species_pattern = re.compile(r'(?<=\(|,)(\d+):')
gene_pattern = re.compile(r'(?<=[\(|,|)])(\d+)(?=(?:<|:))')
assoc_pattern = re.compile(r'<(\d+(?:-\d+)*)>')


event_name_map = {
    "L": "Loss",
    "S": "Speciation",
    "H": "Horizontal Gene Transfer",
    "D": "Duplication"
}


os.makedirs(f"{base_path}/species_trees", exist_ok=True)
os.makedirs(f"{base_path}/gene_trees", exist_ok=True)
os.makedirs(f"{base_path}/associations", exist_ok=True)
os.makedirs(f"{base_path}/reconciliations", exist_ok=True)
os.makedirs(f"{base_path}/Datasets", exist_ok=True)

# High cosp
sim_index = 1
for _ in range(num_trees):
    num_leaves = random.randint(15, 50)
    host_birth_rate = random.uniform(0.5, 1.2)
    host_death_rate = random.uniform(0.2, 0.4) * (host_birth_rate)
    hgt_rate = random.uniform(0.05, 0.3)
    dupl_rate = random.uniform(0.2, 0.4)
    loss_rate = random.uniform(0.2, 0.4)

    for age in time_grid:
        s = te.species_tree_n_age(n=num_leaves, model='BDP', age=age, birth_rate=host_birth_rate, death_rate=host_death_rate)
        g = te.dated_gene_tree(s, dupl_rate=dupl_rate, loss_rate=loss_rate, hgt_rate=hgt_rate)
        s_nwk = to_newick(s)
        g_nwk = to_newick(g)
        
        s_nwk = re.sub(r'(?<=[(,)])(?!(H))(\d+):', r'H\2:', s_nwk)
        g_nwk = gene_pattern.sub(lambda m: f"P{m.group(1)}", g_nwk)
        g_nwk = assoc_pattern.sub(lambda m: "<" + re.sub(r'\d+', lambda n: "H" + n.group(0), m.group(1)) + ">", g_nwk)
        s_path = f"{base_path}/species_trees/species_tree_{sim_index}.nwk"
        g_path = f"{base_path}/gene_trees/gene_tree_{sim_index}.nwk"
        with open(s_path, "w") as f:
            f.write(s_nwk + "\n")

        associations = []
        for match in re.finditer(r'(P\d+)<([^>]+)>', g_nwk):
            p_node = match.group(1)
            h_nodes = match.group(2).split('-')
            for h in h_nodes:
                associations.append((p_node, h))

        # Save associations CSV
        assoc_path = f"{base_path}/associations/associations_{sim_index}.csv"
        with open(assoc_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["parasite_leaf:", "host_leaf"])
            writer.writerows(associations)

        # Remove all angle brackets and their contents from g_nwk before writing
        g_nwk_clean = re.sub(r'<[^>]*>', '', g_nwk)
        with open(g_path, "w") as f:
            f.write(g_nwk_clean + "\n")

        event_counts: Counter[str] = Counter(
            getattr(node, "event")
            for node in g.postorder()
            if getattr(node, "event", None)
        )
        gene_leaf_count = sum(1 for _ in g.leaves())
        speciation_events = event_counts.get("S", 0)
        adjusted_speciation = speciation_events - gene_leaf_count
        total_events = adjusted_speciation + event_counts.get("L", 0) + event_counts.get("H", 0) + event_counts.get("D", 0)
        speciation_proportion = adjusted_speciation / total_events if total_events > 0 else 0
        loss_proportion = event_counts.get("L", 0) / total_events if total_events > 0 else 0
        hgt_proportion = event_counts.get("H", 0) / total_events if total_events > 0 else 0
        dupl_proportion = event_counts.get("D", 0) / total_events if total_events > 0 else 0
        # add proportion to nexus file
        rec_path = f"{base_path}/reconciliations/reconciliation_{sim_index}.txt"
        with open(rec_path, "w") as f:
            for event, count in event_counts.items():
                long_name = event_name_map.get(event, event)
                if event == "S":
                    f.write(f"Speciation: {adjusted_speciation}\n")
                else:
                    f.write(f"{long_name}: {count}\n")
            f.write(f"Total Events: {total_events}\n")
            f.write(f"Speciation_freq: {speciation_proportion:.4f}\n")
            f.write(f"Loss_freq: {loss_proportion:.4f}\n")
            f.write(f"HGT_freq: {hgt_proportion:.4f}\n")
            f.write(f"Duplication_freq: {dupl_proportion:.4f}\n")
            f.write(f"Host_num_leaves: {sum(1 for _ in s.leaves())}\n")
            f.write(f"Symbiont_num_leaves: {gene_leaf_count}\n")
            f.write(f"Sim_time: {age}\n")

        dataset_path = f"{base_path}/Datasets/Dataset{sim_index}.tgl"
        with open(dataset_path, "w") as f:
            f.write("#NEXUS\n")

            # Host tree block
            f.write("BEGIN HOST;\n")
            f.write(f"\tTREE * Host1 = {s_nwk}\n")
            f.write("ENDBLOCK;\n\n")

            # Symbiont tree block
            f.write("BEGIN PARASITE;\n")
            f.write(f"\tTREE * Para1 = {g_nwk_clean}\n")
            f.write("ENDBLOCK;\n\n")

            # Associations block
            f.write("BEGIN DISTRIBUTION;\n")
            f.write('\tRANGE\n')
            for p_node, h_node in associations:
                f.write(f"\t\t{p_node}: {h_node}\n")
            f.write("END;\n\n")

            # Reconciliation block
            for event, count in event_counts.items():
                long_name = event_name_map.get(event, event)
                if event == "S":
                    f.write(f"Speciation: {adjusted_speciation}\n")
                else:
                    f.write(f"{long_name}: {count}\n")
            f.write(f"Total_Events: {total_events}\n")
            f.write(f"Speciation_freq: {speciation_proportion:.4f}\n")
            f.write(f"Loss_freq: {loss_proportion:.4f}\n")
            f.write(f"HGT_freq: {hgt_proportion:.4f}\n")
            f.write(f"Duplication_freq: {dupl_proportion:.4f}\n")
            f.write(f"Host_num_leaves: {sum(1 for _ in s.leaves())}\n")
            f.write(f"Symbiont_num_leaves: {gene_leaf_count}\n")
            f.write(f"Sim_time: {age}\n")
        sim_index += 1

print('Done generating species and gene trees.')


