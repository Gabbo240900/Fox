import os
import random
import argparse
import pandas as pd
import re
import subprocess
import shutil
from ete3 import Tree
from dendropy.simulate import treesim
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
from io import StringIO
from Bio import Phylo

# change host tree generation with birth death model 
def run_tgl_generator(input_file, dataset_name, pc_value, ps_value, pd_value, output_dir, jar_path):
    command = [
        "java", "-jar", jar_path,
        "-i", input_file,
        "-p", dataset_name,
        "-n", "1",
        "-pc", str(round(pc_value, 5)),
        "-ps", str(round(ps_value, 5)),
        "-pd", str(round(pd_value, 5))
    ]
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, cwd=output_dir)
    return input_file, dataset_name, result.returncode

class GenerateHostTree:
    """Class to generate host trees and their corresponding event frequency files."""
    
    def __init__(self, num_trees, min_leaves, max_leaves, output_dir, 
                 cospeciation_range=(50, 100)):
        self.num_trees = num_trees  
        self.min_leaves = min_leaves  
        self.max_leaves = max_leaves  
        self.output_dir = output_dir  

        # Define host tree and frequency directories
        self.host_tree_dir = os.path.join(output_dir, "host_trees")
        self.freq_dir = os.path.join(output_dir, "frequencies")

        # Ensure output directories exist
        os.makedirs(self.host_tree_dir, exist_ok=True)
        os.makedirs(self.freq_dir, exist_ok=True)

        # Define frequency ranges
        self.cospeciation_range = cospeciation_range

    def generate_random_tree(self, num_leaves, prefix="H"):
        """Generates a host tree using a birth-death process with given number of leaves."""
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".nwk")
        t = treesim.birth_death_tree(birth_rate=1.0, death_rate=0.5, num_extant_tips=num_leaves)
        t.write(path=tmp_file.name, schema="newick", suppress_rooting=True)
        tmp_file.close()

        tree = Tree(tmp_file.name)

        # Rename leaves to follow prefix
        for idx, leaf in enumerate(tree.iter_leaves()):
            leaf.name = f"{prefix}{idx+1}"

        return tree

    def save_tree(self, tree, filename):
        """Saves the generated tree in Newick format."""
        tree_path = os.path.join(self.host_tree_dir, filename)
        tree.write(outfile=tree_path, format=9)
        dir_no_branches = './generated_trees/branches_lengths'
        os.makedirs(dir_no_branches, exist_ok=True)
        tree_path2 = os.path.join(dir_no_branches, filename)
        tree.write(outfile=tree_path2, format=5)

    def generate_cophylo_frequencies(self):
        while True:
            cospeciation = random.randint(self.cospeciation_range[0], self.cospeciation_range[1])
            remaining = 100 - cospeciation
            dirichlet_sample = np.random.dirichlet([1, 1, 1])
            scaled = [int(x * remaining) for x in dirichlet_sample]
            # Adjust the last value to ensure the total adds up
            scaled[-1] = remaining - sum(scaled[:-1])
            duplication, loss, switch = scaled
            return cospeciation, loss, switch, duplication

    def generate_and_save_trees(self):
        """Generates host trees and their event frequency files."""
        for i in range(self.num_trees):
            num_leaves = random.randint(self.min_leaves, self.max_leaves)  
            tree = self.generate_random_tree(num_leaves)  
            tree_filename = f"host_tree_{i + 1}.nwk"
            self.save_tree(tree, tree_filename)  

            cospeciation, loss, switch, duplication = self.generate_cophylo_frequencies()

            freq_filename = f"host_tree_{i + 1}_frequencies.csv"
            freq_filepath = os.path.join(self.freq_dir, freq_filename)
            df = pd.DataFrame({
                "Event": ["Cospeciation", "Loss", "Switch", "Duplication"],
                "Frequency (%)": [cospeciation, loss, switch, duplication]
            })
            df.to_csv(freq_filepath, index=False)

        print(f"Generated {self.num_trees} host trees in '{self.host_tree_dir}'.")
        print(f"Frequency files stored in '{self.freq_dir}'.")

    def get_directories(self):
        """Returns the host tree and frequency directories."""
        return self.host_tree_dir, self.freq_dir


class GenerateParasiteTree:
    """Class to generate and post-process parasite trees using Coala's TGLGenerator.jar."""

    def __init__(self, host_dir, freq_dir, output_dir_tgl, jar_path):
        self.host_dir = host_dir
        self.freq_dir = freq_dir
        self.output_dir_tgl = output_dir_tgl
        self.jar_path = jar_path

        # Ensure output directory exists
        os.makedirs(self.output_dir_tgl, exist_ok=True)

    def generate_tgl_files(self):
        pattern = re.compile(r"(\d+)")
        futures = []

        max_workers = args.num_threads if args.num_threads else max(2, int(os.cpu_count() * 0.75))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for filename in sorted(os.listdir(self.host_dir)):
                if filename.endswith(".nwk"):
                    input_file = os.path.join(self.host_dir, filename)
                    input_file = os.path.abspath(input_file)

                    match = pattern.search(filename)
                    dataset_number = match.group(1) if match else "1"
                    dataset_name = f"Dataset{dataset_number}"

                    csv_filename = f"host_tree_{dataset_number}_frequencies.csv"
                    csv_path = os.path.join(self.freq_dir, csv_filename)

                    pc_value = ps_value = pd_value = 0.1
                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)
                        freq_dict = dict(zip(df["Event"], df["Frequency (%)"]))
                        pc_value = freq_dict.get("Cospeciation", 0.0) / 100
                        ps_value = freq_dict.get("Switch", 0.0) / 100
                        pd_value = freq_dict.get("Duplication", 0.0) / 100
                        pl_value = max(0.0, 1.0 - (pc_value + ps_value + pd_value))
                        total_prob = pc_value + ps_value + pd_value + pl_value
                        if total_prob > 0:
                            pc_value /= total_prob
                            ps_value /= total_prob
                            pd_value /= total_prob

                    futures.append(executor.submit(
                        run_tgl_generator, input_file, dataset_name, pc_value, ps_value, pd_value, self.output_dir_tgl, self.jar_path
                    ))

            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating TGLs"):
                input_file, dataset_name, return_code = future.result()
                expected_tgl_filename = f"{dataset_name}-1.tgl"
                generated_tgl_file = os.path.join(self.host_dir, expected_tgl_filename)
                correct_tgl_file = os.path.join(self.output_dir_tgl, expected_tgl_filename)

                if return_code != 0:
                    print(f"Error generating TGL for {input_file}")
                elif os.path.exists(generated_tgl_file):
                    shutil.move(generated_tgl_file, correct_tgl_file)
                    print(f"Moved {generated_tgl_file} → {correct_tgl_file}")
                else:
                    print(f"Warning: {generated_tgl_file} not found, skipping move.")

    def post_process_tgl_files(self):

        def rename_parasite_leaves(content):
            matches = re.findall(r"(H\d+-\d+)", content)
            parasite_map = {h: f"P{i+1}" for i, h in enumerate(sorted(set(matches)))}

            def replace_mapping(match):
                return parasite_map.get(match.group(), match.group())

            updated_content = re.sub(r"H\d+-\d+", replace_mapping, content)
            return updated_content, parasite_map

        def rename_distribution_section(content, parasite_map):
            def replace_mapping(match):
                parasite, host = match.groups()
                return f"\t{parasite_map.get(parasite, parasite)}: {host}"
            return re.sub(r"(H\d+-\d+)\s*:\s*(H\d+)", replace_mapping, content)

        def parse_tgl_file(tgl_path):
            with open(tgl_path, "r") as file:
                content = file.read()

            content, parasite_map = rename_parasite_leaves(content)
            content = rename_distribution_section(content, parasite_map)
            return content

        for tgl_file in os.listdir(self.output_dir_tgl):
            if tgl_file.endswith(".tgl"):
                tgl_path = os.path.join(self.output_dir_tgl, tgl_file)
                updated_content = parse_tgl_file(tgl_path)
                with open(tgl_path, "w") as file:
                    file.write(updated_content)
                print(f"Processed: {tgl_file}")


class InferParasiteBranches:
    def __init__(self, datasets_dir, hosts_dir, output_dir):
        self.datasets_dir = datasets_dir
        self.hosts_dir = hosts_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def parse_nexus_tgl(tgl_text):
        host_tree_str = re.search(r"BEGIN HOST;.*?TREE \* \w+ = (.*?);.*?ENDBLOCK;", tgl_text, re.DOTALL).group(1)
        parasite_tree_str = re.search(r"BEGIN PARASITE;.*?TREE \* \w+ = (.*?);.*?ENDBLOCK;", tgl_text, re.DOTALL).group(1)
        distribution_block = re.search(r"BEGIN DISTRIBUTION;.*?RANGE(.*?)END;", tgl_text, re.DOTALL).group(1)
        mapping = dict(re.findall(r"P(\d+): H(\d+)", distribution_block))
        return host_tree_str, parasite_tree_str, mapping

    @staticmethod
    def get_node_times(tree):
        depths = tree.depths()
        max_depth = max(depths.values())
        return {clade: max_depth - depth for clade, depth in depths.items()}

    @staticmethod
    def infer_parasite_times(p_tree, host_times, mapping):
        times = {}
        for clade in p_tree.get_terminals():
            pid = clade.name
            if pid.startswith("P") and pid[1:] in mapping:
                host_id = f"H{mapping[pid[1:]]}"
                host_time = next((t for c, t in host_times.items() if c.name == host_id), 0)
                times[clade] = host_time + random.uniform(0.01, 0.05)
            else:
                times[clade] = 0.01
        def set_internal_times(clade):
            if clade in times:
                return times[clade]
            child_times = [set_internal_times(c) for c in clade.clades]
            times[clade] = max(child_times) + random.uniform(0.05, 0.1)
            return times[clade]
        set_internal_times(p_tree.root)
        return times

    @staticmethod
    def assign_branch_lengths(tree, node_times):
        """Assign branch lengths to tree nodes using node times."""
        for clade in tree.find_clades(order="postorder"):
            if clade is tree.root:
                clade.branch_length = 0.0
            else:
                # Safely find parent
                parent = next((c for c in tree.find_clades() if clade in c.clades), None)
                if parent is None:
                    print(f"Warning: no parent found for clade {clade.name}, skipping.")
                    continue
                clade.branch_length = max(0.001, node_times[parent] - node_times[clade])
        return tree

    def replace_trees_in_tgl(self, tgl_path, host_tree_newick, parasite_tree_newick):
        with open(tgl_path, "r") as f:
            content = f.read()

        new_host_block = f"BEGIN HOST;\n\tTREE * Host1 = {host_tree_newick};\nENDBLOCK;"
        new_parasite_block = f"BEGIN PARASITE;\n\tTREE * Para1 = {parasite_tree_newick};\nENDBLOCK;"

        content = re.sub(r"BEGIN HOST;.*?ENDBLOCK;", new_host_block, content, flags=re.DOTALL)
        content = re.sub(r"BEGIN PARASITE;.*?ENDBLOCK;", new_parasite_block, content, flags=re.DOTALL)

        with open(tgl_path, "w") as f:
            f.write(content)

    def run(self):
        for file in os.listdir(self.datasets_dir):
            if file.endswith(".tgl"):
                tgl_path = os.path.join(self.datasets_dir, file)
                match = re.search(r'Dataset(\d+)-', file)
                if not match:
                    continue
                host_file = f"host_tree_{match.group(1)}.nwk"
                host_path = os.path.join(self.hosts_dir, host_file)

                with open(tgl_path, "r") as f:
                    nexus_text = f.read()

                h_str, p_str, mapping = self.parse_nexus_tgl(nexus_text)
                host_tree = Phylo.read(host_path, "newick")
                parasite_tree = Phylo.read(StringIO(p_str + ";"), "newick")

                host_node_times = self.get_node_times(host_tree)
                parasite_node_times = self.infer_parasite_times(parasite_tree, host_node_times, mapping)
                parasite_tree = self.assign_branch_lengths(parasite_tree, parasite_node_times)

                output_path = os.path.join(self.output_dir, f"parasite_with_lengths_{file.replace('.tgl','.nwk')}")
                Phylo.write(parasite_tree, output_path, "newick")
                print(f"Saved: {output_path}")
                parasite_newick_str = p_str + ";"
                host_newick_str = h_str + ";"

                buf = StringIO()
                Phylo.write(host_tree, buf, "newick")
                host_newick = buf.getvalue().strip()
                if host_newick.endswith(";"):
                    host_newick = host_newick[:-1]
                buf = StringIO()
                Phylo.write(parasite_tree, buf, "newick")
                parasite_newick = buf.getvalue().strip()
                if parasite_newick.endswith(";"):
                    parasite_newick = parasite_newick[:-1]
                self.replace_trees_in_tgl(tgl_path, host_newick, parasite_newick)



if __name__ == "__main__":
    import time
    parser = argparse.ArgumentParser(description="Generate host and parasite trees")
    parser.add_argument("--num_trees", type=int, required=True, help="Number of trees to generate")
    parser.add_argument("--min_leaves", type=int, default=15, help="Minimum number of leaves per tree")
    parser.add_argument("--max_leaves", type=int, default=30, help="Maximum number of leaves per tree")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for host trees and frequencies")
    parser.add_argument("--output_dir_tgl", type=str, required=True, help="Output directory for generated TGL files")
    parser.add_argument("--jar_path", type=str, required=True, help="Path to TGLGenerator.jar")
    parser.add_argument("--num_threads", type=int, default=None, help="Number of threads for parallel processing (default: 0.75 of CPUs)")
    args = parser.parse_args()

    total_start = time.time()

    # Generate Host Trees
    host_generator = GenerateHostTree(args.num_trees, args.min_leaves, args.max_leaves, args.output_dir)
    start = time.time()
    if not os.listdir(os.path.join(args.output_dir, "host_trees")):
        host_generator.generate_and_save_trees()
    else:
        print("Host trees already exist, skipping generation.")
    print(f"Host tree generation took {time.time() - start:.2f} seconds")

    host_dir, freq_dir = host_generator.get_directories()
    args.jar_path = os.path.abspath(args.jar_path)

    # Generate Parasite Trees
    parasite_generator = GenerateParasiteTree(host_dir, freq_dir, args.output_dir_tgl, args.jar_path)
    
    start = time.time()
    parasite_generator.generate_tgl_files()
    print(f"[TGL generation took {time.time() - start:.2f} seconds")

    start = time.time()
    parasite_generator.post_process_tgl_files()
    print(f"TGL post-processing took {time.time() - start:.2f} seconds")

    print(f"Total execution time: {time.time() - total_start:.2f} seconds")

    infer_branches = InferParasiteBranches(
        datasets_dir="./generated_trees/Datasets",
        hosts_dir="./generated_trees/branches_lengths",
        output_dir="./generated_trees/output_parasite_with_lengths"
    )
    infer_branches.run()