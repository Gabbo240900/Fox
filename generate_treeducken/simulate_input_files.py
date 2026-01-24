import argparse
import subprocess
import os
import glob
import re 
import pandas as pd
import random
from ete3 import Tree
from tqdm import tqdm
from multiprocessing import Pool
import functools

class GenerateTGLFiles:
    def __init__(self, h_lambda, h_mu, c_lambda, s_lambda, s_mu, s_her, time_to_sim, num_trees,numbsim=1):
        self.h_lambda = h_lambda
        self.h_mu = h_mu
        self.c_lambda = c_lambda
        self.s_lambda = s_lambda
        self.s_mu = s_mu
        self.s_her = s_her
        self.time_to_sim = time_to_sim
        self.num_trees = num_trees
        self.numbsim = numbsim

    def run_r_script(self, sim_index):
        """Run the R script with parameters."""
        args = [
            "Rscript",
            '/lustre/fswork/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/treeducken.r',
            str(self.h_lambda),
            str(self.h_mu),
            str(self.c_lambda),
            str(self.s_lambda),
            str(self.s_mu),
            str(self.s_her),
            str(self.time_to_sim),
            str(sim_index)
        ]
        try:
            subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"Error running R script: {e}")
    
    def generate_tgl_files(self, sim_index):
        # Find the most recent files for this simulation
        host_file = f"/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/host_tree/host_tree_{sim_index}.nwk"
        symb_file = f"/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/symb_tree/symb_tree_{sim_index}.nwk"
        assoc_file = f"/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/associations/association_{sim_index}.csv"
        summary_file = f"/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/summaries/summary_{sim_index}.csv"
        
        # host_file = f"/Users/gabriele/Co-phyloformer/generate_treeducken/generated_trees/host_tree/host_tree_{sim_index}.nwk"
        # symb_file = f"/Users/gabriele/Co-phyloformer/generate_treeducken/generated_trees/symb_tree/symb_tree_{sim_index}.nwk"
        # assoc_file = f"/Users/gabriele/Co-phyloformer/generate_treeducken/generated_trees/associations/association_{sim_index}.csv"
        # summary_file = f"/Users/gabriele/Co-phyloformer/generate_treeducken/generated_trees/summaries/summary_{sim_index}.csv"

        if not all(os.path.exists(f) for f in [host_file, symb_file, assoc_file, summary_file]):
            print(f"[WARN] One or more files missing for sim_index {sim_index}, skipping TGL generation.")
            return
        
        with open(host_file) as f:
            host_tree = f.read().strip() 
        host_tree = host_tree.replace('X', '')
        with open(symb_file) as f:
            symb_tree = f.read().strip()
        # Replace 'X' with 'P' and 'S' with 'P' in symb_tree
        symb_tree = symb_tree.replace('X', '')
        symb_tree = symb_tree.replace('S', 'P')
        assoc_df = pd.read_csv(assoc_file)
        # Replace 'S' with 'P' in symbiont and host columns
        assoc_df['symbiont'] = assoc_df['symbiont'].str.replace('S', 'P')
        assoc_df['host'] = assoc_df['host'].str.replace('S', 'P')
        assoc_lines = [f'\t\t{row["symbiont"]}: {row["host"]}' for _, row in assoc_df.iterrows()]
        assoc_block = "\n".join(assoc_lines).rstrip(',')
        summary_df = pd.read_csv(summary_file, index_col=0).T.iloc[0]

        # Normalize event counts to proportions
        event_keys = [
            "Cospeciations",
            "Host_Spread/Switches",
            "Symbiont_Speciations",
            'Host_Speciations',
            "Host_Extinctions",
            "Symbiont_Extinctions"
        ]
        total_events = summary_df[event_keys].sum()
        for key in event_keys:
            summary_df[key] = summary_df[key] / total_events
            if summary_df[key] > 1.0:
                print(f"[WARNING] Normalized values exceed 1 for sim_index {sim_index}")

        host_leaf_count = len(Tree(host_tree, format=1).get_leaves())
        symb_leaf_count = len(Tree(symb_tree, format=1).get_leaves())

        summary_df["Host_num_leaves"] = host_leaf_count
        summary_df["Parasite_num_leaves"] = symb_leaf_count
        summary_df["Sim_time"] = self.time_to_sim
        summary_df['Total Events'] = total_events

        content = f"""#NEXUS
BEGIN HOST;
    TREE * Host1 = {host_tree};
ENDBLOCK;

BEGIN PARASITE;
    TREE * Para1 = {symb_tree};
ENDBLOCK;

BEGIN DISTRIBUTION;
    RANGE
{assoc_block}
    ;
END;

{summary_df.to_string(index=True, header=False)}
"""
        out_dir = '/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/Datasets'
        #out_dir = '/Users/gabriele/Co-phyloformer/generate_treeducken/generated_trees/Datasets'
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join('/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/Datasets', f'Dataset{sim_index}.tgl')
        #out_path = os.path.join('/Users/gabriele/Co-phyloformer/generate_treeducken/generated_trees/Datasets', f'Dataset{sim_index}.tgl')
        with open(out_path, 'w') as out_file:
            out_file.write(content)

def process_single_simulation(params):
    """Worker function for parallel processing."""
    h_lambda, h_mu, c_lambda, s_lambda, s_mu, s_her, time_to_sim, sim_index = params
    
    generator = GenerateTGLFiles(
        h_lambda=h_lambda,
        h_mu=h_mu,
        c_lambda=c_lambda,
        s_lambda=s_lambda,
        s_mu=s_mu,
        s_her=s_her,
        time_to_sim=time_to_sim,
        num_trees=1,  # Not used in the actual generation
        numbsim=1
    )
    generator.run_r_script(sim_index)
    generator.generate_tgl_files(sim_index)
    return sim_index

def main():
    parser = argparse.ArgumentParser(description="Run Treeducken simulation and generate outputs.")
    parser.add_argument("--h_lambda", type=float, nargs=2, metavar=('MIN', 'MAX'), required=True)
    parser.add_argument("--c_lambda", type=float, nargs=2, metavar=('MIN', 'MAX'), required=True)
    parser.add_argument("--s_lambda", type=float, nargs=2, metavar=('MIN', 'MAX'), required=True)
    parser.add_argument("--s_her", type=float, nargs=2, metavar=('MIN', 'MAX'), required=True)
    parser.add_argument("--num_trees", type=int, required=True)
    parser.add_argument("--n_cores", type=int, default=16, help="Number of parallel processes")

    args = parser.parse_args()

    time_grid = [1.5, 2, 2.5, 3, 3.5]
    
    # Prepare all parameter combinations upfront
    all_params = []
    sim_index = 1
    
    for _ in range(args.num_trees):
        h_lambda = random.uniform(*args.h_lambda)
        c_lambda = random.uniform(*args.c_lambda)
        s_lambda = random.uniform(*args.s_lambda)
        s_her = random.uniform(*args.s_her)

        h_mu = random.uniform(0.2, 0.4) * (h_lambda + c_lambda)
        s_mu = random.uniform(0.2, 0.4) * (s_lambda + c_lambda + s_her)

        for time_to_sim in time_grid:
            all_params.append((h_lambda, h_mu, c_lambda, s_lambda, s_mu, s_her, time_to_sim, sim_index))
            sim_index += 1
    
    # Process in parallel using all available cores
    print(f"Processing {len(all_params)} simulations using {args.n_cores} cores...")
    with Pool(args.n_cores) as pool:
        results = list(tqdm(
            pool.imap(process_single_simulation, all_params),
            total=len(all_params),
            desc="Simulating trees"
        ))
    
    print(f"Completed {len(results)} simulations")

if __name__ == "__main__":
    main()