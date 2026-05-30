import argparse
import os
import pickle
import random
import subprocess
import numpy as np
import re
import tempfile
from Bio import SeqIO
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool

ALPHAS_PATH = os.path.join(os.path.dirname(__file__), "generated_trees", "hogenom_alphas.txt")
MAX_ATTEMPTS_DEFAULT = 20


def load_list(listpath):
    with open(listpath, "rb") as file:
        return pickle.load(file)

def sample_scale(scales):
    mean = random.sample(scales, 1)[0]
    scale = np.random.normal(loc=mean, scale=mean / 10)
    return max(scale, 0.05)

def extract_leaves(tree_str):
    """Extracts all unique leaf names from a Newick tree, including Pxx and Hxx."""
    return set(re.findall(r"[HP]\d+(?:-\d+)?", tree_str))

def parse_tgl_file(tgl_path):
    """Extracts host and parasite trees from a TGL (NEXUS) file."""
    with open(tgl_path, "r") as file:
        content = file.read()

    host_tree_match = re.search(r"BEGIN HOST;.*?TREE \* Host1 = (.+?);.*?ENDBLOCK;", content, re.DOTALL)
    parasite_tree_match = re.search(r"BEGIN PARASITE;.*?TREE \* Para1 = (.+?);.*?ENDBLOCK;", content, re.DOTALL)

    host_tree = host_tree_match.group(1).strip() if host_tree_match else None
    parasite_tree = parasite_tree_match.group(1).strip() if parasite_tree_match else None

    return content, host_tree, parasite_tree

def format_alignment(alignment_str):
    """Formats the alignment to keep correct spacing and indentation."""
    lines = alignment_str.strip().split("\n")
    formatted_lines = []
    
    for line in lines:
        match = re.match(r"^([HP]\d+(?:-\d+)?)\s+(.+)$", line)
        if match:
            formatted_lines.append("\t" + match.group(1) + "  " + match.group(2))

    return "\n".join(formatted_lines)

def simulate_alignment(tree_str, substitution, gamma, binary, custom_model_def, custom_model_args, length, max_attempts, threads, rate, alphas, temp_dir=None):
    """Runs IQ-TREE simulation and generates alignment based on tree."""
    success = False
    attempt = 1

    tree_str = tree_str.strip()
    if not tree_str.startswith("("):    
        tree_str = "(" + tree_str  
    if not tree_str.endswith(";"):
        tree_str += ";"

    # --- MODIFICATION START ---
    # Create the temporary file in the specified directory, if provided
    tree_file = tempfile.NamedTemporaryFile(mode="w", suffix=".nwk", delete=False, dir=temp_dir)
    # --- MODIFICATION END ---
    tree_file_path = tree_file.name
    tree_file.write(tree_str + "\n")
    tree_file.close()

    while not success:
        if attempt > max_attempts:
            os.remove(tree_file_path)
            return f"Simulation failed after {max_attempts} attempts"

        model_args = [substitution]
        if custom_model_args:
            model_args.append(f"+{custom_model_args}")
        if gamma:
            alpha = sample_scale(alphas)
            model_args.append(f"+G{{{alpha}}}")

        output_alignment = tree_file_path.replace(".nwk", ".fa")

        cmd = [
            binary, "--alisim", output_alignment,
            "--length", str(length),
            "--seqtype", "AA",
            "-t", tree_file_path,
            "-m", "+".join(model_args),
            "--threads", str(threads)
        ]
        cmd.extend(['--rate', str(rate)])

        try:
            # Note: IQ-TREE often places its temporary files in the same directory as the input tree file.
            # By placing the input tree (`tree_file_path`) in `temp_dir`, we implicitly control
            # where most of IQ-TREE's related output (like the .phy alignment) will go.
            process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if process.returncode != 0:
                os.remove(tree_file_path)
                return f"Error: {process.stderr.strip()}"

            alignment_file = output_alignment + ".phy"

            if os.path.exists(alignment_file):
                with open(alignment_file, "r") as f:
                    alignment_content = f.read()
                # Clean up the IQ-TREE output file
                os.remove(alignment_file)
            else:
                alignment_content = "Error: Alignment file was not found."

            # Clean up the temporary tree file
            os.remove(tree_file_path)
            return alignment_content

        except subprocess.CalledProcessError as e:
            # Clean up the temporary tree file on error
            os.remove(tree_file_path)
            return f"Error: {e.stderr.strip()}"

        attempt += 1

    # Clean up the temporary tree file if the loop somehow exits unexpectedly
    os.remove(tree_file_path)
    return "Simulation failed"


def process_single_tgl(args_tuple):
    """Worker function to process one TGL file in parallel"""
    tgl_file, substitution, gamma, iqtree, custom_model, length, max_attempts, alphas, temp_dir = args_tuple
    
    try:
        file_content, host_tree, parasite_tree = parse_tgl_file(tgl_file)
        # --- SKIP FILES THAT ALREADY CONTAIN MSAs ---
        if "ALIGNMENT * Host1" in file_content or "ALIGNMENT * Para1" in file_content:
            return f"Skipped {tgl_file}: already contains MSAs"
        if not host_tree or not parasite_tree:
            return f"Skipped {tgl_file}: missing tree"

        # Pass temp_dir to simulate_alignment
        host_alignment = simulate_alignment(
            host_tree,
            substitution,
            gamma,
            iqtree,
            custom_model if custom_model else "",
            "",
            length,
            max_attempts,
            1,  # Single thread per alignment (we parallelize across files)
            rate=1.0,
            alphas=alphas,
            temp_dir=temp_dir # <--- Passed here
        )

        # Pass temp_dir to simulate_alignment
        parasite_alignment = simulate_alignment(
            parasite_tree,
            substitution,
            gamma,
            iqtree,
            custom_model if custom_model else "",
            "",
            length,
            max_attempts,
            1,
            rate=5.0,
            alphas=alphas,
            temp_dir=temp_dir # <--- Passed here
        )

        formatted_host_alignment = format_alignment(host_alignment)
        formatted_parasite_alignment = format_alignment(parasite_alignment)

        updated_content = re.sub(
            r"(BEGIN HOST;\s*TREE \* \w+ = .+?;)",
            r"\1\n\tALIGNMENT * Host1 = '\n" + formatted_host_alignment + "\n\t'",
            file_content,
            flags=re.DOTALL
        )

        updated_content = re.sub(
            r"(BEGIN PARASITE;\s*TREE \* \w+ = .+?;)",
            r"\1\n\tALIGNMENT * Para1 = '\n" + formatted_parasite_alignment + "\n\t'",
            updated_content,
            flags=re.DOTALL
        )

        # Atomic write to avoid partial/corrupted files under heavy I/O
        tmp_path = tgl_file + ".tmp"
        with open(tmp_path, "w") as f:
            f.write(updated_content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, tgl_file)
        
        return f"Success: {tgl_file}"
    
    except Exception as e:
        return f"Error processing {tgl_file}: {str(e)}"


if __name__ == "__main__":
    print('Starting alignment simulation...')
    parser = argparse.ArgumentParser("Alignment simulator for TGL files")
    parser.add_argument("input_dir", type=str, help="Path to the directory containing TGL files")
    parser.add_argument("--length", "-l", default=500, type=int, help="Length of the alignment")
    parser.add_argument("--gamma", "-g", default=None, type=str, help="Gamma model for rate heterogeneity")
    parser.add_argument("--substitution", "-s", default="LG", type=str, help="Substitution model")
    parser.add_argument("--custom-model", "-c", type=str, default=None, help="Path to a custom model definition")
    parser.add_argument("--iqtree", "-t", type=str, required=True, help="Path to IQTree2 binary")
    parser.add_argument("--max-attempts", "-m", default=MAX_ATTEMPTS_DEFAULT, type=int, help="Max attempts for alignment")
    parser.add_argument("--allow-duplicate-sequences", "-d", action="store_true", help="Allow duplicate sequences")
    parser.add_argument("--n_cores", "-n", type=int, default=16, help="Number of parallel processes")
    # --- MODIFICATION START ---
    parser.add_argument("--temp-dir", default=None, type=str, 
                        help="Optional path to a custom directory for temporary files (e.g., IQ-TREE input/output).")
    # --- MODIFICATION END ---

    args = parser.parse_args()
    
    alphas = load_list(ALPHAS_PATH)

    dataset_dir = args.input_dir
    if not os.path.exists(dataset_dir):
        print(f"Error: Directory {dataset_dir} does not exist")
        exit(1)

    # Check and create temp directory if specified
    temp_dir_to_use = None
    if args.temp_dir:
        temp_dir_to_use = os.path.abspath(args.temp_dir)
        if not os.path.exists(temp_dir_to_use):
            print(f"Creating temporary directory: {temp_dir_to_use}")
            os.makedirs(temp_dir_to_use, exist_ok=True)
        else:
            print(f"Using existing temporary directory: {temp_dir_to_use}")


    tgl_files = glob(os.path.join(dataset_dir, "*.tgl"))

    if not tgl_files:
        print(f"Error: No TGL files found in {dataset_dir}")
        exit(1)

    print(f"Found {len(tgl_files)} TGL files to process")
    print(f"Using {args.n_cores} parallel processes")

    # Prepare arguments for all tasks, including the new temp_dir
    tasks = [
        (tgl_file, args.substitution, args.gamma, args.iqtree,
         args.custom_model, args.length, args.max_attempts, alphas, temp_dir_to_use)
        for tgl_file in tgl_files
    ]

    # Process in parallel
    with Pool(processes=args.n_cores, maxtasksperchild=1000) as pool:
        results = list(tqdm(
            pool.imap(process_single_tgl, tasks),
            total=len(tasks),
            desc="Simulating alignments"
        ))

    # Summary
    successes = sum(1 for r in results if r.startswith("Success"))
    # Get the dataset name from the input directory path
    dataset_name = os.path.basename(os.path.normpath(args.input_dir))
    
    # Updated print statement
    print(f"\nCompleted alignment simulation for Dataset '{dataset_name}': {successes}/{len(results)} files processed successfully.")