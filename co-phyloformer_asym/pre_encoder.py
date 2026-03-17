import torch, os, glob
from data import CophylogenyDataset   
from tqdm import tqdm

src_dir = '/Users/gabriele/Co-phyloformer/generate_asymmetree/generated_trees/Datasets'
dst_dir = '/Users/gabriele/Co-phyloformer/generate_asymmetree/generated_trees/test/'


# src_dir = "/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_asymmetree/generated_trees/Dataset_final/"
# dst_dir = "/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_asymmetree/generated_trees/test/"

os.makedirs(dst_dir, exist_ok=True)

dataset = CophylogenyDataset(src_dir).get_data()

for i, sample in enumerate(tqdm(dataset, total=len(dataset))):
    if len(sample["host_msas"]) == 0 or len(sample["parasite_msas"]) == 0:
        continue
    out_path = os.path.join(dst_dir, f"{i:07d}.pt")
    torch.save(sample, out_path)
