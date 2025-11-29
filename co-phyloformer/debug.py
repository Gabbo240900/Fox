import os, torch, copy
from data import CophylogenyDataset

#dataset_dir = "/Users/gabriele/Co-phyloformer/generate_treeducken/generated_trees/Datasets"
dataset_dir = "/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/Dataset_final/"
dataset = CophylogenyDataset(dataset_dir).get_data()

# 🔧 Flatten recursively if needed
def flatten(data):
    flat = []
    for x in data:
        if isinstance(x, list):
            flat.extend(flatten(x))
        else:
            flat.append(x)
    return flat

dataset = flatten(list(dataset))
print(f"Flattened dataset to {len(dataset)} total samples")

# 💡 Split indices
total = len(dataset)
indices = torch.randperm(total)
split = int(total * 0.8)
train_indices, val_indices = indices[:split], indices[split:]

# 🧠 Create *independent* deep-copied lists
train_subset = [copy.deepcopy(dataset[i]) for i in train_indices]
val_subset   = [copy.deepcopy(dataset[i]) for i in val_indices]

print(f"Total: {len(dataset)}")
print(f"Train: {len(train_subset)}, Val: {len(val_subset)}")

# 🔬 Check overlap of references
train_ids = set(id(x) for x in train_subset[:100])
val_ids = set(id(x) for x in val_subset[:100])
print(f"Overlap among first 100 samples: {len(train_ids & val_ids)}")

# ✅ Check if train/val lists themselves are distinct
print("List object IDs:")
print("  train list:", id(train_subset))
print("  val list:  ", id(val_subset))