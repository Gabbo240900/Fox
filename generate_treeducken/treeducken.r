library(treeducken)
library(ape)

args <- commandArgs(trailingOnly = TRUE)

h_lambda   <- as.numeric(args[1])
h_mu       <- as.numeric(args[2])
c_lambda   <- as.numeric(args[3])
s_lambda   <- as.numeric(args[4])
s_mu       <- as.numeric(args[5])
s_her      <- as.numeric(args[6])
time_to_sim <- as.numeric(args[7])
sim_index   <- as.integer(args[8])

host_symb_sets <- sim_cophyBD(
  hbr = h_lambda,
  hdr = h_mu,
  sbr = s_lambda,
  cosp_rate = c_lambda,
  sdr = s_mu,
  host_exp_rate = s_her,
  time_to_sim = time_to_sim,
  numbsim = 1,
  hs_mode = 'both'
)

# Output folders
dir.create("/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/host_tree", showWarnings = FALSE)
dir.create("/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/symb_tree", showWarnings = FALSE)
dir.create("/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/associations", showWarnings = FALSE)
dir.create("/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/summaries", showWarnings = FALSE)

# dir.create("/Users/gabriele/Co-phyloformer/generate_treeducken/generated_trees/host_tree", showWarnings = FALSE)
# dir.create("/Users/gabriele/Co-phyloformer/generate_treeducken/generated_trees/symb_tree", showWarnings = FALSE)
# dir.create("/Users/gabriele/Co-phyloformer/generate_treeducken/generated_trees/associations", showWarnings = FALSE)
# dir.create("/Users/gabriele/Co-phyloformer/generate_treeducken/generated_trees/summaries", showWarnings = FALSE)

host_tree <- host_tree(host_symb_sets[[1]])
symb_tree <- symb_tree(host_symb_sets[[1]])
# write.tree(host_tree, file = file.path("/Users/gabriele/Co-phyloformer/generate_treeducken/generated_trees/host_tree", paste0("host_tree_", sim_index, ".nwk")))
# write.tree(symb_tree, file = file.path("/Users/gabriele/Co-phyloformer/generate_treeducken/generated_trees/symb_tree", paste0("symb_tree_", sim_index, ".nwk")))

write.tree(host_tree, file = file.path("/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/host_tree", paste0("host_tree_", sim_index, ".nwk")))
write.tree(symb_tree, file = file.path("/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/symb_tree", paste0("symb_tree_", sim_index, ".nwk")))


assoc_mat <- association_mat(host_symb_sets[[1]])
assoc_pairs <- which(assoc_mat == 1, arr.ind = TRUE)
assoc_df <- data.frame(
  symbiont = colnames(assoc_mat)[assoc_pairs[, "col"]],
  host = rownames(assoc_mat)[assoc_pairs[, "row"]]
)
write.csv(assoc_df, file = file.path("/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/associations", paste0("association_", sim_index, ".csv")), row.names = FALSE)
# write.csv(assoc_df, file = file.path("/Users/gabriele/Co-phyloformer/generate_treeducken/generated_trees/associations", paste0("association_", sim_index, ".csv")), row.names = FALSE)
summary_list <- summarize_1cophy(host_symb_sets, 1)
write.csv(summary_list, file = file.path("/lustre/fsn1/projects/rech/vcu/commun/Co-Phyloformer/generate_treeducken/generated_trees/summaries", paste0("summary_", sim_index, ".csv")), row.names = TRUE)
# write.csv(summary_list, file = file.path("/Users/gabriele/Co-phyloformer/generate_treeducken/generated_trees/summaries", paste0("summary_", sim_index, ".csv")), row.names = TRUE)