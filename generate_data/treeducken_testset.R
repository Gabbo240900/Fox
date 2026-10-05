suppressMessages({
  library(treeducken)
  library(ape)
})

args <- commandArgs(trailingOnly = TRUE)
opt <- list(out = "test_data/treeducken/Datasets", n = "100", seed = "1",
            min_host = "15", max_host = "50", min_symb = "10", max_symb = "80",
            max_tries = "200")
for (a in args) {
  kv <- strsplit(a, "=", fixed = TRUE)[[1]]
  opt[[kv[1]]] <- kv[2]
}
n_target  <- as.integer(opt$n)
min_host  <- as.integer(opt$min_host);  max_host <- as.integer(opt$max_host)
min_symb  <- as.integer(opt$min_symb);  max_symb <- as.integer(opt$max_symb)
max_tries <- as.integer(opt$max_tries)
set.seed(as.integer(opt$seed))
dir.create(opt$out, showWarnings = FALSE, recursive = TRUE)

time_grid <- c(1.5, 2, 2.5, 3, 3.5)   # same ages as generate_trees.py

# Same rate ranges as Fox's training data (generate_trees.py, AsymmeTree):
#   host birth 0.5-1.2: in AsymmeTree every host speciation also splits the symbiont,
#     so it is the cospeciation rate here, and there is no host speciation without it (hbr = 0);
#   duplication 0.2-0.4, HGT 0.05-0.3, loss 0.2-0.4.
#   Host death is 0: AsymmeTree's host tree (species_tree_n_age) keeps no extinct hosts,
#     so no symbiont dies with its host there; here every symbiont of an extinct host
#     would become a loss.
draw_params <- function() {
  list(cosp = runif(1, 0.5, 1.2),    # cospeciation
       hbr  = 0,                     # host speciation without the symbiont
       hdr  = 0,                     # host extinction
       sbr  = runif(1, 0.2, 0.4),    # symbiont speciation without the host (duplication)
       her  = runif(1, 0.05, 0.3),   # host switch
       sdr  = runif(1, 0.2, 0.4),    # symbiont extinction (loss)
       age  = sample(time_grid, 1))
}

# Keep only lineages alive at the present (treeducken names extinct tips X<n>).
prune_extinct <- function(tr) {
  dead <- grep("^X", tr$tip.label, value = TRUE)
  if (length(dead) > 0) {
    if (length(dead) >= Ntip(tr) - 1) return(NULL)
    tr <- drop.tip(tr, dead)
  }
  tr$node.label <- NULL
  tr$root.edge <- NULL
  tr
}

# Event counts with the AsymmeTree convention, on the full symbiont tree (extinct tips kept):
#   - a branching node counts as its event (CSP = Speciation, SSP = Duplication,
#     SHS/SHE = HGT) only if both of its sides have a descendant alive today;
#   - every extinct tip counts as one Loss; the node above it is not counted.
# So Total = all symbiont leaves - 1 and Total - Loss = extant symbiont leaves - 1.
# treeducken logs events with times, so each node is matched to its event by time.
count_events <- function(sim, age) {
  branch_types <- c("CSP", "SSP", "SHS", "SHE")
  tr <- symb_tree(sim)
  ev <- event_history(sim)
  ev <- ev[ev$Event_Type %in% branch_types, ]
  n_tip <- Ntip(tr)
  extant <- !grepl("^X", tr$tip.label)
  depth <- node.depth.edgelength(tr)
  inode <- n_tip + seq_len(tr$Nnode)
  node_time <- age - (max(depth[seq_len(n_tip)][extant]) - depth[inode])
  key_node <- round(node_time, 6)
  key_ev <- round(ev$Event_Time, 6)
  if (!identical(sort(key_node), sort(key_ev))) return(NULL)
  type <- character(length(inode))
  for (k in unique(key_node)) {
    tps <- unique(ev$Event_Type[key_ev == k])
    if (length(tps) != 1) return(NULL)   # two kinds of event at the same time: can't tell them apart
    type[key_node == k] <- tps
  }
  alive <- c(extant, logical(tr$Nnode))   # node has an extant descendant
  po <- tr$edge[postorder(tr), , drop = FALSE]
  for (i in seq_len(nrow(po))) alive[po[i, 1]] <- alive[po[i, 1]] || alive[po[i, 2]]
  surviving <- vapply(inode, function(n) all(alive[tr$edge[tr$edge[, 1] == n, 2]]), logical(1))
  t <- table(factor(type[surviving], levels = branch_types))
  c(Speciation  = t[["CSP"]],
    HGT         = t[["SHS"]] + t[["SHE"]],
    Loss        = sum(!extant),
    Duplication = t[["SSP"]],
    all_leaves  = n_tip)
}

simulate_one <- function(p) {
  sim <- tryCatch(
    sim_cophyBD(hbr = p$hbr, hdr = p$hdr, sbr = p$sbr, cosp_rate = p$cosp,
                sdr = p$sdr, host_exp_rate = p$her, time_to_sim = p$age,
                numbsim = 1, hs_mode = "switch")[[1]],
    error = function(e) NULL)
  if (is.null(sim)) return(NULL)
  h <- prune_extinct(host_tree(sim))
  s <- prune_extinct(symb_tree(sim))
  if (is.null(h) || is.null(s)) return(NULL)
  if (Ntip(h) < min_host || Ntip(h) > max_host) return(NULL)
  if (Ntip(s) < min_symb || Ntip(s) > max_symb) return(NULL)

  # association_mat: rows = extant hosts, cols = extant symbionts
  am <- association_mat(sim)
  idx <- which(am == 1, arr.ind = TRUE)
  assoc <- data.frame(symb = colnames(am)[idx[, "col"]],
                      host = rownames(am)[idx[, "row"]])
  assoc <- assoc[assoc$symb %in% s$tip.label & assoc$host %in% h$tip.label, ]
  if (!setequal(unique(assoc$symb), s$tip.label)) return(NULL)  # every symbiont needs a host

  counts <- count_events(sim, p$age)
  if (is.null(counts) || sum(counts[c("Speciation", "HGT", "Loss", "Duplication")]) == 0) return(NULL)

  # S<n> -> P<n> so the names match the rest of the pipeline (H<n> / P<n>)
  s$tip.label <- sub("^S", "P", s$tip.label)
  assoc$symb  <- sub("^S", "P", assoc$symb)

  list(h = h, s = s, assoc = assoc, counts = counts[c("Speciation", "HGT", "Loss", "Duplication")],
       all_leaves = counts[["all_leaves"]], multi = sum(table(assoc$symb) > 1))
}

write_tgl <- function(path, r, p) {
  freqs <- r$counts / sum(r$counts)
  nwk <- function(tr) sub(";$", "", write.tree(tr))
  lines <- c(
    "#NEXUS",
    "BEGIN HOST;",
    paste0("\tTREE * Host1 = ", nwk(r$h), ";"),
    "ENDBLOCK;", "",
    "BEGIN PARASITE;",
    paste0("\tTREE * Para1 = ", nwk(r$s), ";"),
    "ENDBLOCK;", "",
    "BEGIN DISTRIBUTION;",
    "\tRANGE",
    paste0("\t\t", r$assoc$symb, ": ", r$assoc$host),
    "END;", "",
    paste0(names(r$counts), ": ", r$counts),
    paste0("Total_Events: ", sum(r$counts)),
    sprintf("Speciation_freq: %.4f", freqs[["Speciation"]]),
    sprintf("Loss_freq: %.4f", freqs[["Loss"]]),
    sprintf("HGT_freq: %.4f", freqs[["HGT"]]),
    sprintf("Duplication_freq: %.4f", freqs[["Duplication"]]),
    paste0("Host_num_leaves: ", Ntip(r$h)),
    paste0("Symbiont_num_leaves: ", r$all_leaves),   # incl. extinct, as in generate_trees.py
    paste0("Symbiont_num_extant: ", Ntip(r$s)),
    paste0("Multi_host_symbionts: ", r$multi),
    sprintf("Rates: cosp=%.4f hbr=%.4f hdr=%.4f sbr=%.4f sdr=%.4f her=%.4f",
            p$cosp, p$hbr, p$hdr, p$sbr, p$sdr, p$her),
    paste0("Sim_time: ", p$age)
  )
  writeLines(lines, path)
}

made <- 0; tries <- 0; rejected <- 0
while (made < n_target) {
  p <- draw_params()
  r <- NULL
  for (k in seq_len(max_tries)) {   # retry the same rates until the size fits
    tries <- tries + 1
    r <- simulate_one(p)
    if (!is.null(r)) break
    rejected <- rejected + 1
  }
  if (is.null(r)) next              # these rates never give a usable size: draw new ones
  made <- made + 1
  write_tgl(file.path(opt$out, paste0("Dataset", made, ".tgl")), r, p)
}
cat(sprintf("Wrote %d datasets to %s (%d simulations, %d rejected)\n",
            made, opt$out, tries, rejected))
