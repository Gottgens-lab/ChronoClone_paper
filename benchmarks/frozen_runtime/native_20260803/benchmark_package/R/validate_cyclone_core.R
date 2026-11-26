#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
  stop(paste(
    "usage: validate_cyclone_core.R INPUT_ABUNDANCE_CSV N_CLUST",
    "OUTPUT_LABELS_CSV OUTPUT_CORRELATION_CSV OUTPUT_DISTANCE_CSV"
  ))
}

input_path <- args[[1]]
n_clust <- as.integer(args[[2]])
output_path <- args[[3]]
correlation_path <- args[[4]]
distance_path <- args[[5]]
if (is.na(n_clust) || n_clust < 2) {
  stop("N_CLUST must be an integer >= 2")
}

x <- as.matrix(read.csv(input_path, row.names = 1, check.names = FALSE))
mat <- t(x)
row_totals <- rowSums(mat)
if (any(row_totals <= 0)) {
  stop("Every time point must have positive total abundance")
}
mat <- sweep(mat, 1, row_totals, "/") * 100
row_sd <- apply(mat, 1, sd)
mat <- (mat - rowMeans(mat)) / row_sd
mat_cor <- cor(mat, method = "pearson", use = "pairwise.complete.obs")
d <- dist(mat_cor, method = "euclidean")
if (any(!is.finite(as.matrix(d)))) {
  stop("Final R-compatible clone distances are non-finite")
}
hc <- hclust(d, method = "ward.D2")
ct <- cutree(hc, k = n_clust)
write.csv(
  data.frame(clone_id = names(ct), label = unname(ct)),
  output_path,
  row.names = FALSE
)
write.csv(mat_cor, correlation_path, row.names = TRUE)
write.csv(as.matrix(d), distance_path, row.names = TRUE)
