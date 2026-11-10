# Apply the cyclic algorithm to one prepared cohort and export medication-level predictions.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 6L) {
  stop('Usage: Rscript R/run_cyclic.R input.csv output.csv patient_col drug_col date_col i,j')
}
source('R/cyclic_algorithm.R')
input <- read.csv(args[[1]], stringsAsFactors = FALSE)
params <- as.integer(strsplit(args[[6]], ',', fixed = TRUE)[[1]])
if (length(params) != 2L || any(is.na(params))) stop('Invalid cyclic parameters')
result <- apply_cyclic_algorithm(input, patient_col = args[[3]], drug_col = args[[4]],
                                 date_col = args[[5]], i = params[[1]], j = params[[2]])
if (!'line_number' %in% names(result)) stop('Missing benchmark line_number')
output <- data.frame(patient_id = result[[args[[3]]]],
                     true_class = pmin(as.integer(result$line_number), 3L) - 1L,
                     pred_class = pmin(as.integer(result$cyclic_line), 3L) - 1L)
if ('tumour_type' %in% names(result)) output$tumour_type <- result$tumour_type
if (anyNA(output[, c('patient_id', 'true_class', 'pred_class')])) stop('Missing predictions or labels')
dir.create(dirname(args[[2]]), recursive = TRUE, showWarnings = FALSE)
write.csv(output, args[[2]], row.names = FALSE)
