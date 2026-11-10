# Apply the cyclic LoT algorithm to a standardized medication-record data frame.
#
# Required columns: patient_id, drug_name, ordered_date.
# The data frame must already be restricted to the eligible cohort and ordered
# using the intended medication chronology. This function sorts by patient/date.
#
# Parameters: i = 2 previous prescriptions; j = 30 days.

suppressPackageStartupMessages({
  library(dplyr)
})

apply_cyclic_algorithm <- function(data,
                                   patient_col = "patient_id",
                                   drug_col = "drug_name",
                                   date_col = "ordered_date",
                                   i = 2L,
                                   j = 30) {
  required <- c(patient_col, drug_col, date_col)
  missing <- setdiff(required, names(data))
  if (length(missing)) {
    stop("Missing required columns: ", paste(missing, collapse = ", "))
  }
  if (i != 2L || j != 30) {
    warning("The manuscript-specified settings are i=2 and j=30.")
  }

  data %>%
    mutate(.original_row = row_number(),
           .algorithm_date = as.Date(.data[[date_col]]),
           .algorithm_drug = as.character(.data[[drug_col]])) %>%
    arrange(.data[[patient_col]], .algorithm_date, .original_row) %>%
    group_by(.data[[patient_col]]) %>%
    group_modify(function(.x, .y) {
      n <- nrow(.x)
      flags <- integer(n)
      if (n > 0) {
        flags[1] <- 1L
        if (n >= 2) {
          for (k in 2:n) {
            prior_idx <- seq.int(max(1L, k - i), k - 1L)
            same_recent_drug <- !is.na(.x$.algorithm_drug[k]) &&
              any(.x$.algorithm_drug[prior_idx] == .x$.algorithm_drug[k], na.rm = TRUE)
            gap <- as.numeric(.x$.algorithm_date[k] - .x$.algorithm_date[k - 1L])
            flags[k] <- as.integer(!same_recent_drug && !is.na(gap) && gap >= j)
          }
        }
      }
      .x$cyclic_line <- cumsum(flags)
      .x$cyclic_new_line_flag <- flags
      .x
    }) %>%
    ungroup() %>%
    arrange(.original_row) %>%
    select(-.original_row, -.algorithm_date, -.algorithm_drug)
}
