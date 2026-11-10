# Patient-level bootstrap for cyclic algorithm performance.

suppressPackageStartupMessages({
  library(dplyr)
  library(purrr)
  library(yardstick)
  library(tibble)
  library(readr)
})

collapse_lot <- function(x) {
  ifelse(is.na(x), NA_integer_, pmin(as.integer(x), 3L))
}

evaluate_cyclic <- function(data,
                            patient_col = "patient_id",
                            truth_col = "line_number",
                            pred_col = "cyclic_line",
                            n_boot = 1000L,
                            seed = 123L) {
  required <- c(patient_col, truth_col, pred_col)
  missing <- setdiff(required, names(data))
  if (length(missing)) stop("Missing columns: ", paste(missing, collapse = ", "))

  eval_data <- data %>%
    transmute(
      patient_id = .data[[patient_col]],
      truth = factor(collapse_lot(.data[[truth_col]]), levels = 1:3),
      estimate = factor(collapse_lot(.data[[pred_col]]), levels = 1:3)
    ) %>%
    filter(!is.na(patient_id), !is.na(truth), !is.na(estimate))

  point <- tibble(
    metric = c("accuracy", "weighted_f1"),
    estimate = c(
      mean(eval_data$truth == eval_data$estimate),
      yardstick::f_meas_vec(eval_data$truth, eval_data$estimate,
                            estimator = "macro_weighted", na_rm = TRUE)
    )
  )

  ids <- unique(eval_data$patient_id)
  if (!length(ids)) stop("No eligible patients for evaluation.")
  set.seed(seed)
  boot <- map_dfr(seq_len(n_boot), function(b) {
    sampled <- sample(ids, size = length(ids), replace = TRUE)
    boot_data <- map_dfr(sampled, function(id) {
      eval_data %>% filter(patient_id == id)
    })
    tibble(
      iteration = b,
      accuracy = mean(boot_data$truth == boot_data$estimate),
      weighted_f1 = yardstick::f_meas_vec(
        boot_data$truth, boot_data$estimate,
        estimator = "macro_weighted", na_rm = TRUE
      )
    )
  })

  summary <- tibble(
    metric = c("accuracy", "weighted_f1"),
    ci_lower = c(
      quantile(boot$accuracy, 0.025, na.rm = TRUE, names = FALSE),
      quantile(boot$weighted_f1, 0.025, na.rm = TRUE, names = FALSE)
    ),
    ci_upper = c(
      quantile(boot$accuracy, 0.975, na.rm = TRUE, names = FALSE),
      quantile(boot$weighted_f1, 0.975, na.rm = TRUE, names = FALSE)
    )
  ) %>%
    left_join(point, by = "metric") %>%
    select(metric, estimate, ci_lower, ci_upper)

  list(summary = summary, bootstrap = boot)
}
