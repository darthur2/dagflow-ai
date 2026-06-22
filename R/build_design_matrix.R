build_design_matrix <- function(target, data, formula_entry) {
  if (!is.data.frame(data)) {
    stop("`data` must be a data frame")
  }
  if (length(target) != 1 || !is.character(target)) {
    stop("`target` must be a single character string")
  }
  if (!is.list(formula_entry) || is.null(formula_entry$predictors)) {
    stop("`formula_entry` must have a `predictors` field")
  }

  predictors <- formula_entry$predictors
  n <- nrow(data)

  cols <- list()
  beta1_init <- numeric(0)

  for (p in predictors) {
    col_name <- p$column
    if (!col_name %in% colnames(data)) {
      stop(sprintf("predictor '%s' for '%s' not found in data", col_name, target))
    }

    if (!is.null(p$reference)) {
      # Categorical predictor — expand to k-1 dummy columns
      categories <- p$categories
      coefs <- as.numeric(p$coefficient)

      if (length(categories) != length(coefs)) {
        stop(sprintf(
          "mismatch: %s has %d categories but %d coefficients for '%s'",
          col_name, length(categories), length(coefs), target
        ))
      }

      values <- data[[col_name]]
      for (i in seq_along(categories)) {
        cat_val <- categories[i]
        dummy_name <- paste0(col_name, "_", cat_val)
        cols[[dummy_name]] <- as.numeric(values == cat_val)
        beta1_init <- c(beta1_init, coefs[i])
      }
    } else {
      # Continuous predictor
      vals <- data[[col_name]]
      if (!is.numeric(vals)) {
        vals <- as.numeric(vals)
      }
      if (any(is.na(vals))) {
        stop(sprintf("NAs in continuous predictor '%s' for '%s'", col_name, target))
      }
      cols[[col_name]] <- vals
      beta1_init <- c(beta1_init, as.numeric(p$coefficient))
    }
  }

  if (length(cols) == 0) {
    X <- matrix(0, nrow = n, ncol = 0)
  } else {
    X <- do.call(cbind, cols)
    storage.mode(X) <- "double"
  }

  list(X = X, beta1_init = beta1_init)
}
